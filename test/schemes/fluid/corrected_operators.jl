@trixi_testset "Corrected fluid operators" begin
    TP = TrixiParticles

    function edac(ic; correction=nothing, reduction=true, viscosity=nothing,
                  buffer_size=nothing)
        return EntropicallyDampedSPHSystem(ic;
                                           smoothing_kernel=WendlandC6Kernel{ndims(ic)}(),
                                           smoothing_length=2ic.particle_spacing,
                                           sound_speed=10.0,
                                           density_calculator=ContinuityDensity(),
                                           pressure_acceleration=nothing, correction,
                                           average_pressure_reduction=reduction, viscosity,
                                           buffer_size)
    end

    function discretize(systems...; interaction_matrix=nothing)
        semi = Semidiscretization(systems...; neighborhood_search=nothing,
                                  interaction_matrix,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1); reset_threads=false)
        return (; semi=ode.p.semi, v=Array(ode.u0.x[1]), u=Array(ode.u0.x[2]))
    end

    function acceleration(setup)
        (; semi, v, u) = setup
        dv = zero(v)
        TP.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0)
        result = zeros(ndims(first(semi.systems)), 0)
        mass = Float64[]
        GC.@preserve dv begin
            for system in semi.systems
                local_dv = TP.wrap_v(dv, system, semi)
                indices = TP.each_active_particle(system)
                result = hcat(result, Array(local_dv[1:ndims(system), indices]))
                append!(mass, system.mass[indices])
            end
        end
        return result, mass
    end

    @testset "Constant pressure reference retains affine acceleration and pressure gauge" begin
        dx = 1 / 20
        ic = RectangularShape(dx, (20, 20), (0.0, 0.0); density=1.0,
                              pressure=x -> 2 + 3x[1] - 2x[2])
        setup = discretize(edac(ic))
        first_acceleration, mass = acceleration(setup)
        system = first(setup.semi.systems)
        @test all(p -> p ≈ sum(ic.pressure) / length(ic.pressure),
                  system.cache.pressure_average)
        interior = [i
                    for i in axes(ic.coordinates, 2)
                    if minimum(min.(ic.coordinates[:, i], 1 .- ic.coordinates[:, i])) > 0.3]
        @test all(i -> isapprox(first_acceleration[:, i], [-3.0, 2.0]; rtol=0.005),
                  interior)
        @test norm(first_acceleration * mass) < 1e-12
        GC.@preserve setup begin
            local_v = TP.wrap_v(setup.v, system, setup.semi)
            local_v[3, :] .+= 1000
        end
        shifted, _ = acceleration(setup)
        @test shifted≈first_acceleration atol=2e-10 rtol=2e-10
    end

    @testset "Common reference across split fluids" begin
        dx = 0.1
        pressure(x) = 2 + 3x[1] - 2x[2]
        whole = RectangularShape(dx, (10, 6), (0.0, 0.0); density=1.0, pressure)
        left = RectangularShape(dx, (4, 6), (0.0, 0.0); density=1.0, pressure)
        right = RectangularShape(dx, (6, 6), (0.4, 0.0); density=1.0, pressure)
        unified = discretize(edac(whole))
        split = discretize(edac(left), edac(right))
        whole_acceleration, _ = acceleration(unified)
        split_acceleration, masses = acceleration(split)
        @test norm(split_acceleration * masses) < 2e-12
        for system in split.semi.systems
            @test all(p -> p ≈ sum(whole.pressure) / length(whole.pressure),
                      system.cache.pressure_average)
        end
        # Match physical points instead of depending on concatenation/shape ordering.
        coordinates = hcat(left.coordinates, right.coordinates)
        for i in axes(coordinates, 2)
            j = argmin([norm(whole.coordinates[:, j] - coordinates[:, i])
                        for j in axes(whole.coordinates, 2)])
            @test split_acceleration[:, i] ≈ whole_acceleration[:, j] atol=2e-12
        end
    end

    @testset "Reference topology and inactive particles" begin
        patches = [RectangularShape(0.1, (2, 2), (i * 0.2, 0.0);
                                    density=1.0, pressure=Float64(i)) for i in 1:4]
        interaction = Matrix{Bool}(I, 4, 4)
        interaction[1, 2] = true
        interaction[2, 3] = true
        setup = discretize(edac.(patches)...; interaction_matrix=interaction)
        TP.update_systems_and_nhs(setup.v, setup.u, setup.semi, 0.0)
        @test all(s -> all(==(2.0), s.cache.pressure_average), setup.semi.systems[1:3])
        @test all(==(4.0), setup.semi.systems[4].cache.pressure_average)

        buffered = discretize(edac(patches[1]; buffer_size=3))
        system = first(buffered.semi.systems)
        GC.@preserve buffered begin
            v = TP.wrap_v(buffered.v, system, buffered.semi)
            u = TP.wrap_u(buffered.u, system, buffered.semi)
            v[3, :] .= NaN
            v[3, 1:4] .= [1.0, 2.0, 3.0, 4.0]
            TP.deactivate_particle!(system, 2, v, u)
            TP.update_system_buffer!(system.buffer)
            TP.update_average_pressure!(system, Val(true), buffered.v, buffered.u,
                                        buffered.semi)
            @test all(p -> p ≈ 8 / 3, system.cache.pressure_average)
            for i in TP.each_active_particle(system)
                TP.deactivate_particle!(system, i, v, u)
            end
            TP.update_system_buffer!(system.buffer)
            TP.update_average_pressure!(system, Val(true), buffered.v, buffered.u,
                                        buffered.semi)
            @test all(iszero, system.cache.pressure_average)
        end
    end

    @testset "Reciprocal pressure offset with one enabled endpoint and mixed schemes" begin
        ic = RectangularShape(0.1, (3, 3), (0.0, 0.0); density=1.0, pressure=5.0)
        a = edac(ic)
        b = edac(ic; reduction=false)
        equation = StateEquationCole(; reference_density=1.0, sound_speed=10.0, exponent=1)
        c = WeaklyCompressibleSPHSystem(ic; smoothing_kernel=WendlandC6Kernel{2}(),
                                        smoothing_length=0.2,
                                        density_calculator=ContinuityDensity(),
                                        state_equation=equation)
        fill!(a.cache.pressure_average, 5.0)
        @test TP.pair_pressure_offset(a, b, 1, 2) == TP.pair_pressure_offset(b, a, 2, 1) ==
              5
        @test TP.pair_pressure_offset(a, c, 1, 2) == TP.pair_pressure_offset(c, a, 2, 1) ==
              5
        @test iszero(TP.pair_pressure_offset(b, c, 1, 2))

        # Exercise the actual WCSPH and EDAC momentum paths, not just offset lookup.
        other = RectangularShape(0.1, (3, 3), (0.3, 0.0); density=1.02)
        c = WeaklyCompressibleSPHSystem(other; smoothing_kernel=WendlandC6Kernel{2}(),
                                        smoothing_length=0.2,
                                        density_calculator=ContinuityDensity(),
                                        state_equation=equation)
        setup = discretize(a, c)
        dv, mass = acceleration(setup)
        @test norm(dv * mass) < 1e-12
        setup = discretize(edac(ic), edac(other; reduction=false))
        dv, mass = acceleration(setup)
        @test norm(dv * mass) < 1e-12
    end

    @testset "Viscous momentum with asymmetric gradients" begin
        for dim in (2, 3),
            correction in (nothing, KernelCorrection(), GradientCorrection(),
            BlendedGradientCorrection(0.5), MixedKernelGradientCorrection()),
            viscosity in (ViscosityAdami(nu=0.01), ViscosityMorris(nu=0.01),
            ArtificialViscosityMonaghan(alpha=0.1))
            shape = RectangularShape(0.1, ntuple(_ -> 4, dim), ntuple(_ -> 0.0, dim);
                                     density=1.0)
            coordinates = copy(shape.coordinates)
            for i in eachindex(coordinates)
                coordinates[i] += 0.008sin(i)
            end
            count = size(coordinates, 2)
            mass = shape.mass .* [1 + 0.2sin(i) for i in 1:count]
            density = [1 + 0.1cos(i) for i in 1:count]
            velocity = [sin(i + 0.3j) for i in 1:dim, j in 1:count]
            ic = InitialCondition(; coordinates, mass, density, velocity,
                                  pressure=zeros(count), particle_spacing=0.1)
            setup = discretize(edac(ic; correction, viscosity, reduction=false))
            dv, masses = acceleration(setup)
            forces = dv .* masses'
            scale = max(1.0, sum(abs, forces))
            @test norm(vec(sum(forces; dims=2))) < 5e-13scale
            @test all(isfinite, dv)
        end
    end

    @testset "Viscous pairs with only one corrected endpoint" begin
        left = RectangularShape(0.1, (3, 3), (0.0, 0.0); density=1.0,
                                velocity=x -> (x[2], -x[1]))
        right = RectangularShape(0.1, (3, 3), (0.3, 0.0); density=1.0,
                                 velocity=x -> (-x[1], x[2]))
        a = edac(left; viscosity=ViscosityAdami(nu=0.01), reduction=false)
        b = edac(right; correction=MixedKernelGradientCorrection(),
                 viscosity=ViscosityAdami(nu=0.02), reduction=false)
        setup = discretize(a, b)
        dv, mass = acceleration(setup)
        @test norm(dv * mass) < 1e-12
        @test all(isfinite, dv)
    end

    @testset "Quadratic viscous field in the bulk" begin
        dx = 1 / 20
        ic = RectangularShape(dx, (20, 20), (0.0, 0.0); density=1.0,
                              velocity=x -> (x[1]^2 + x[2]^2, x[1]^2 - x[2]^2))
        interior = [i
                    for i in axes(ic.coordinates, 2)
                    if minimum(min.(ic.coordinates[:, i], 1 .- ic.coordinates[:, i])) > 8dx]
        for correction in (nothing, GradientCorrection(), MixedKernelGradientCorrection()),
            viscosity in (ViscosityAdami(nu=0.01), ViscosityMorris(nu=0.01))
            setup = discretize(edac(ic; correction, viscosity, reduction=false))
            dv, _ = acceleration(setup)
            # nu * Laplacian(v) = (4nu, 0); allow the model's epsilon regularization.
            @test all(i -> norm(dv[:, i] - [0.04, 0.0]) < 0.03 * 0.04, interior)
        end
    end
end
