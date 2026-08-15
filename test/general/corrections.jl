@trixi_testset "Correction Consistency" begin
    function correction_setup(correction=nothing; n=9, perturbation=false,
                              density_calculator=ContinuityDensity(), edac=false,
                              density_correction=nothing, gradient_correction=nothing,
                              pressure_acceleration=:default,
                              velocity=(pos -> SVector(pos[1], pos[2])))
        particle_spacing = 1.0 / n
        smoothing_length = 2.0 * particle_spacing
        smoothing_kernel = WendlandC6Kernel{2}()
        fluid = RectangularShape(particle_spacing, (n, n), (0.0, 0.0);
                                 density=1000.0, velocity,
                                 coordinates_perturbation=perturbation ? 0.1 : nothing)

        if edac
            if pressure_acceleration === :default
                system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                     smoothing_length, sound_speed=10.0,
                                                     density_calculator, correction,
                                                     density_correction,
                                                     gradient_correction)
            else
                system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                     smoothing_length, sound_speed=10.0,
                                                     density_calculator, correction,
                                                     density_correction,
                                                     gradient_correction,
                                                     pressure_acceleration)
            end
        else
            state_equation = StateEquationCole(; sound_speed=10.0,
                                               reference_density=1000.0, exponent=1)
            if pressure_acceleration === :default
                system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                     smoothing_length, density_calculator,
                                                     state_equation, correction,
                                                     density_correction,
                                                     gradient_correction)
            else
                system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                     smoothing_length, density_calculator,
                                                     state_equation, correction,
                                                     density_correction,
                                                     gradient_correction,
                                                     pressure_acceleration)
            end
        end

        semi = Semidiscretization(system; parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        semi = ode.p.semi
        system = first(semi.systems)

        return (; system, semi, v_ode, u_ode, particle_spacing)
    end

    function fill_correction_cache!(system, value)
        for name in (:kernel_correction_coefficient, :dw_gamma, :correction_matrix)
            hasproperty(system.cache, name) || continue
            fill!(getproperty(system.cache, name), value)
        end
        return system
    end

    function update_correction!(setup)
        (; system, semi, v_ode, u_ode) = setup
        fill_correction_cache!(system, NaN)
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
        return setup
    end

    function correction_moments(setup; field=(pos -> 1.0))
        (; system, semi, v_ode, u_ode) = setup
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        coordinates = Array(TrixiParticles.current_coordinates(u, system))
        values = [field(SVector{2}(view(coordinates, :, particle)))
                  for particle in TrixiParticles.eachparticle(system)]
        n_particles = TrixiParticles.nparticles(system)

        zeroth_gradient_moment = zeros(2, n_particles)
        first_gradient_moment = zeros(2, 2, n_particles)
        direct_gradient = zeros(2, n_particles)
        difference_gradient = zeros(2, n_particles)

        GC.@preserve v_ode u_ode begin
            TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                                  semi) do particle, neighbor, pos_diff,
                                                           distance
                pos_diff_ = SVector(pos_diff)
                volume = TrixiParticles.hydrodynamic_mass(system, neighbor) /
                         TrixiParticles.current_density(v, system, neighbor)
                gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff_, distance,
                                                                particle)
                neighbor_offset = -pos_diff_

                for i in 1:2
                    zeroth_gradient_moment[i, particle] += volume * gradient[i]
                    direct_gradient[i, particle] += volume * values[neighbor] * gradient[i]
                    difference_gradient[i,
                                        particle] += volume *
                                                     (values[neighbor] - values[particle]) *
                                                     gradient[i]
                    for j in 1:2
                        first_gradient_moment[i, j,
                                              particle] += volume * gradient[i] *
                                                           neighbor_offset[j]
                    end
                end
            end
        end

        return (; zeroth_gradient_moment, first_gradient_moment, direct_gradient,
                difference_gradient)
    end

    function corner_particle(system)
        coordinates = TrixiParticles.initial_coordinates(system)
        return argmin(eachindex(axes(coordinates, 2))) do particle
            coordinates[1, particle] + coordinates[2, particle]
        end
    end

    @testset "Cache lifecycle" begin
        corrections = (KernelCorrection(), GradientCorrection(),
                       BlendedGradientCorrection(0.4), MixedKernelGradientCorrection())

        for edac in (false, true), correction in corrections
            setup = correction_setup(correction; edac)
            update_correction!(setup)

            for name in (:kernel_correction_coefficient, :dw_gamma, :correction_matrix)
                hasproperty(setup.system.cache, name) || continue
                @test all(isfinite, getproperty(setup.system.cache, name))
            end
        end

        setup = correction_setup(ShepardKernelCorrection();
                                 density_calculator=SummationDensity())
        update_correction!(setup)
        density = TrixiParticles.current_density(TrixiParticles.wrap_v(setup.v_ode,
                                                                       setup.system,
                                                                       setup.semi),
                                                 setup.system)
        @test setup.system.pressure ≈ setup.system.state_equation.(density)

        setup_edac = correction_setup(ShepardKernelCorrection();
                                      density_calculator=SummationDensity(), edac=true)
        update_correction!(setup_edac)
        @test all(isfinite, setup_edac.system.cache.kernel_correction_coefficient)
        @test all(isfinite, setup_edac.system.cache.density)

        for edac in (false, true)
            combined = correction_setup(; density_calculator=SummationDensity(), edac,
                                        density_correction=ShepardKernelCorrection(),
                                        gradient_correction=MixedKernelGradientCorrection())
            update_correction!(combined)
            @test combined.system.correction isa CorrectionConfiguration
            @test all(isfinite, combined.system.cache.kernel_correction_coefficient)
            @test all(isfinite, combined.system.cache.dw_gamma)
            @test all(isfinite, combined.system.cache.correction_matrix)
        end

        @test_throws ArgumentError correction_setup(GradientCorrection();
                                                    gradient_correction=GradientCorrection())
        @test_throws ArgumentError correction_setup(;
                                                    density_calculator=ContinuityDensity(),
                                                    density_correction=ShepardKernelCorrection())
        @test_throws ArgumentError CorrectionConfiguration(; density=GradientCorrection())
        @test_throws ArgumentError CorrectionConfiguration(;
                                                           gradient=ShepardKernelCorrection())
        @test_throws ArgumentError BlendedGradientCorrection(-0.1)
        @test_throws ArgumentError BlendedGradientCorrection(1.1)

        iisph_particles = RectangularShape(0.1, (2, 2), (0.0, 0.0); density=1000.0)
        @test_throws ArgumentError ImplicitIncompressibleSPHSystem(iisph_particles;
                                                                   smoothing_kernel=WendlandC6Kernel{2}(),
                                                                   smoothing_length=0.2,
                                                                   reference_density=1000.0,
                                                                   time_step=0.01,
                                                                   correction=GradientCorrection())

        coefficients = ones(TrixiParticles.nparticles(setup.system))
        coefficients[1] = 0.0
        coefficients[2] = NaN
        TrixiParticles.sanitize_kernel_correction_coefficient!(coefficients, setup.system,
                                                               setup.semi)
        @test coefficients[1:2] == ones(2)

        n = 5
        particle_spacing = 1.0 / n
        smoothing_kernel = WendlandC6Kernel{2}()
        particles = RectangularShape(particle_spacing, (n, n), (0.0, 0.0);
                                     density=1000.0)
        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=1000.0, exponent=1)
        boundary_model = BoundaryModelDummyParticles(particles.density, particles.mass,
                                                     SummationDensity(), smoothing_kernel,
                                                     2particle_spacing; state_equation,
                                                     correction=ShepardKernelCorrection())
        boundary = WallBoundarySystem(particles, boundary_model)
        semi = Semidiscretization(boundary; parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        boundary = first(ode.p.semi.systems)
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)
        @test boundary.boundary_model.pressure ≈
              state_equation.(boundary.boundary_model.cache.density)

        combined_boundary_model = BoundaryModelDummyParticles(particles.density,
                                                              particles.mass,
                                                              SummationDensity(),
                                                              smoothing_kernel,
                                                              2particle_spacing;
                                                              state_equation,
                                                              density_correction=ShepardKernelCorrection(),
                                                              gradient_correction=MixedKernelGradientCorrection())
        combined_boundary = WallBoundarySystem(particles, combined_boundary_model)
        combined_semi = Semidiscretization(combined_boundary;
                                           parallelization_backend=SerialBackend())
        combined_ode = semidiscretize(combined_semi, (0.0, 1.0); reset_threads=false)
        combined_v = Array(combined_ode.u0.x[1])
        combined_u = Array(combined_ode.u0.x[2])
        combined_boundary = first(combined_ode.p.semi.systems)
        TrixiParticles.update_systems_and_nhs(combined_v, combined_u, combined_ode.p.semi,
                                              0.0)
        @test combined_boundary.boundary_model.correction isa CorrectionConfiguration
        @test all(isfinite,
                  combined_boundary.boundary_model.cache.kernel_correction_coefficient)
        @test all(isfinite, combined_boundary.boundary_model.cache.dw_gamma)
        @test all(isfinite, combined_boundary.boundary_model.cache.correction_matrix)

        density32 = fill(1000.0f0, 4)
        mass32 = fill(10.0f0, 4)
        boundary32 = BoundaryModelDummyParticles(density32, mass32, SummationDensity(),
                                                 WendlandC6Kernel{2}(), 0.2f0;
                                                 state_equation,
                                                 correction=MixedKernelGradientCorrection())
        @test eltype(boundary32.cache.dw_gamma) == Float32
        @test eltype(boundary32.cache.correction_matrix) == Float32
    end

    @testset "Discrete moments and polynomial reproduction" begin
        identity_matrix = Matrix{Float64}(I, 2, 2)
        linear_field(pos) = 2.0 + 3.0 * pos[1] - 2.0 * pos[2]
        exact_gradient = [3.0, -2.0]

        for perturbation in (false, true)
            raw_setup = update_correction!(correction_setup(nothing; perturbation))
            raw_moments = correction_moments(raw_setup; field=linear_field)

            kernel_setup = update_correction!(correction_setup(KernelCorrection();
                                                               perturbation))
            kernel_moments = correction_moments(kernel_setup; field=linear_field)
            @test maximum(abs, kernel_moments.zeroth_gradient_moment) < 2e-12

            gradient_setup = update_correction!(correction_setup(GradientCorrection();
                                                                 perturbation))
            gradient_moments = correction_moments(gradient_setup; field=linear_field)
            @test maximum(particle -> norm(gradient_moments.first_gradient_moment[:, :,
                                                                                  particle] -
                                           identity_matrix),
                          TrixiParticles.eachparticle(gradient_setup.system)) < 2e-12
            @test maximum(particle -> norm(gradient_moments.difference_gradient[:,
                                                                                particle] -
                                           exact_gradient),
                          TrixiParticles.eachparticle(gradient_setup.system)) < 5e-12

            mixed_setup = update_correction!(correction_setup(MixedKernelGradientCorrection();
                                                              perturbation))
            mixed_moments = correction_moments(mixed_setup; field=linear_field)
            @test maximum(abs, mixed_moments.zeroth_gradient_moment) < 3e-12
            @test maximum(particle -> norm(mixed_moments.first_gradient_moment[:, :,
                                                                               particle] -
                                           identity_matrix),
                          TrixiParticles.eachparticle(mixed_setup.system)) < 3e-12
            @test maximum(particle -> norm(mixed_moments.direct_gradient[:, particle] -
                                           exact_gradient),
                          TrixiParticles.eachparticle(mixed_setup.system)) < 1e-11

            blending_factor = 0.4
            blended_setup = update_correction!(correction_setup(BlendedGradientCorrection(blending_factor);
                                                                perturbation))
            blended_moments = correction_moments(blended_setup; field=linear_field)
            expected = (1 - blending_factor) * raw_moments.first_gradient_moment
            for particle in TrixiParticles.eachparticle(blended_setup.system)
                expected[:, :, particle] .+= blending_factor * identity_matrix
            end
            @test maximum(abs, blended_moments.first_gradient_moment - expected) < 2e-12

            corner = corner_particle(raw_setup.system)
            @test norm(raw_moments.first_gradient_moment[:, :, corner] - identity_matrix) >
                  1e-2
        end

        combined_setup = update_correction!(correction_setup(;
                                                             density_calculator=SummationDensity(),
                                                             density_correction=ShepardKernelCorrection(),
                                                             gradient_correction=MixedKernelGradientCorrection()))
        combined_moments = correction_moments(combined_setup; field=linear_field)
        @test maximum(abs, combined_moments.zeroth_gradient_moment) < 3e-12
        @test maximum(particle -> norm(combined_moments.first_gradient_moment[:, :,
                                                                              particle] -
                                       identity_matrix),
                      TrixiParticles.eachparticle(combined_setup.system)) < 3e-12

        particle_spacing = 0.25
        smoothing_kernel = WendlandC6Kernel{3}()
        particles = RectangularShape(particle_spacing, (4, 4, 4), (0.0, 0.0, 0.0);
                                     density=1000.0)
        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=1000.0, exponent=1)
        system = WeaklyCompressibleSPHSystem(particles; smoothing_kernel,
                                             smoothing_length=2particle_spacing,
                                             density_calculator=ContinuityDensity(),
                                             state_equation,
                                             correction=GradientCorrection())
        semi = Semidiscretization(system; parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        system = first(ode.p.semi.systems)
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)
        v = TrixiParticles.wrap_v(v_ode, system, ode.p.semi)
        u = TrixiParticles.wrap_u(u_ode, system, ode.p.semi)
        coordinates = Array(TrixiParticles.current_coordinates(u, system))
        first_moment = zeros(3, 3)
        GC.@preserve v_ode u_ode begin
            TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                                  ode.p.semi;
                                                  points=1:1) do particle,
                                                                 neighbor,
                                                                 pos_diff,
                                                                 distance
                volume = TrixiParticles.hydrodynamic_mass(system, neighbor) /
                         TrixiParticles.current_density(v, system, neighbor)
                gradient = TrixiParticles.smoothing_kernel_grad(system, SVector(pos_diff),
                                                                distance, particle)
                for j in 1:3, i in 1:3
                    first_moment[i, j] -= volume * gradient[i] * pos_diff[j]
                end
            end
        end
        @test first_moment ≈ Matrix{Float64}(I, 3, 3) atol = 3e-12

        collinear_coordinates = [0.0 0.1 0.2; 0.0 0.0 0.0]
        collinear = InitialCondition(; coordinates=collinear_coordinates,
                                     velocity=zeros(2, 3), density=fill(1000.0, 3),
                                     particle_spacing=0.1)
        collinear_system = WeaklyCompressibleSPHSystem(collinear;
                                                       smoothing_kernel=WendlandC6Kernel{2}(),
                                                       smoothing_length=0.2,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation,
                                                       correction=GradientCorrection())
        collinear_semi = Semidiscretization(collinear_system;
                                            parallelization_backend=SerialBackend())
        collinear_ode = semidiscretize(collinear_semi, (0.0, 1.0);
                                       reset_threads=false)
        collinear_v = Array(collinear_ode.u0.x[1])
        collinear_u = Array(collinear_ode.u0.x[2])
        collinear_system = first(collinear_ode.p.semi.systems)
        TrixiParticles.update_systems_and_nhs(collinear_v, collinear_u,
                                              collinear_ode.p.semi, 0.0)
        for particle in TrixiParticles.eachparticle(collinear_system)
            @test TrixiParticles.correction_matrix(collinear_system, particle) == I
        end

        nearly_collinear_coordinates = [0.0 0.1 0.2; 0.0 1.0e-12 0.0]
        nearly_collinear = InitialCondition(; coordinates=nearly_collinear_coordinates,
                                            velocity=zeros(2, 3),
                                            density=fill(1000.0, 3),
                                            particle_spacing=0.1)
        nearly_collinear_system = WeaklyCompressibleSPHSystem(nearly_collinear;
                                                              smoothing_kernel=WendlandC6Kernel{2}(),
                                                              smoothing_length=0.2,
                                                              density_calculator=ContinuityDensity(),
                                                              state_equation,
                                                              correction=GradientCorrection())
        nearly_collinear_semi = Semidiscretization(nearly_collinear_system;
                                                   parallelization_backend=SerialBackend())
        nearly_collinear_ode = semidiscretize(nearly_collinear_semi, (0.0, 1.0);
                                              reset_threads=false)
        nearly_collinear_v = Array(nearly_collinear_ode.u0.x[1])
        nearly_collinear_u = Array(nearly_collinear_ode.u0.x[2])
        nearly_collinear_system = first(nearly_collinear_ode.p.semi.systems)
        TrixiParticles.update_systems_and_nhs(nearly_collinear_v, nearly_collinear_u,
                                              nearly_collinear_ode.p.semi, 0.0)
        for particle in TrixiParticles.eachparticle(nearly_collinear_system)
            @test TrixiParticles.correction_matrix(nearly_collinear_system, particle) == I
        end
    end

    @testset "Shepard partition of unity" begin
        setup = correction_setup(nothing)
        (; system, semi, v_ode, u_ode) = setup
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        coefficient = zeros(TrixiParticles.nparticles(system))
        numerator = zero(coefficient)

        TrixiParticles.compute_shepard_coeff!(system,
                                              TrixiParticles.current_coordinates(u, system),
                                              v_ode, u_ode, semi, coefficient)
        coordinates = TrixiParticles.current_coordinates(u, system)
        TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                              semi) do particle, neighbor, pos_diff,
                                                       distance
            numerator[particle] += TrixiParticles.hydrodynamic_mass(system, neighbor) *
                                   TrixiParticles.smoothing_kernel(system, distance,
                                                                   particle)
        end

        @test numerator ./ coefficient ≈ fill(1000.0, length(numerator)) atol = 2e-12
        @test TrixiParticles.current_density(v, system) == fill(1000.0, length(numerator))
    end

    @testset "Manufactured continuity equation" begin
        analytic_density_rate = -2000.0
        errors = Dict{Any, Float64}()
        corrections = (nothing, KernelCorrection(), GradientCorrection(),
                       BlendedGradientCorrection(0.4), MixedKernelGradientCorrection())

        for correction in corrections
            setup = correction_setup(correction)
            dv_ode = zero(setup.v_ode)
            TrixiParticles.kick!(dv_ode, setup.v_ode, setup.u_ode,
                                 (; semi=setup.semi, split_integration_data=nothing), 0.0)
            dv = TrixiParticles.wrap_v(dv_ode, setup.system, setup.semi)
            error = dv[end, :] .- analytic_density_rate
            errors[correction] = sqrt(sum(abs2, error) / length(error))
        end

        @test errors[GradientCorrection()] < 2e-10
        @test errors[MixedKernelGradientCorrection()] < 2e-10
        @test errors[BlendedGradientCorrection(0.4)] < errors[nothing]
        @test errors[nothing] > 1.0

        for correction in (GradientCorrection(), MixedKernelGradientCorrection())
            setup = correction_setup(correction; edac=true)
            dv_ode = zero(setup.v_ode)
            TrixiParticles.kick!(dv_ode, setup.v_ode, setup.u_ode,
                                 (; semi=setup.semi, split_integration_data=nothing), 0.0)
            dv = TrixiParticles.wrap_v(dv_ode, setup.system, setup.semi)
            pressure_error = dv[3, :] .+ 200000.0
            density_error = dv[4, :] .+ 2000.0
            @test sqrt(sum(abs2, pressure_error) / length(pressure_error)) < 2e-8
            @test sqrt(sum(abs2, density_error) / length(density_error)) < 2e-10
        end
    end

    @testset "Supported pressure variation matrix" begin
        function set_pressure_field!(setup, edac)
            pressure = range(1.0, 2.0;
                             length=TrixiParticles.nparticles(setup.system))
            if edac
                v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
                v[3, :] .= pressure
            elseif setup.system.density_calculator isa ContinuityDensity
                v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
                v[end, :] .= 1000.0 .+ pressure
            else
                setup.system.pressure .= pressure
            end
            return setup
        end

        summation_corrections = ((; correction=nothing, density_correction=nothing,
                                  gradient_correction=nothing),
                                 (; correction=ShepardKernelCorrection(),
                                  density_correction=nothing,
                                  gradient_correction=nothing),
                                 (; correction=KernelCorrection(),
                                  density_correction=nothing,
                                  gradient_correction=nothing),
                                 (; correction=GradientCorrection(),
                                  density_correction=nothing,
                                  gradient_correction=nothing),
                                 (; correction=BlendedGradientCorrection(0.5),
                                  density_correction=nothing,
                                  gradient_correction=nothing),
                                 (; correction=MixedKernelGradientCorrection(),
                                  density_correction=nothing,
                                  gradient_correction=nothing),
                                 (; correction=nothing,
                                  density_correction=ShepardKernelCorrection(),
                                  gradient_correction=MixedKernelGradientCorrection()))
        continuity_corrections = ((; correction=nothing, density_correction=nothing,
                                   gradient_correction=nothing),
                                  (; correction=KernelCorrection(),
                                   density_correction=nothing,
                                   gradient_correction=nothing),
                                  (; correction=GradientCorrection(),
                                   density_correction=nothing,
                                   gradient_correction=nothing),
                                  (; correction=BlendedGradientCorrection(0.5),
                                   density_correction=nothing,
                                   gradient_correction=nothing),
                                  (; correction=MixedKernelGradientCorrection(),
                                   density_correction=nothing,
                                   gradient_correction=nothing),
                                  (; correction=nothing, density_correction=nothing,
                                   gradient_correction=MixedKernelGradientCorrection()))
        summation_pressure = (nothing,
                              TrixiParticles.pressure_acceleration_summation_density,
                              TrixiParticles.inter_particle_averaged_pressure)
        continuity_pressure = (nothing,
                               TrixiParticles.pressure_acceleration_continuity_density,
                               TrixiParticles.inter_particle_averaged_pressure)

        for edac in (false, true), configuration in summation_corrections,
            pressure_acceleration in summation_pressure
            setup = correction_setup(configuration.correction; n=4, edac,
                                     density_calculator=SummationDensity(),
                                     density_correction=configuration.density_correction,
                                     gradient_correction=configuration.gradient_correction,
                                     pressure_acceleration)
            set_pressure_field!(setup, edac)
            dv_ode = zero(setup.v_ode)
            TrixiParticles.kick!(dv_ode, setup.v_ode, setup.u_ode,
                                 (; semi=setup.semi, split_integration_data=nothing), 0.0)
            @test all(isfinite, dv_ode)
            @test any(!iszero, view(dv_ode, 1:2, :))
        end

        for edac in (false, true), configuration in continuity_corrections,
            pressure_acceleration in continuity_pressure
            setup = correction_setup(configuration.correction; n=4, edac,
                                     density_calculator=ContinuityDensity(),
                                     density_correction=configuration.density_correction,
                                     gradient_correction=configuration.gradient_correction,
                                     pressure_acceleration)
            set_pressure_field!(setup, edac)
            dv_ode = zero(setup.v_ode)
            TrixiParticles.kick!(dv_ode, setup.v_ode, setup.u_ode,
                                 (; semi=setup.semi, split_integration_data=nothing), 0.0)
            @test all(isfinite, dv_ode)
            @test any(!iszero, view(dv_ode, 1:2, :))
        end

        for edac in (false, true)
            setup_tensile = correction_setup(; n=4, edac,
                                             density_calculator=ContinuityDensity(),
                                             pressure_acceleration=tensile_instability_control)
            set_pressure_field!(setup_tensile, edac)
            dv_tensile = zero(setup_tensile.v_ode)
            TrixiParticles.kick!(dv_tensile, setup_tensile.v_ode, setup_tensile.u_ode,
                                 (; semi=setup_tensile.semi,
                                  split_integration_data=nothing), 0.0)
            @test all(isfinite, dv_tensile)
            @test any(!iszero, view(dv_tensile, 1:2, :))

            for configuration in continuity_corrections[2:end]
                @test_throws ArgumentError correction_setup(configuration.correction;
                                                            n=4, edac,
                                                            density_calculator=ContinuityDensity(),
                                                            density_correction=configuration.density_correction,
                                                            gradient_correction=configuration.gradient_correction,
                                                            pressure_acceleration=tensile_instability_control)
            end
        end
    end

    @testset "Continuity density reinitialization" begin
        setup = correction_setup()
        v = TrixiParticles.wrap_v(setup.v_ode, setup.system, setup.semi)
        u = TrixiParticles.wrap_u(setup.u_ode, setup.system, setup.semi)
        TrixiParticles.reinit_density!(setup.system, v, u, setup.v_ode, setup.u_ode,
                                       setup.semi)

        @test TrixiParticles.current_density(v, setup.system) ≈ fill(1000.0, 81) atol = 2e-12
        @test maximum(abs, setup.system.pressure) < 2e-10
    end

    @testset "Analytical operator scaling" begin
        include(joinpath(validation_dir(), "corrections", "convergence.jl"))
        results = CorrectionConvergence.run_convergence(; resolutions=(12, 24, 48))
        @test all(result -> isfinite(result.error), results)

        function finest(method, operator, region)
            return last(filter(result -> result.method == method &&
                                         result.operator == operator &&
                                         result.region == region,
                               results))
        end

        raw_interpolation_boundary = finest(:none, :interpolation, :boundary)
        shepard_interpolation_boundary = finest(:shepard, :interpolation, :boundary)
        raw_difference_boundary = finest(:none, :difference_gradient, :boundary)
        gradient_difference_boundary = finest(:gradient, :difference_gradient, :boundary)
        blended_difference_boundary = finest(:blended, :difference_gradient, :boundary)
        mixed_difference_boundary = finest(:mixed, :difference_gradient, :boundary)
        raw_direct_boundary = finest(:none, :direct_gradient, :boundary)
        kernel_direct_boundary = finest(:kernel, :direct_gradient, :boundary)
        mixed_direct_boundary = finest(:mixed, :direct_gradient, :boundary)

        @test shepard_interpolation_boundary.order >
              raw_interpolation_boundary.order + 0.9
        @test gradient_difference_boundary.order > raw_difference_boundary.order + 0.9
        @test mixed_difference_boundary.order > raw_difference_boundary.order + 0.9
        @test kernel_direct_boundary.order > raw_direct_boundary.order + 0.9
        @test mixed_direct_boundary.order > kernel_direct_boundary.order + 0.9
        @test blended_difference_boundary.error < raw_difference_boundary.error

        shepard_interpolation_interior = finest(:shepard, :interpolation, :interior)
        gradient_difference_interior = finest(:gradient, :difference_gradient, :interior)
        mixed_difference_interior = finest(:mixed, :difference_gradient, :interior)
        mixed_direct_interior = finest(:mixed, :direct_gradient, :interior)
        raw_density_boundary = finest(:none, :summation_density, :boundary)
        shepard_density_boundary = finest(:shepard, :summation_density, :boundary)
        reinitialized_density_boundary = finest(:shepard, :density_reinitialization,
                                                :boundary)
        reinitialized_density_interior = finest(:shepard, :density_reinitialization,
                                                :interior)

        @test shepard_interpolation_interior.order > 1.8
        @test gradient_difference_interior.order > 1.8
        @test mixed_difference_interior.order > 1.8
        @test mixed_direct_interior.order > 1.8
        @test shepard_density_boundary.error < raw_density_boundary.error
        @test abs(shepard_density_boundary.order) < 0.1
        @test reinitialized_density_boundary.order > 0.9
        @test reinitialized_density_interior.order > 1.8

        pressure_operators = (:pressure_summation,
                              :pressure_interparticle_summation,
                              :pressure_continuity,
                              :pressure_interparticle_continuity)
        for operator in pressure_operators
            @test finest(:gradient, operator, :interior).order > 1.8
            @test finest(:mixed, operator, :interior).order > 1.8
        end
        for operator in (:pressure_summation, :pressure_interparticle_summation)
            @test finest(:shepard_mixed, operator, :interior).order > 1.8
        end

        constant_pressure_results = filter(results) do result
            startswith(string(result.operator), "constant_pressure_") &&
                result.region == :interior && result.resolution == 48
        end
        @test !isempty(constant_pressure_results)
        @test maximum(result -> result.error, constant_pressure_results) < 1e-7

        for region in (:boundary, :interior)
            tensile = finest(:none, :pressure_tensile_positive, region)
            continuity = finest(:none, :pressure_continuity, region)
            @test tensile.error ≈ continuity.error rtol = 5e-13
        end
    end
end
