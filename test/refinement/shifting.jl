@testset verbose=true "Shifting" begin
    smoothing_kernel = WendlandC2Kernel{2}()
    sound_speed = 10.0
    reference_density = 1000.0
    state_equation = StateEquationCole(; sound_speed, reference_density, exponent=7)
    particle_spacing = 0.05
    smoothing_length = 1.2 * particle_spacing

    # The default update strategy does not support inactive particles
    neighborhood_search = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())

    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2, min_spacing=0.025,
                                                   max_spacing=0.05, reference_density)

    # A closed tank, so that the lattice continues into the walls
    tank = RectangularTank(particle_spacing, (1.0, 1.0), (1.0, 1.0), reference_density;
                           n_layers=3)
    boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length;
                                                 state_equation)
    nearest_particle(coordinates,
                     x) = argmin(vec(sum(abs2, coordinates .- x, dims=1)))

    # A particle in the middle of the tank, far away from the walls
    center = nearest_particle(tank.fluid.coordinates, [0.475, 0.475])

    function wcsph(fluid; density_calculator=ContinuityDensity(), buffer_size=4,
                   particle_refinement=refinement, kwargs...)
        return WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           state_equation, density_calculator,
                                           particle_refinement, buffer_size, kwargs...)
    end

    function edac(fluid; density_calculator=SummationDensity(), buffer_size=4,
                  particle_refinement=refinement, kwargs...)
        return EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           sound_speed, density_calculator,
                                           particle_refinement, buffer_size, kwargs...)
    end

    function semidiscretize_systems(fluid_system; compute_rhs=true)
        semi = Semidiscretization(fluid_system,
                                  WallBoundarySystem(tank.boundary, boundary_model);
                                  neighborhood_search,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x

        # Compute the summation density and the boundary pressure
        compute_rhs && TrixiParticles.kick!(zero(v_ode), v_ode, u_ode, ode.p, 0.0)

        return v_ode, u_ode, ode.p.semi
    end

    # The fluid of the tank with the particle `center` moved by `offset`
    function perturbed_fluid(offset)
        fluid = deepcopy(tank.fluid)
        fluid.coordinates[:, center] .+= offset

        return fluid
    end

    @testset verbose=true "Kernel Inflection Point" begin
        kernels = (SchoenbergCubicSplineKernel{2}(), SchoenbergQuinticSplineKernel{2}(),
                   WendlandC2Kernel{2}())

        @testset "$(nameof(typeof(kernel)))" for kernel in kernels
            h = 0.3
            xi = TrixiParticles.kernel_inflection_point(kernel)
            dW(r) = TrixiParticles.kernel_deriv(kernel, r, h)

            # The second derivative of the kernel changes its sign at `xi * h`
            eps = 1e-6
            d2W(r) = (dW(r + eps) - dW(r - eps)) / (2 * eps)

            @test abs(d2W(xi * h)) < 1e-5 * abs(d2W(0.1 * h))
            @test sign(d2W((xi - 0.05) * h)) != sign(d2W((xi + 0.05) * h))
        end

        @test_throws "shifting with particle refinement is not supported" TrixiParticles.kernel_inflection_point(GaussianKernel{2}())
    end

    setups = [
        ("WCSPH with `ContinuityDensity`", wcsph, ContinuityDensity()),
        ("WCSPH with `SummationDensity`", wcsph, SummationDensity()),
        ("EDAC with `ContinuityDensity`", edac, ContinuityDensity()),
        ("EDAC with `SummationDensity`", edac, SummationDensity())
    ]

    @testset verbose=true "$name" for (name, create_fluid, density_calculator) in setups
        @testset "Uniform Lattice is Not Shifted" begin
            v_ode, u_ode,
            semi = semidiscretize_systems(create_fluid(tank.fluid; density_calculator))
            system = semi.systems[1]
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u_old = copy(u)
            v_old = copy(v)

            # The smoothing lengths are reset to the spacing of the masses (Eq. 35)
            TrixiParticles.smoothing_length(system) .= particle_spacing

            TrixiParticles.shift_particles!(semi, v_ode, u_ode)

            @test isapprox(u, u_old, atol=1e-14)
            @test isapprox(v, v_old, atol=1e-10)

            active = TrixiParticles.each_active_particle(system)
            @test all(≈(smoothing_length),
                      TrixiParticles.smoothing_length(system)[active])
        end

        @testset "Perturbed Particle is Shifted Back" begin
            offset = [0.2, -0.1] * particle_spacing
            v_ode, u_ode,
            semi = semidiscretize_systems(create_fluid(perturbed_fluid(offset);
                                                       density_calculator))
            system = semi.systems[1]
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            u_inactive = copy(u[:, (end - 3):end])

            TrixiParticles.shift_particles!(semi, v_ode, u_ode)

            # The particle moves back towards its lattice position
            displacement = u[:, center] - tank.fluid.coordinates[:, center]
            @test norm(displacement) < 0.5 * norm(offset)

            # The total shift of all iterations is stored for the property corrections
            total_shift = reshape(system.cache.total_shift, 2, :)
            @test u[:, center] ≈
                  perturbed_fluid(offset).coordinates[:, center] +
                  total_shift[:, center]

            # Inactive particles are not shifted
            @test u[:, (end - 3):end] == u_inactive
        end

        @testset "Properties of Linear Fields are Corrected" begin
            offset = [0.2, -0.1] * particle_spacing
            v_ode, u_ode,
            semi = semidiscretize_systems(create_fluid(perturbed_fluid(offset);
                                                       density_calculator))
            system = semi.systems[1]
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            v = TrixiParticles.wrap_v(v_ode, system, semi)

            grad_p = SVector(100.0, -50.0)
            grad_v = TrixiParticles.SMatrix{2, 2}(1.0, 0.5, -2.0, 3.0)
            grad_rho = SVector(2.0, 1.0)

            for particle in TrixiParticles.each_active_particle(system)
                x = TrixiParticles.current_coords(u, system, particle)
                TrixiParticles.set_particle_pressure!(v, system, particle,
                                                      1000.0 + dot(grad_p, x))
                TrixiParticles.set_particle_velocity!(v, system, particle, grad_v * x)
                TrixiParticles.set_particle_density!(v, system, particle,
                                                     reference_density + dot(grad_rho, x))
            end

            p_old = TrixiParticles.current_pressure(v, system, center)
            v_old = TrixiParticles.current_velocity(v, system, center)
            rho_old = TrixiParticles.current_density(v, system, center)

            TrixiParticles.shift_particles!(semi, v_ode, u_ode)

            shift = reshape(system.cache.total_shift, 2, :)[:, center]
            @test norm(shift) > 0.01 * particle_spacing

            # The gradients of linear fields are only approximated by SPH
            p_new = TrixiParticles.current_pressure(v, system, center)
            @test isapprox(p_new - p_old, dot(grad_p, shift), rtol=0.05)

            v_new = TrixiParticles.current_velocity(v, system, center)
            @test isapprox(v_new - v_old, grad_v * shift, rtol=0.05)

            rho_new = TrixiParticles.current_density(v, system, center)
            if density_calculator isa ContinuityDensity
                @test isapprox(rho_new - rho_old, dot(grad_rho, shift), rtol=0.05)
            else
                # The summation density is computed from the shifted positions
                # in the next evaluation of the right-hand side
                @test rho_new == rho_old
            end
        end
    end

    @testset verbose=true "Smoothing Length (Eq. 35)" begin
        v_ode, u_ode, semi = semidiscretize_systems(edac(tank.fluid))
        system = semi.systems[1]
        u = TrixiParticles.wrap_u(u_ode, system, semi)

        # Change the mass of one particle and use a different smoothing length for all.
        # The compact support must be inside the search radius of the neighborhood search.
        TrixiParticles.set_particle_mass!(system, center, 2 * system.mass[center])
        TrixiParticles.smoothing_length(system) .= particle_spacing

        TrixiParticles.update_smoothing_lengths!(system, u, semi)

        # Brute force: the neighbors within the compact support of the old smoothing length
        active = collect(TrixiParticles.each_active_particle(system))
        radius = TrixiParticles.compact_support(smoothing_kernel, particle_spacing)
        function expected_smoothing_length(particle)
            neighbors = filter(neighbor -> norm(u[:, neighbor] - u[:, particle]) <= radius,
                               active)
            mass_avg = sum(system.mass[neighbors]) / length(neighbors)

            return 1.2 * sqrt(mass_avg / reference_density)
        end

        h = TrixiParticles.smoothing_length(system)
        neighbor_of_center = nearest_particle(tank.fluid.coordinates, [0.525, 0.475])
        far_away = nearest_particle(tank.fluid.coordinates, [0.125, 0.875])

        @test h[center] > smoothing_length
        @test isapprox(h[center], expected_smoothing_length(center), rtol=1e-12)
        @test isapprox(h[neighbor_of_center], expected_smoothing_length(neighbor_of_center),
                       rtol=1e-12)
        @test isapprox(h[far_away], smoothing_length, rtol=1e-12)
    end

    @testset verbose=true "Shifting Technique of the Refinement" begin
        offset = [0.2, -0.1] * particle_spacing

        # Shift the perturbed fluid and return the shifted position of `center`
        function shifted_center(shifting_technique)
            particle_refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2,
                                                                    min_spacing=0.025,
                                                                    max_spacing=0.05,
                                                                    reference_density,
                                                                    shifting_technique)
            fluid = perturbed_fluid(offset)
            v_ode, u_ode,
            semi = semidiscretize_systems(edac(fluid; particle_refinement))
            system = semi.systems[1]
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            TrixiParticles.smoothing_length(system) .= particle_spacing

            TrixiParticles.shift_particles!(semi, v_ode, u_ode)

            # The smoothing lengths are updated with and without shifting
            active = TrixiParticles.each_active_particle(system)
            @test all(≈(smoothing_length),
                      TrixiParticles.smoothing_length(system)[active])

            return u[:, center] - fluid.coordinates[:, center]
        end

        @testset "Default" begin
            (; shifting_technique) = refinement
            @test shifting_technique isa TrixiParticles.ShiftingHaftu
            @test shifting_technique.max_iterations == 100
            @test shifting_technique.tolerance == 2e-3
            @test shifting_technique.theta == 0.5
            @test shifting_technique.max_shift_factor == 0.05
        end

        @testset "Without Shifting" begin
            @test iszero(shifted_center(nothing))
        end

        @testset "One Iteration" begin
            shift_default = shifted_center(TrixiParticles.ShiftingHaftu())
            shift_single = shifted_center(TrixiParticles.ShiftingHaftu(max_iterations=1))

            # One iteration shifts by `theta` times Eq. 15, but at most `0.05 h`
            @test 0 < norm(shift_single) <= (1 + 1e-12) * 0.05 * smoothing_length
            @test norm(shift_single) < norm(shift_default)
        end
    end

    @testset verbose=true "Particle Shifting Technique" begin
        # A perturbed lattice, so that the shifting velocity is not zero
        fluid = deepcopy(tank.fluid)
        fluid.coordinates .+= 0.1 * particle_spacing *
                              [sin(i + 3 * j) for i in 1:2, j in 1:nparticles(fluid)]

        function shifting_velocity(particle_refinement)
            shifting = ParticleShiftingTechnique()
            fluid_system = edac(fluid; density_calculator=ContinuityDensity(),
                                buffer_size=nothing, particle_refinement,
                                shifting_technique=shifting)
            # The right-hand side requires the `UpdateCallback` with this shifting
            v_ode, u_ode, semi = semidiscretize_systems(fluid_system, compute_rhs=false)
            system = semi.systems[1]
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            TrixiParticles.set_particle_velocity!(v, system, 1, [1.0, 0.0])

            TrixiParticles.update_shifting_inner!(system, shifting, v, u, v_ode, u_ode,
                                                  semi)

            return copy(system.cache.delta_v)
        end

        @testset "Uniform Smoothing Length Matches Constant Smoothing Length" begin
            delta_v_constant = shifting_velocity(nothing)
            delta_v_uniform = shifting_velocity(refinement)

            @test any(!iszero, delta_v_constant)
            @test isapprox(delta_v_uniform, delta_v_constant, rtol=1e-12)
        end
    end

    @testset verbose=true "Momentum Equation Term (Eq. 12 and 13)" begin
        coordinates = [0.0 0.07
                       0.0 0.02]
        mass = [2.5, 1.5]
        rho_a, rho_b = 1010.0, 990.0
        pair = InitialCondition(; coordinates, mass, density=[rho_a, rho_b],
                                particle_spacing)
        pos_diff = SVector(-0.07, -0.02)
        distance = norm(pos_diff)
        v_a, v_b = SVector(1.0, -0.5), SVector(0.3, 0.8)
        delta_v_a, delta_v_b = SVector(0.02, 0.01), SVector(-0.03, 0.04)
        m_a, m_b = mass
        term = TrixiParticles.MomentumEquationTermSun2019()

        function shifting_dv(system, h_a, h_b)
            TrixiParticles.set_particle_smoothing_length!(system, 1, h_a)
            TrixiParticles.set_particle_smoothing_length!(system, 2, h_b)
            system.cache.delta_v .= [delta_v_a; delta_v_b]

            grad_kernel = TrixiParticles.symmetric_kernel_grad(system, system, pos_diff,
                                                               distance, 1, 2)

            return TrixiParticles.add_dv_shifting(zero(v_a), term, system, system,
                                                  nothing, nothing, 1, 2, m_a, m_b,
                                                  rho_a, rho_b, v_a, v_b, pos_diff,
                                                  distance, grad_kernel, nothing)
        end

        function expected(h_a, h_b, beta_a, beta_b)
            grad_a = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, distance, h_a)
            grad_b = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, distance, h_b)
            A_a = v_a * delta_v_a' / (rho_a * beta_a)
            A_b = v_b * delta_v_b' / (rho_b * beta_b)

            return m_b * (A_a * grad_a + A_b * grad_b) +
                   m_b / (rho_b * beta_a) * v_a * dot(delta_v_a - delta_v_b, grad_a)
        end

        shifting_technique = ParticleShiftingTechnique()

        @testset "WCSPH" begin
            system = WeaklyCompressibleSPHSystem(pair; smoothing_kernel, smoothing_length,
                                                 state_equation,
                                                 density_calculator=ContinuityDensity(),
                                                 shifting_technique,
                                                 particle_refinement=refinement)
            h_a, h_b = 0.05, 0.065
            dv = shifting_dv(system, h_a, h_b)

            @test !iszero(dv)
            @test isapprox(dv, expected(h_a, h_b, 1, 1), rtol=1e-12)
        end

        @testset "EDAC" begin
            system = EntropicallyDampedSPHSystem(pair; smoothing_kernel, smoothing_length,
                                                 sound_speed,
                                                 density_calculator=ContinuityDensity(),
                                                 shifting_technique,
                                                 particle_refinement=refinement)
            beta_a, beta_b = 1.1, 0.9
            system.cache.beta .= [beta_a, beta_b]
            h_a, h_b = 0.05, 0.065
            dv = shifting_dv(system, h_a, h_b)

            @test isapprox(dv, expected(h_a, h_b, beta_a, beta_b), rtol=1e-12)
        end

        @testset "Uniform Smoothing Length Matches Constant Smoothing Length" begin
            # Equal densities, so that both formulations agree
            pair_ = InitialCondition(; coordinates, mass, density=1000.0, particle_spacing)
            rho = 1000.0

            function dv(particle_refinement)
                system = WeaklyCompressibleSPHSystem(pair_; smoothing_kernel,
                                                     smoothing_length, state_equation,
                                                     density_calculator=ContinuityDensity(),
                                                     shifting_technique,
                                                     particle_refinement)
                system.cache.delta_v .= [delta_v_a; delta_v_b]
                grad_kernel = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff,
                                                         distance, smoothing_length)

                return TrixiParticles.add_dv_shifting(zero(v_a), term, system, system,
                                                      nothing, nothing, 1, 2, m_a, m_b,
                                                      rho, rho, v_a, v_b, pos_diff,
                                                      distance, grad_kernel, nothing)
            end

            dv_constant = dv(nothing)

            @test !iszero(dv_constant)
            @test isapprox(dv(refinement), dv_constant, rtol=1e-12)
        end
    end
end
