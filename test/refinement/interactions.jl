@testset verbose=true "Variable Smoothing Length Interactions" begin
    smoothing_kernel = WendlandC2Kernel{2}()
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=7)
    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=0.5, min_spacing=0.05,
                                                   max_spacing=0.1,
                                                   reference_density=1000.0)

    # Evaluate the right-hand side for the given systems
    function rhs(systems...)
        semi = Semidiscretization(systems...; parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

        return parent(dv_ode), ode.p.semi
    end

    @testset verbose=true "Uniform Smoothing Length Matches Constant Smoothing Length" begin
        particle_spacing = 0.1
        smoothing_length = 1.2 * particle_spacing
        tank = RectangularTank(particle_spacing, (0.5, 0.5), (1.0, 1.0), 1000.0;
                               n_layers=3)

        # Use a non-trivial state, so that all terms contribute
        n_fluid = nparticles(tank.fluid)
        fluid = deepcopy(tank.fluid)
        fluid.velocity .= [sin(i + j) for i in 1:2, j in 1:n_fluid]
        fluid.density .+= [10 * cos(i) for i in 1:n_fluid]

        boundary_density_calculators = (ContinuityDensity(), SummationDensity(),
                                        AdamiPressureExtrapolation(),
                                        AdamiPressureExtrapolation(allow_loop_flipping=false))

        @testset "$(nameof(typeof(boundary_density_calculator)))" for boundary_density_calculator in
                                                                      boundary_density_calculators

            function dv(particle_refinement)
                viscosity = ViscosityAdami(nu=1e-3)
                fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                           smoothing_length, viscosity,
                                                           density_calculator=ContinuityDensity(),
                                                           state_equation,
                                                           density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
                                                           particle_refinement)

                boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                             tank.boundary.mass,
                                                             boundary_density_calculator,
                                                             smoothing_kernel,
                                                             smoothing_length;
                                                             state_equation,
                                                             viscosity)
                boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

                return first(rhs(fluid_system, boundary_system))
            end

            dv_constant = dv(nothing)
            dv_uniform = dv(refinement)

            @test any(!iszero, dv_constant)
            @test isapprox(dv_uniform, dv_constant, rtol=1e-12)
        end
    end

    @testset verbose=true "Fluid-Fluid Pair" begin
        # The neighborhood search radius is 2 * 1.2 * 0.1 = 0.24
        coordinates = [0.0 0.15
                       0.0 0.0]
        mass = [1.0, 0.5]
        density = [1100.0, 1100.0]
        initial_condition = InitialCondition(; coordinates, mass, density,
                                             particle_spacing=0.1)

        function pair_dv(h_a, h_b)
            fluid_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                       smoothing_length=0.12,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation,
                                                       particle_refinement=refinement)
            semi = Semidiscretization(fluid_system;
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.1))
            system = ode.p.semi.systems[1]
            TrixiParticles.set_particle_smoothing_length!(system, 1, h_a)
            TrixiParticles.set_particle_smoothing_length!(system, 2, h_b)

            v_ode, u_ode = ode.u0.x
            dv_ode = zero(v_ode)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

            return TrixiParticles.wrap_v(dv_ode, system, ode.p.semi)
        end

        @testset "Different Smoothing Lengths are Antisymmetric" begin
            dv = pair_dv(0.1, 0.12)

            @test !iszero(dv[1, 1])
            @test isapprox(mass[1] * dv[1, 1], -mass[2] * dv[1, 2], rtol=1e-12)
        end

        @testset "Inside the Larger Compact Support Only" begin
            # The distance of 0.15 is outside of the compact support of the first particle
            dv = pair_dv(0.06, 0.12)

            @test !iszero(dv[1, 1])
            @test isapprox(mass[1] * dv[1, 1], -mass[2] * dv[1, 2], rtol=1e-12)
        end

        @testset "Outside of Both Compact Supports" begin
            # The neighborhood search returns this pair, but it must not interact
            @test iszero(pair_dv(0.06, 0.06))
        end

        @testset "Each Pressure Uses the Kernel Gradient of Its Own Particle" begin
            # Different densities give different pressures
            rho_a, rho_b = 1100.0, 1050.0
            pair = InitialCondition(; coordinates, mass, density=[rho_a, rho_b],
                                    particle_spacing=0.1)
            fluid_system = WeaklyCompressibleSPHSystem(pair; smoothing_kernel,
                                                       smoothing_length=0.12,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation,
                                                       particle_refinement=refinement)
            semi = Semidiscretization(fluid_system;
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.1))
            system = ode.p.semi.systems[1]
            h_a, h_b = 0.1, 0.12
            TrixiParticles.set_particle_smoothing_length!(system, 1, h_a)
            TrixiParticles.set_particle_smoothing_length!(system, 2, h_b)

            v_ode, u_ode = ode.u0.x
            dv_ode = zero(v_ode)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
            dv = TrixiParticles.wrap_v(dv_ode, system, ode.p.semi)

            pos_diff = SVector(-0.15, 0.0)
            grad_a = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, h_a)
            grad_b = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, h_b)
            p_a, p_b = state_equation(rho_a), state_equation(rho_b)

            # `pressure_acceleration_continuity_density` with `p_a ∇W(h_a) + p_b ∇W(h_b)`
            expected = -mass[2] / (rho_a * rho_b) * (p_a * grad_a + p_b * grad_b)
            averaged = -mass[2] * (p_a + p_b) / (rho_a * rho_b) * (grad_a + grad_b) / 2

            @test !isapprox(expected[1], averaged[1])
            @test isapprox(dv[1, 1], expected[1], rtol=1e-12)
            @test isapprox(mass[1] * dv[1, 1], -mass[2] * dv[1, 2], rtol=1e-12)
        end
    end

    @testset "Formulation Without Two Gradients is Rejected" begin
        tank = RectangularTank(0.1, (0.5, 0.5), (1.0, 1.0), 1000.0; n_layers=3)
        error_str = "when using particle refinement, the passed pressure " *
                    "acceleration formulation must provide a version with the " *
                    "arguments `m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, W_b`"

        @test_throws ArgumentError(error_str) WeaklyCompressibleSPHSystem(tank.fluid;
                                                                          smoothing_kernel,
                                                                          smoothing_length=0.12,
                                                                          density_calculator=ContinuityDensity(),
                                                                          state_equation,
                                                                          pressure_acceleration=tensile_instability_control,
                                                                          particle_refinement=refinement)
    end

    @testset verbose=true "Wall Next to Fluid With Larger Smoothing Length" begin
        # The distance is outside of the compact support of the wall (2 * 0.06 = 0.12),
        # but inside of the shared compact support (2 * (0.06 + 0.12) / 2 = 0.18).
        fluid = InitialCondition(; coordinates=[0.15; 0.0;;], mass=[10.0], density=[1100.0],
                                 velocity=[1.0; 0.0;;], particle_spacing=0.1)
        wall = InitialCondition(; coordinates=[0.0; 0.0;;], mass=[10.0], density=[1000.0],
                                particle_spacing=0.05)

        function create_systems(boundary_density_calculator)
            fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                       smoothing_length=0.12,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation,
                                                       particle_refinement=refinement)
            boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                         boundary_density_calculator,
                                                         smoothing_kernel, 0.06;
                                                         state_equation)

            return fluid_system, WallBoundarySystem(wall, boundary_model)
        end

        @testset "Search Radius" begin
            fluid_system, boundary_system = create_systems(ContinuityDensity())

            # The compact support of the largest fluid smoothing length, which is
            # 1.2 times the spacing of `max_mass_factor` times the mass at `max_spacing`
            @test TrixiParticles.compact_support(boundary_system, fluid_system) ≈
                  2 * 1.2 * sqrt(1.05) * 0.1

            # Without variable smoothing lengths, the search radius is unchanged
            constant_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                          smoothing_length=0.12,
                                                          density_calculator=ContinuityDensity(),
                                                          state_equation)
            @test TrixiParticles.compact_support(boundary_system, constant_system) ≈ 0.12
        end

        @testset "Continuity Equation" begin
            dv_ode, semi = rhs(create_systems(ContinuityDensity())...)
            dv_boundary = TrixiParticles.wrap_v(dv_ode, semi.systems[2], semi)

            # With an inactive wall gradient, only half of the fluid gradient remains
            grad_kernel = TrixiParticles.kernel_grad(smoothing_kernel, SVector(-0.15, 0.0),
                                                     0.15, 0.12) / 2
            expected = 10.0 * (1000.0 / 1100.0) * dot(SVector(-1.0, 0.0), grad_kernel)

            @test !iszero(dv_boundary[end, 1])
            @test isapprox(dv_boundary[end, 1], expected, rtol=1e-12)
        end

        @testset "Pressure Extrapolation Does Not Depend on the Loop Order" begin
            # With a single fluid and a single boundary particle and one thread,
            # `allow_loop_flipping=true` selects the loop over the fluid particles.
            pressure(allow_loop_flipping) = begin
                calculator = AdamiPressureExtrapolation(; allow_loop_flipping)
                _, semi = rhs(create_systems(calculator)...)
                only(semi.systems[2].boundary_model.pressure)
            end

            @test pressure(false) > 0
            @test isapprox(pressure(false), pressure(true), rtol=1e-12)
        end
    end

    @testset verbose=true "Corrections Are Rejected" begin
        tank = RectangularTank(0.1, (0.5, 0.5), (1.0, 1.0), 1000.0; n_layers=3)
        error_str = "corrections are not supported with particle refinement"

        @testset "Fluid" begin
            fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                       smoothing_length=0.12,
                                                       density_calculator=SummationDensity(),
                                                       state_equation,
                                                       correction=KernelCorrection(),
                                                       particle_refinement=refinement)
            @test_throws "$error_str" Semidiscretization(fluid_system)
        end

        @testset "Boundary Model" begin
            fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                       smoothing_length=0.12,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation,
                                                       particle_refinement=refinement)
            boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                         tank.boundary.mass,
                                                         AdamiPressureExtrapolation(),
                                                         smoothing_kernel, 0.12;
                                                         state_equation,
                                                         correction=KernelCorrection())
            boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

            @test_throws "$error_str" Semidiscretization(fluid_system, boundary_system)
        end
    end

    @testset verbose=true "Boundary Models and Structures" begin
        fluid = InitialCondition(; coordinates=[0.15; 0.0;;], mass=[10.0], density=[1100.0],
                                 velocity=[1.0; 0.0;;], particle_spacing=0.1)
        boundary = InitialCondition(; coordinates=[0.0; 0.0;;], mass=[10.0],
                                    density=[1000.0], particle_spacing=0.1)

        function create_fluid_system(particle_refinement)
            return WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                               smoothing_length=0.12,
                                               density_calculator=ContinuityDensity(),
                                               state_equation, particle_refinement)
        end

        function fluid_dv(systems...)
            dv_ode, semi = rhs(systems...)
            return collect(TrixiParticles.wrap_v(dv_ode, semi.systems[1], semi))
        end

        @testset "Monaghan-Kajtar Wall" begin
            function dv(particle_refinement)
                boundary_model = BoundaryModelMonaghanKajtar(10.0, 1.0, 0.1, [10.0])
                boundary_system = WallBoundarySystem(boundary, boundary_model)

                return fluid_dv(create_fluid_system(particle_refinement), boundary_system)
            end

            @test any(!iszero, dv(nothing))
            @test isapprox(dv(refinement), dv(nothing), rtol=1e-12)
        end

        @testset "TLSPH With Different Smoothing Length" begin
            function dv(particle_refinement)
                structure = RectangularShape(0.1, (3, 3), (-0.2, -0.1); density=1000.0)
                boundary_model = BoundaryModelDummyParticles(structure.density,
                                                             structure.mass,
                                                             AdamiPressureExtrapolation(),
                                                             smoothing_kernel, 0.12;
                                                             state_equation)
                structure_system = TotalLagrangianSPHSystem(structure; smoothing_kernel,
                                                            smoothing_length=0.2,
                                                            young_modulus=1e4,
                                                            poisson_ratio=0.3,
                                                            boundary_model)

                return fluid_dv(create_fluid_system(particle_refinement), structure_system)
            end

            @test any(!iszero, dv(nothing))
            @test isapprox(dv(refinement), dv(nothing), rtol=1e-12)
        end

        @testset "Rigid Body Forces Are Opposite" begin
            function forces(h_fluid)
                # A density above the reference density gives a non-zero rigid body pressure,
                # which is what remains outside of the compact support of the fluid particle.
                boundary_model = BoundaryModelDummyParticles([1100.0], [10.0],
                                                             ContinuityDensity(),
                                                             smoothing_kernel, 0.12;
                                                             state_equation)
                rigid_system = RigidBodySystem(boundary; boundary_model)

                semi = Semidiscretization(create_fluid_system(refinement), rigid_system;
                                          parallelization_backend=SerialBackend())
                ode = semidiscretize(semi, (0.0, 0.1))
                fluid_system, rigid_system = ode.p.semi.systems
                TrixiParticles.set_particle_smoothing_length!(fluid_system, 1, h_fluid)

                v_ode, u_ode = ode.u0.x
                dv_ode = zero(v_ode)
                TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
                dv = TrixiParticles.wrap_v(dv_ode, fluid_system, ode.p.semi)

                return 10.0 * dv[1, 1], rigid_system.force_per_particle[1, 1]
            end

            # The distance of 0.15 is inside and outside of the fluid's compact support
            for h_fluid in (0.1, 0.06)
                force_fluid, force_rigid = forces(h_fluid)

                @test !iszero(force_fluid)
                @test isapprox(force_fluid, -force_rigid, rtol=1e-12)
            end
        end
    end
end
