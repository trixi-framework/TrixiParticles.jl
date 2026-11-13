@testset verbose=true "EDAC With Variable Smoothing Lengths" begin
    smoothing_kernel = WendlandC2Kernel{2}()
    sound_speed = 10.0
    reference_density = 1000.0
    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2, min_spacing=0.05,
                                                   max_spacing=0.1, reference_density)

    function edac_system(initial_condition; smoothing_length=0.12, kwargs...)
        return EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                           smoothing_length, sound_speed,
                                           density_calculator=ContinuityDensity(),
                                           particle_refinement=refinement, kwargs...)
    end

    # Evaluate the right-hand side after setting the smoothing lengths of the first system
    function rhs(systems...; smoothing_lengths=nothing)
        semi = Semidiscretization(systems...; parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        semi = ode.p.semi

        if !isnothing(smoothing_lengths)
            for (particle, h) in enumerate(smoothing_lengths)
                TrixiParticles.set_particle_smoothing_length!(semi.systems[1], particle, h)
            end
        end

        v_ode, u_ode = ode.u0.x
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

        return dv_ode, semi
    end

    # Two particles. The neighborhood search radius is 2 * 1.2 * 0.1 = 0.24.
    coordinates = [0.0 0.15
                   0.0 0.0]
    mass = [1.0, 0.5]
    density = [1100.0, 1200.0]
    pressure = [50.0, 20.0]
    velocity = [1.0 -1.0
                0.0 0.0]
    pair = InitialCondition(; coordinates, mass, density, pressure, velocity,
                            particle_spacing=0.1)

    # The beta of a particle with a neighbor at a distance of `r = 0.15`
    function pair_beta(h, m_b, rho_a)
        dw_dr = TrixiParticles.kernel_deriv(smoothing_kernel, 0.15, h)

        return -m_b * 0.15 * dw_dr / (rho_a * 2)
    end

    @testset verbose=true "Beta" begin
        @testset "Regular Grid" begin
            particle_spacing = 0.1
            n = 11
            grid = RectangularShape(particle_spacing, (n, n), (0.0, 0.0);
                                    density=reference_density)

            dv_ode, semi = rhs(edac_system(grid; smoothing_length=0.12))

            # Direct sum over the neighbors of the center particle with a full kernel support
            center = (n^2 + 1) ÷ 2
            grid_coordinates = semi.systems[1].initial_condition.coordinates
            sum_ = sum(1:(n ^ 2)) do particle
                r = norm(grid_coordinates[:, particle] - grid_coordinates[:, center])
                return grid.mass[particle] * r *
                       TrixiParticles.kernel_deriv(smoothing_kernel, r, 0.12)
            end
            expected = -sum_ / (reference_density * 2)

            @test semi.systems[1].cache.beta[center] ≈ expected

            # `beta` is close to one on a regular grid
            @test isapprox(expected, 1.0, atol=0.05)
        end

        @testset "Pair" begin
            _, semi = rhs(edac_system(pair))

            beta = semi.systems[1].cache.beta
            @test beta[1] ≈ pair_beta(0.12, 0.5, 1100.0)
            @test beta[2] ≈ pair_beta(0.12, 1.0, 1200.0)
        end

        @testset "No Neighbors" begin
            single = InitialCondition(; coordinates=[0.0; 0.0;;], mass=[1.0],
                                      density=[1000.0], pressure=[1.0],
                                      particle_spacing=0.1)
            _, semi = rhs(edac_system(single))

            @test only(semi.systems[1].cache.beta) == 1
        end

        @testset "Includes Boundary Particles" begin
            fluid = InitialCondition(; coordinates=[0.15; 0.0;;], mass=[1.0],
                                     density=[1100.0], particle_spacing=0.1)
            wall = InitialCondition(; coordinates=[0.0; 0.0;;], mass=[0.5],
                                    density=[1000.0], particle_spacing=0.1)
            boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                         AdamiPressureExtrapolation(),
                                                         smoothing_kernel, 0.12;
                                                         state_equation=StateEquationCole(;
                                                                                          sound_speed,
                                                                                          reference_density,
                                                                                          exponent=7))
            _, semi = rhs(edac_system(fluid), WallBoundarySystem(wall, boundary_model))

            @test only(semi.systems[1].cache.beta) ≈ pair_beta(0.12, 0.5, 1100.0)
        end
    end

    @testset verbose=true "Fluid-Fluid Pair" begin
        h_a, h_b = 0.1, 0.12
        beta_a = pair_beta(h_a, 0.5, 1100.0)
        beta_b = pair_beta(h_b, 1.0, 1200.0)
        pos_diff = SVector(-0.15, 0.0)
        grad_a = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, h_a)
        grad_b = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, h_b)

        function pair_dv(average_pressure_reduction)
            system = edac_system(pair; average_pressure_reduction)
            dv_ode, semi = rhs(system; smoothing_lengths=(h_a, h_b))

            return TrixiParticles.wrap_v(dv_ode, semi.systems[1], semi), semi
        end

        dv, semi = pair_dv(false)

        @testset "Pressure Acceleration" begin
            expected = -0.5 * (50.0 / (1100.0^2 * beta_a) * grad_a +
                        20.0 / (1200.0^2 * beta_b) * grad_b)

            @test !iszero(expected[1])
            @test isapprox(dv[1, 1], expected[1], rtol=1e-12)
        end

        @testset "Pressure Acceleration is Antisymmetric" begin
            @test isapprox(mass[1] * dv[1, 1], -mass[2] * dv[1, 2], rtol=1e-12)
        end

        @testset "Pressure Evolution" begin
            nu_edac = semi.systems[1].nu_edac
            h_ref = TrixiParticles.initial_smoothing_length(semi.systems[1])
            nu_a, nu_b = nu_edac * h_a / h_ref, nu_edac * h_b / h_ref
            nu_ab = 4 * nu_a * nu_b / (nu_a + nu_b)
            volume_b = 0.5 / 1200.0
            grad_avg = (grad_a + grad_b) / 2

            eos = reference_density * sound_speed^2 / beta_a * volume_b *
                  dot(SVector(2.0, 0.0), grad_a)
            damping = volume_b * nu_ab * (50.0 - 20.0) / beta_a *
                      dot(grad_avg, pos_diff) / (0.15^2 + ((h_a + h_b) / 2)^2 / 100)

            @test !iszero(damping)
            @test isapprox(dv[3, 1], eos + damping, rtol=1e-12)
        end

        @testset "Inside the Larger Compact Support Only" begin
            # The distance of 0.15 is outside of the compact support of the first particle.
            # Without a neighbor in its own compact support, its `beta` falls back to one.
            system = edac_system(pair; average_pressure_reduction=false)
            dv_ode, semi = rhs(system; smoothing_lengths=(0.06, 0.12))
            dv = TrixiParticles.wrap_v(dv_ode, semi.systems[1], semi)

            beta_b = pair_beta(0.12, 1.0, 1200.0)
            grad = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, 0.12)
            expected = -0.5 * 20.0 / (1200.0^2 * beta_b) * grad

            @test semi.systems[1].cache.beta[1] == 1
            @test isapprox(dv[1, 1], expected[1], rtol=1e-12)
        end

        @testset "Outside of Both Compact Supports" begin
            system = edac_system(pair; average_pressure_reduction=false)
            dv_ode, _ = rhs(system; smoothing_lengths=(0.06, 0.06))

            @test iszero(dv_ode)
        end

        @testset "Average Pressure Reduction" begin
            dv_reduced, semi = pair_dv(true)

            @test !isapprox(dv_reduced[1, 1], dv[1, 1])
        end
    end

    @testset verbose=true "Continuity Equation" begin
        v_diff = SVector(2.0, 0.0)
        pos_diff = SVector(-0.15, 0.0)
        system = edac_system(pair; average_pressure_reduction=false)

        @testset "Uses the Kernel Gradient of the Particle" begin
            h_a, h_b = 0.1, 0.12
            dv_ode, semi = rhs(system; smoothing_lengths=(h_a, h_b))
            dv = TrixiParticles.wrap_v(dv_ode, semi.systems[1], semi)

            grad_a = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, h_a)
            expected = 0.5 * dot(v_diff, grad_a)

            @test !iszero(expected)
            @test isapprox(dv[4, 1], expected, rtol=1e-12)
        end

        @testset "Inside the Compact Support of the Neighbor Only" begin
            dv_ode, semi = rhs(system; smoothing_lengths=(0.06, 0.12))
            dv = TrixiParticles.wrap_v(dv_ode, semi.systems[1], semi)

            @test iszero(dv[4, 1])
            @test !iszero(dv[4, 2])
        end
    end

    @testset verbose=true "Pressure Acceleration Formulations" begin
        summation_density = TrixiParticles.pressure_acceleration_summation_density
        continuity_density = TrixiParticles.pressure_acceleration_continuity_density
        inter_particle = TrixiParticles.inter_particle_averaged_pressure

        h_a, h_b = 0.1, 0.12
        m_a, m_b = 1.0, 0.5
        rho_a, rho_b = 1100.0, 1200.0
        p_a, p_b = 50.0, 20.0
        beta_a = pair_beta(h_a, m_b, rho_a)
        beta_b = pair_beta(h_b, m_a, rho_b)
        pos_diff = SVector(-0.15, 0.0)
        grad_a = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, h_a)
        grad_b = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, h_b)
        volume_term = ((m_a / rho_a)^2 + (m_b / rho_b)^2) / m_a

        function formulation_dv(pressure_acceleration)
            system = edac_system(pair; pressure_acceleration,
                                 average_pressure_reduction=false)
            dv_ode, semi = rhs(system; smoothing_lengths=(h_a, h_b))

            return TrixiParticles.wrap_v(dv_ode, semi.systems[1], semi), semi
        end

        @testset "Default Formulation" begin
            function formulation(; kwargs...)
                system = EntropicallyDampedSPHSystem(pair; smoothing_kernel,
                                                     smoothing_length=0.12, sound_speed,
                                                     kwargs...)
                return system.pressure_acceleration_formulation
            end

            @test formulation() === inter_particle
            @test formulation(particle_refinement=refinement) === summation_density
            @test formulation(particle_refinement=refinement,
                              density_calculator=ContinuityDensity()) === summation_density
        end

        @testset "Explicit Formulation is Used" begin
            system = edac_system(pair; pressure_acceleration=inter_particle)
            @test system.pressure_acceleration_formulation === inter_particle
        end

        @testset "Nothing Chooses the Formulation of the Density Calculator" begin
            for (density_calculator, expected) in ((SummationDensity(), summation_density),
                 (ContinuityDensity(),
                  continuity_density))
                for particle_refinement in (nothing, refinement)
                    system = EntropicallyDampedSPHSystem(pair; smoothing_kernel,
                                                         smoothing_length=0.12, sound_speed,
                                                         density_calculator,
                                                         pressure_acceleration=nothing,
                                                         particle_refinement)
                    @test system.pressure_acceleration_formulation === expected
                end
            end
        end

        @testset "Formulation Without Two Gradients is Rejected" begin
            error_str = "when using particle refinement, the passed pressure " *
                        "acceleration formulation must provide a version with the " *
                        "arguments `m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, W_b`"

            @test_throws ArgumentError(error_str) edac_system(pair;
                                                              pressure_acceleration=TrixiParticles.tensile_instability_control)
        end

        @testset "Inter-Particle Averaged Pressure With Symmetric Gradient" begin
            W_a = SVector(1.0, 2.0)

            @test inter_particle(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, -W_a) ≈
                  inter_particle(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
        end

        @testset "Summation Density" begin
            dv, _ = formulation_dv(summation_density)
            expected = -m_b * (p_a / (rho_a^2 * beta_a) * grad_a +
                        p_b / (rho_b^2 * beta_b) * grad_b)

            @test isapprox(dv[1, 1], expected[1], rtol=1e-12)
        end

        @testset "Continuity Density" begin
            dv, _ = formulation_dv(continuity_density)
            expected = -m_b / (rho_a * rho_b) *
                       (p_a / beta_a * grad_a + p_b / beta_b * grad_b)

            @test !iszero(expected[1])
            @test isapprox(dv[1, 1], expected[1], rtol=1e-12)
        end

        @testset "Inter-Particle Averaged Pressure" begin
            dv, semi = formulation_dv(inter_particle)
            expected = -volume_term *
                       (rho_b * p_a / beta_a * grad_a + rho_a * p_b / beta_b * grad_b) /
                       (rho_a + rho_b)

            @test !iszero(expected[1])
            @test isapprox(dv[1, 1], expected[1], rtol=1e-12)
            @test isapprox(m_a * dv[1, 1], -m_b * dv[1, 2], rtol=1e-12)

            @testset "Pressure Evolution" begin
                nu_edac = semi.systems[1].nu_edac
                h_ref = TrixiParticles.initial_smoothing_length(semi.systems[1])
                eta_a = rho_a * nu_edac * h_a / h_ref
                eta_b = rho_b * nu_edac * h_b / h_ref
                eta_ab = 2 * eta_a * eta_b / (eta_a + eta_b)
                grad_avg = (grad_a + grad_b) / 2

                eos = reference_density * sound_speed^2 / beta_a * (m_b / rho_b) *
                      dot(SVector(2.0, 0.0), grad_a)
                damping = volume_term * eta_ab * (p_a - p_b) *
                          dot(grad_avg, pos_diff) / (0.15^2 + ((h_a + h_b) / 2)^2 / 100)

                @test !iszero(damping)
                @test isapprox(dv[3, 1], eos + damping, rtol=1e-12)
            end
        end
    end

    # As in the code of Haftu et al., the viscosity is not divided by `beta`
    @testset verbose=true "Viscosity is Not Scaled by Beta" begin
        viscosity = ViscosityAdami(nu=1e-3)
        no_pressure = InitialCondition(; coordinates, mass, density, velocity,
                                       particle_spacing=0.1)

        dv_ode,
        semi = rhs(edac_system(no_pressure; viscosity,
                               average_pressure_reduction=false))
        dv = TrixiParticles.wrap_v(dv_ode, semi.systems[1], semi)

        # The same pair without refinement has `beta = 1` and uses the same gradient
        constant = EntropicallyDampedSPHSystem(no_pressure; smoothing_kernel,
                                               smoothing_length=0.12, sound_speed,
                                               density_calculator=ContinuityDensity(),
                                               viscosity,
                                               average_pressure_reduction=false)
        dv_constant_ode, semi_constant = rhs(constant)
        dv_constant = TrixiParticles.wrap_v(dv_constant_ode, semi_constant.systems[1],
                                            semi_constant)

        @test !iszero(dv_constant[1, 1])
        @test !isapprox(semi.systems[1].cache.beta[1], 1)
        @test isapprox(dv[1, 1], dv_constant[1, 1], rtol=1e-12)
    end

    # `EDACEquationSolids` and `EDACEquationSolids2` in the code of Haftu et al.
    @testset verbose=true "Damping Term With Wall Neighbors" begin
        h_a, h_b = 0.1, 0.12
        m_a, m_b = 1.0, 0.5
        rho_a, rho_b = 1100.0, 1000.0
        p_a, p_b = 50.0, 20.0
        pos_diff = SVector(0.15, 0.0)
        grad_a = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, h_a)
        grad_b = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, 0.15, h_b)
        grad_avg = (grad_a + grad_b) / 2

        fluid = InitialCondition(; coordinates=[0.15; 0.0;;], mass=[m_a], density=[rho_a],
                                 particle_spacing=0.1)
        wall = InitialCondition(; coordinates=[0.0; 0.0;;], mass=[m_b], density=[rho_b],
                                particle_spacing=0.1)
        state_equation = StateEquationCole(; sound_speed, reference_density, exponent=7)
        boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, h_b; state_equation)

        # Only the damping term remains without a velocity difference
        function damping_term(pressure_acceleration)
            system = edac_system(fluid; pressure_acceleration,
                                 average_pressure_reduction=false)
            _,
            semi = rhs(system, WallBoundarySystem(wall, boundary_model);
                       smoothing_lengths=(h_a,))
            fluid_system, wall_system = semi.systems

            dv = zeros(3, 1)
            TrixiParticles.pressure_evolution!(dv, fluid_system, wall_system,
                                               SVector(0.0, 0.0), grad_avg, 1, 1,
                                               pos_diff, 0.15, sound_speed, m_a, m_b,
                                               p_a, p_b, rho_a, rho_b,
                                               fluid_system.nu_edac)

            return dv[3, 1], fluid_system
        end

        denominator = 0.15^2 + ((h_a + h_b) / 2)^2 / 100

        @testset "Eq. 6 Uses the Kernel Gradient of the Particle Without Beta" begin
            damping,
            system = damping_term(TrixiParticles.pressure_acceleration_summation_density)

            h_ref = TrixiParticles.initial_smoothing_length(system)
            nu_a = system.nu_edac * h_a / h_ref
            nu_b = system.nu_edac * h_b / h_ref
            nu_ab = 4 * nu_a * nu_b / (nu_a + nu_b)
            expected = (m_b / rho_b) * nu_ab * (p_a - p_b) * dot(grad_a, pos_diff) /
                       denominator

            @test !isapprox(system.cache.beta[1], 1)
            @test !iszero(expected)
            @test isapprox(damping, expected, rtol=1e-12)
        end

        @testset "Inter-Particle Averaged Pressure Uses the Kernel Gradient of the Particle" begin
            damping, system = damping_term(TrixiParticles.inter_particle_averaged_pressure)

            h_ref = TrixiParticles.initial_smoothing_length(system)
            eta_a = rho_a * system.nu_edac * h_a / h_ref
            eta_b = rho_b * system.nu_edac * h_b / h_ref
            eta_ab = 2 * eta_a * eta_b / (eta_a + eta_b)
            volume_term = ((m_a / rho_a)^2 + (m_b / rho_b)^2) / m_a
            expected = volume_term * eta_ab * (p_a - p_b) * dot(grad_a, pos_diff) /
                       denominator

            @test !iszero(expected)
            @test isapprox(damping, expected, rtol=1e-12)
        end
    end

    @testset verbose=true "Boundary Models and Structures" begin
        state_equation = StateEquationCole(; sound_speed, reference_density, exponent=7)
        fluid = InitialCondition(; coordinates=[0.15; 0.0;;], mass=[10.0], density=[1100.0],
                                 pressure=[50.0], velocity=[1.0; 0.0;;],
                                 particle_spacing=0.1)
        boundary = InitialCondition(; coordinates=[0.0; 0.0;;], mass=[10.0],
                                    density=[1000.0], particle_spacing=0.1)

        function fluid_dv(systems...)
            dv_ode, semi = rhs(systems...)
            return collect(TrixiParticles.wrap_v(dv_ode, semi.systems[1], semi))
        end

        @testset "Monaghan-Kajtar Wall Applies the Boundary Force" begin
            boundary_model = BoundaryModelMonaghanKajtar(10.0, 1.0, 0.1, [10.0])
            boundary_system = WallBoundarySystem(boundary, boundary_model)

            dv = fluid_dv(edac_system(fluid; average_pressure_reduction=false),
                          boundary_system)

            expected = TrixiParticles.pressure_acceleration(edac_system(fluid),
                                                            boundary_system, 1, 1,
                                                            10.0, 10.0, 50.0, 0.0,
                                                            1100.0, 1000.0,
                                                            SVector(0.15, 0.0), 0.15,
                                                            SVector(0.0, 0.0), nothing)

            @test expected[1] > 0
            @test isapprox(dv[1, 1], expected[1], rtol=1e-12)
        end

        @testset "TLSPH Uses the Smoothing Length of the Boundary Model" begin
            function dv(smoothing_length_structure)
                structure = RectangularShape(0.1, (3, 3), (-0.2, -0.1); density=1000.0)
                boundary_model = BoundaryModelDummyParticles(structure.density,
                                                             structure.mass,
                                                             AdamiPressureExtrapolation(),
                                                             smoothing_kernel, 0.12;
                                                             state_equation)
                structure_system = TotalLagrangianSPHSystem(structure; smoothing_kernel,
                                                            smoothing_length=smoothing_length_structure,
                                                            young_modulus=1e4,
                                                            poisson_ratio=0.3,
                                                            boundary_model)

                return fluid_dv(edac_system(fluid), structure_system)
            end

            @test any(!iszero, dv(0.12))
            @test isapprox(dv(0.2), dv(0.12), rtol=1e-12)
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

                dv_ode,
                semi = rhs(edac_system(fluid; average_pressure_reduction=false),
                           rigid_system; smoothing_lengths=(h_fluid,))
                dv = TrixiParticles.wrap_v(dv_ode, semi.systems[1], semi)

                return 10.0 * dv[1, 1], semi.systems[2].force_per_particle[1, 1]
            end

            # The distance of 0.15 is inside and outside of the fluid's compact support
            for h_fluid in (0.1, 0.06)
                force_fluid, force_rigid = forces(h_fluid)

                @test !iszero(force_fluid)
                @test isapprox(force_fluid, -force_rigid, rtol=1e-12)
            end
        end
    end

    @testset "Corrections Are Rejected" begin
        system = EntropicallyDampedSPHSystem(pair; smoothing_kernel, smoothing_length=0.12,
                                             sound_speed,
                                             correction=ShepardKernelCorrection(),
                                             particle_refinement=refinement)
        @test_throws "corrections are not supported with particle refinement" Semidiscretization(system)
    end
end
