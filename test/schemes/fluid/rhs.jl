@testset verbose=true "RHS" begin
    @testset verbose=true "`pressure_acceleration`" begin
        # Use `@trixi_testset` to isolate the mock functions in a separate namespace
        @trixi_testset "Symmetry" begin
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
            masses = [[0.01, 0.01], [0.73, 0.31]]
            densities = [
                [1000.0, 1000.0],
                [1000.0, 1000.0],
                [900.0, 1201.0],
                [1003.0, 353.4]
            ]
            pressures = [
                [0.0, 0.0],
                [10_000.0, 10_000.0],
                [10.0, 10_000.0],
                [1000.0, -1000.0]
            ]
            grad_kernels = [0.3, 104.0]
            particle = 2
            neighbor = 3

            # Not used for fluid-fluid interaction
            pos_diff = 0
            distance = 0

            density_calculators = [ContinuityDensity(), SummationDensity()]

            pressure_accelerations = [
                TrixiParticles.inter_particle_averaged_pressure,
                TrixiParticles.pressure_acceleration_continuity_density,
                TrixiParticles.pressure_acceleration_summation_density
            ]

            # Partly copied from constructor test, just to create a WCSPH system
            coordinates = zeros(2, 3)
            velocity = zeros(2, 3)
            mass = zeros(3)
            density = ones(3)
            state_equation = Val(:state_equation)
            smoothing_kernel = Val(:smoothing_kernel)
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
            smoothing_length = -1.0

            fluid = InitialCondition(; coordinates, velocity, mass, density)

            @testset "`$(nameof(typeof(density_calculator)))`" for density_calculator in
                                                                   density_calculators

                @testset "`$(nameof(typeof(pressure_acceleration)))`" for pressure_acceleration in
                                                                          pressure_accelerations

                    for (m_a, m_b) in masses, (rho_a, rho_b) in densities,
                        (p_a, p_b) in pressures, grad_kernel in grad_kernels
                        @testset verbose=true "$system_name" for system_name in [
                            "WCSPH",
                            "EDAC",
                            "IISPH"
                        ]
                            if system_name == "WCSPH"
                                system = WeaklyCompressibleSPHSystem(fluid;
                                                                     smoothing_kernel,
                                                                     smoothing_length,
                                                                     density_calculator,
                                                                     state_equation,
                                                                     pressure_acceleration)
                            elseif system_name == "EDAC"
                                system = EntropicallyDampedSPHSystem(fluid;
                                                                     smoothing_kernel,
                                                                     smoothing_length,
                                                                     sound_speed=0.0,
                                                                     density_calculator,
                                                                     pressure_acceleration)
                            elseif system_name == "IISPH"
                                system = ImplicitIncompressibleSPHSystem(fluid;
                                                                         smoothing_kernel,
                                                                         smoothing_length,
                                                                         reference_density=1000.0,
                                                                         time_step=0.001)
                            end

                            # Compute accelerations a -> b and b -> a
                            dv1 = TrixiParticles.pressure_acceleration(system, system,
                                                                       -1, -1,
                                                                       m_a, m_b, p_a, p_b,
                                                                       rho_a, rho_b,
                                                                       pos_diff,
                                                                       distance,
                                                                       grad_kernel,
                                                                       nothing)

                            dv2 = TrixiParticles.pressure_acceleration(system, system,
                                                                       -1, -1,
                                                                       m_b, m_a, p_b, p_a,
                                                                       rho_b, rho_a,
                                                                       -pos_diff,
                                                                       distance,
                                                                       -grad_kernel,
                                                                       nothing)

                            # Test that both forces are identical but in opposite directions
                            @test isapprox(m_a * dv1, -m_b * dv2, rtol=2eps())
                        end
                    end
                end
            end
        end
    end

    # The following tests for linear and angular momentum and total energy conservation
    # are based on Sections 3.3.4 and 3.4.2 of
    # Daniel J. Price. "Smoothed Particle Hydrodynamics and Magnetohydrodynamics."
    # In: Journal of Computational Physics 231.3 (2012), pages 759–94.
    # https://doi.org/10.1016/j.jcp.2010.12.011
    @testset verbose=true "Momentum and Total Energy Conservation" begin
        # We are testing the momentum conservation of SPH with random initial configurations
        density_calculators = [ContinuityDensity(), SummationDensity()]

        particle_spacing = 0.1

        # The state equation is only needed to unpack `sound_speed`, so we can mock
        # it by using a `NamedTuple`.
        state_equation = (; sound_speed=0.0)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2 * particle_spacing
        search_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)

        @testset "`$(nameof(typeof(density_calculator)))`" for density_calculator in
                                                               density_calculators
            # Run three times with different seed for the random initial condition
            for seed in 1:3
                # A larger number of particles will increase accumulated errors in the
                # summation. A larger tolerance has to be used for the tests below.
                fluid = rectangular_patch(particle_spacing, (3, 3); seed)
                system_wcsph = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                           smoothing_length,
                                                           density_calculator,
                                                           state_equation)

                system_edac = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                          smoothing_length,
                                                          sound_speed=0.0,
                                                          pressure_acceleration=nothing,
                                                          density_calculator)

                system_iisph = ImplicitIncompressibleSPHSystem(fluid; smoothing_kernel,
                                                               smoothing_length,
                                                               reference_density=1000.0,
                                                               time_step=0.001)

                n_particles = TrixiParticles.nparticles(system_edac)

                # Overwrite `system.pressure` because we skip the update step
                system_wcsph.pressure .= fluid.pressure
                system_iisph.pressure .= fluid.pressure
                if density_calculator isa TrixiParticles.SummationDensity
                    systems = (system_wcsph, system_edac, system_iisph)
                else
                    # IISPH is always using `SummationDensity``
                    systems = (system_wcsph, system_edac)
                end
                @testset "`$(nameof(typeof(system)))`" for system in systems
                    u = fluid.coordinates
                    if density_calculator isa SummationDensity
                        # Density is stored in the cache
                        v = fluid.velocity
                        if system isa WeaklyCompressibleSPHSystem
                            system.cache.density .= fluid.density
                        elseif system isa EntropicallyDampedSPHSystem
                            # Pressure is integrated
                            system.cache.density .= fluid.density
                            v = vcat(fluid.velocity, fluid.pressure')
                        else
                            system.density .= fluid.density
                        end
                    else
                        # Density is integrated with `ContinuityDensity`

                        if system isa EntropicallyDampedSPHSystem
                            v = vcat(fluid.velocity, fluid.pressure', fluid.density')
                        else
                            v = vcat(fluid.velocity, fluid.density')
                        end
                    end

                    semi = DummySemidiscretization()

                    # Result
                    dv = zero(v)
                    TrixiParticles.interact!(dv, v, u, v, u, system, system, semi)

                    # Linear momentum conservation
                    # ∑ m_a dv_a
                    deriv_linear_momentum = sum(fluid.mass' .* view(dv, 1:2, :), dims=2)

                    @test isapprox(deriv_linear_momentum, zeros(2, 1), atol=5e-14)

                    # Angular momentum conservation
                    # m_a (r_a × dv_a)
                    function deriv_angular_momentum(particle)
                        r_a = SVector(u[1, particle], u[2, particle], 0.0)
                        dv_a = SVector(dv[1, particle], dv[2, particle], 0.0)

                        return fluid.mass[particle] * cross(r_a, dv_a)
                    end

                    # ∑ m_a (r_a × dv_a)
                    deriv_angular_momentum = sum(deriv_angular_momentum, 1:n_particles)

                    # Cross product is always 3-dimensional
                    @test isapprox(deriv_angular_momentum, zeros(3), atol=4e-15)

                    # Total energy conservation
                    function drho(::ContinuityDensity, ::TrixiParticles.AbstractFluidSystem,
                                  particle)
                        return dv[end, particle]
                    end

                    function drho(::SummationDensity, system, particle)
                        return sum(neighbor -> drho_particle(particle, neighbor),
                                   1:n_particles)
                    end

                    # Derivative of the density summation. This is a slightly different
                    # formulation of the continuity equation.
                    function drho_particle(particle, neighbor)
                        m_b = TrixiParticles.hydrodynamic_mass(system, neighbor)
                        v_diff = TrixiParticles.current_velocity(v, system, particle) -
                                 TrixiParticles.current_velocity(v, system, neighbor)

                        pos_diff = TrixiParticles.current_coords(u, system, particle) -
                                   TrixiParticles.current_coords(u, system, neighbor)
                        distance = norm(pos_diff)

                        # Only consider particles with a distance > 0
                        distance < sqrt(eps()) && return 0.0

                        grad_kernel = TrixiParticles.smoothing_kernel_grad(system, pos_diff,
                                                                           distance,
                                                                           particle)

                        return m_b * dot(v_diff, grad_kernel)
                    end

                    # m_a (v_a ⋅ dv_a + dte_a),
                    # where `te` is the thermal energy, called `u` in the Price paper.
                    function deriv_energy(particle)
                        p_a = fluid.pressure[particle]
                        rho_a = fluid.density[particle]
                        dte_a = p_a / rho_a^2 * drho(density_calculator, system, particle)
                        v_a = TrixiParticles.extract_svector(v, system, particle)
                        dv_a = TrixiParticles.extract_svector(dv, system, particle)

                        return fluid.mass[particle] * (dot(v_a, dv_a) + dte_a)
                    end

                    # ∑ m_a (v_a ⋅ dv_a + dte_a)
                    deriv_total_energy = sum(deriv_energy, 1:n_particles)

                    @test isapprox(deriv_total_energy, 0.0, atol=6e-15)
                end
            end
        end
    end

    # The force that a structure particle experiences from a fluid particle must be
    # exactly the opposite of the force that the fluid particle experiences from the
    # structure particle, except for the extra terms of shifting techniques in the momentum
    # equation, which must not be applied to the structure.
    # See the comment in `interact_structure_fluid!` for an explanation.
    @testset verbose=true "Fluid-Structure Interaction Forces" begin
        # A single fluid particle at the origin and a single structure particle at rest
        # at `(1.5h, 0)`, where the kernel gradient is nonzero.
        # This isolates the fluid-structure pair force from all other forces.
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.0
        particle_spacing = 1.0
        fluid_density = 1000.0
        sound_speed = 10.0
        state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                           exponent=1)

        function create_fluid_system(scheme, velocity, density, viscosity;
                                     shifting_technique=nothing, correction=nothing)
            fluid = InitialCondition(; coordinates=zeros(2, 1),
                                     velocity=reshape(collect(velocity), 2, 1),
                                     mass=[fluid_density * particle_spacing^2],
                                     density=[density], pressure=state_equation(density),
                                     particle_spacing)

            if scheme == "WCSPH"
                return WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length, state_equation,
                                                   density_calculator=ContinuityDensity(),
                                                   viscosity, shifting_technique,
                                                   correction)
            end

            # Note that the average pressure reduction is enabled by default
            # when using shifting.
            average_pressure_reduction = scheme == "EDAC with average pressure reduction"
            return EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                               sound_speed, viscosity, shifting_technique,
                                               average_pressure_reduction,
                                               density_calculator=ContinuityDensity())
        end

        function create_structure_system(structure_type, viscosity;
                                         boundary_density=AdamiPressureExtrapolation(),
                                         clamped=false,
                                         elastic_kernel=smoothing_kernel,
                                         elastic_smoothing_length=smoothing_length)
            # The material mass is twice the hydrodynamic mass, so using the wrong mass
            # to convert the force on the structure to an acceleration fails the test.
            structure = InitialCondition(; coordinates=reshape([1.5, 0.0], 2, 1),
                                         velocity=zeros(2, 1),
                                         mass=[2 * fluid_density * particle_spacing^2],
                                         density=[2 * fluid_density], particle_spacing)
            boundary_model = BoundaryModelDummyParticles([fluid_density],
                                                         [fluid_density *
                                                          particle_spacing^2],
                                                         boundary_density,
                                                         smoothing_kernel,
                                                         smoothing_length;
                                                         state_equation, viscosity)

            if structure_type === TotalLagrangianSPHSystem
                return TotalLagrangianSPHSystem(structure; smoothing_kernel=elastic_kernel,
                                                smoothing_length=elastic_smoothing_length,
                                                young_modulus=1e5,
                                                poisson_ratio=0.3, boundary_model,
                                                clamped_particles=clamped ? (1:1) : (1:0))
            end

            return RigidBodySystem(structure; boundary_model)
        end

        # Run the regular update step to compute the boundary pressure, the wall velocity
        # for the viscosity, and the average pressure of EDAC. Return the systems
        # stored in the semidiscretization and the wrapped arrays.
        function initialize(fluid_system, structure_system)
            semi = if structure_system isa TotalLagrangianSPHSystem
                @test_logs (:info,
                            r"^To create the self-interaction neighborhood search of a `TotalLagrangianSPHSystem`") begin
                    Semidiscretization(fluid_system, structure_system;
                                       parallelization_backend=SerialBackend())
                end
            else
                @test_logs Semidiscretization(fluid_system, structure_system;
                                              parallelization_backend=SerialBackend())
            end
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)

            fluid, structure = ode.p.semi.systems
            semi = ode.p.semi
            arrays = (TrixiParticles.wrap_v(v_ode, fluid, semi),
                      TrixiParticles.wrap_u(u_ode, fluid, semi),
                      TrixiParticles.wrap_v(v_ode, structure, semi),
                      TrixiParticles.wrap_u(u_ode, structure, semi))

            return fluid, structure, arrays, semi, ode
        end

        # Force on the fluid particle and force on the structure particle
        function pair_forces(fluid, structure, arrays, semi)
            v_fluid, u_fluid, v_structure, u_structure = arrays

            dv_fluid = zero(v_fluid)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            force_fluid = fluid.mass[1] * dv_fluid[1:2, 1]

            if structure isa RigidBodySystem
                structure.force_per_particle .= 0
            end

            # Include clamped TLSPH particles when checking their hydrodynamic loads.
            dv_structure = zeros(eltype(v_structure), size(v_structure, 1),
                                 nparticles(structure))
            if structure isa TotalLagrangianSPHSystem
                TrixiParticles.interact!(dv_structure, v_structure, u_structure,
                                         v_fluid, u_fluid, structure, fluid, semi;
                                         eachparticle=eachparticle(structure))
            else
                TrixiParticles.interact!(dv_structure, v_structure, u_structure,
                                         v_fluid, u_fluid, structure, fluid, semi)
            end

            if structure isa RigidBodySystem
                # The rigid body accumulates the forces per particle
                force_structure = structure.force_per_particle[:, 1]
            else
                # TLSPH accelerations are scaled by the material mass
                force_structure = structure.mass[1] * dv_structure[1:2, 1]
            end

            return force_fluid, force_structure
        end

        # Exercise all viscosity models with WCSPH and both EDAC pressure variants.
        configurations = [
            ("WCSPH", nothing),
            ("WCSPH", ViscosityAdami(nu=0.1)),
            ("WCSPH", ViscosityMorris(nu=0.1)),
            ("WCSPH", ArtificialViscosityMonaghan(alpha=0.1, beta=0.2)),
            ("EDAC", nothing),
            ("EDAC", ViscosityAdami(nu=0.1)),
            ("EDAC", ViscosityMorris(nu=0.1)),
            ("EDAC", ArtificialViscosityMonaghan(alpha=0.1, beta=0.2)),
            ("EDAC with average pressure reduction", nothing),
            ("EDAC with average pressure reduction", ViscosityAdami(nu=0.1)),
            ("EDAC with average pressure reduction", ViscosityMorris(nu=0.1)),
            ("EDAC with average pressure reduction",
             ArtificialViscosityMonaghan(alpha=0.1, beta=0.2))
        ]

        # The structure lies to the right of the fluid particle, so a positive
        # x-velocity of the fluid particle approaches the structure.
        # With the reference density, the pressure is zero, which isolates the viscosity.
        fluid_states = [
            (name="approaching, zero pressure", velocity=(1.0, 0.5), density=1000.0),
            (name="receding, zero pressure", velocity=(-1.0, -0.5), density=1000.0),
            (name="approaching, positive pressure", velocity=(1.0, 0.5), density=1005.0)
        ]

        structure_types = (TotalLagrangianSPHSystem, RigidBodySystem)
        structure_configurations = ((TotalLagrangianSPHSystem, false),
                                    (TotalLagrangianSPHSystem, true),
                                    (RigidBodySystem, false))
        correction_methods = (nothing, AkinciFreeSurfaceCorrection(fluid_density),
                              KernelCorrection(), GradientCorrection(),
                              BlendedGradientCorrection(0.5),
                              MixedKernelGradientCorrection())

        @testset "$(config[1]), Viscosity `$(nameof(typeof(config[2])))`" for config in
                                                                              configurations

            scheme, viscosity = config
            corrections = scheme == "WCSPH" ? correction_methods : (nothing,)

            @testset "Correction `$(nameof(typeof(correction)))`" for correction in
                                                                      corrections

                @testset "`$structure_type`, clamped=$clamped" for (structure_type,
                                                                    clamped) in
                                                                   structure_configurations

                    boundary_densities = if correction isa KernelCorrection ||
                                            correction isa MixedKernelGradientCorrection
                        (AdamiPressureExtrapolation(), SummationDensity(),
                         PressureMirroring())
                    elseif scheme == "WCSPH" && isnothing(correction) &&
                           structure_type === RigidBodySystem
                        (AdamiPressureExtrapolation(), ContinuityDensity(),
                         PressureMirroring())
                    else
                        (AdamiPressureExtrapolation(), PressureMirroring())
                    end

                    @testset "Boundary `$(nameof(typeof(boundary_density)))`" for boundary_density in
                                                                                  boundary_densities

                        @testset "$(state.name)" for state in fluid_states
                            fluid_system = create_fluid_system(scheme, state.velocity,
                                                               state.density, viscosity;
                                                               correction)
                            structure_system = create_structure_system(structure_type,
                                                                       viscosity;
                                                                       boundary_density,
                                                                       clamped)
                            (fluid, structure, arrays,
                             semi, ode) = initialize(fluid_system, structure_system)
                            # Isolate the pair using the same updated correction and
                            # boundary-interpolation caches as the full RHS.
                            v_ode, u_ode = ode.u0.x
                            dv_ode = zero(v_ode)
                            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
                            (force_fluid,
                             force_structure) = pair_forces(fluid, structure, arrays, semi)
                            v_fluid, _, _, _ = arrays

                            if correction isa KernelCorrection ||
                               correction isa MixedKernelGradientCorrection
                                # Verify that reversing the displacement does not just
                                # change the sign of the corrected gradient in this fixture.
                                pos_diff = SVector(1.5, 0.0)
                                grad_kernel = TrixiParticles.smoothing_kernel_grad_unsafe(fluid,
                                                                                          pos_diff,
                                                                                          1.5,
                                                                                          1)
                                grad_kernel_fluid = TrixiParticles.smoothing_kernel_grad_unsafe(fluid,
                                                                                                -pos_diff,
                                                                                                1.5,
                                                                                                1)
                                @test !isapprox(grad_kernel_fluid, -grad_kernel)
                            end

                            # Newton's third law, including hydrodynamic/material mass conversion.
                            @test isapprox(force_structure, -force_fluid,
                                           rtol=sqrt(eps()), atol=sqrt(eps()))

                            if state.density == fluid_density
                                @test iszero(TrixiParticles.current_pressure(v_fluid, fluid,
                                                                             1))
                                @test iszero(structure.boundary_model.pressure[1])
                                if isnothing(viscosity) ||
                                   (viscosity isa ArtificialViscosityMonaghan &&
                                    state.velocity[1] < 0)
                                    # No pressure, and Monaghan viscosity is inactive for receding particles.
                                    @test iszero(force_structure)
                                else
                                    # The fluid drags the structure along.
                                    @test force_structure[1] * state.velocity[1] > 0
                                end
                            elseif isnothing(viscosity)
                                if scheme == "EDAC with average pressure reduction"
                                    # `PressureMirroring` leaves the cache at its
                                    # initial value, so only compare for Adami.
                                    if boundary_density isa AdamiPressureExtrapolation
                                        @test TrixiParticles.current_pressure(v_fluid,
                                                                              fluid,
                                                                              1) ≈
                                              structure.boundary_model.pressure[1]
                                    end
                                    @test isapprox(force_structure, zeros(2),
                                                   atol=sqrt(eps()))
                                else
                                    @test force_structure[1] > 0
                                end
                            else
                                @test !iszero(force_fluid)
                            end

                            # Also check that the normal RHS carries the pair reaction
                            # through to integrated TLSPH particles and rigid-body resultants.
                            dv_structure = TrixiParticles.wrap_v(dv_ode, structure, semi)
                            if structure isa RigidBodySystem
                                @test isapprox(structure.resultant_force[], -force_fluid,
                                               rtol=sqrt(eps()), atol=sqrt(eps()))
                            elseif !clamped
                                @test isapprox(structure.mass[1] * dv_structure[1:2, 1],
                                               -force_fluid,
                                               rtol=sqrt(eps()), atol=sqrt(eps()))
                            end

                            if boundary_density isa ContinuityDensity
                                # Independently evaluate the structure-first continuity
                                # operator using the cubic-spline derivative at r/h = 1.5.
                                kernel_derivative = -0.75 * (2 - 1.5)^2 * 10 / (7pi)
                                expected_density_rate = -fluid_density / state.density *
                                                        fluid.mass[1] * state.velocity[1] *
                                                        kernel_derivative
                                @test dv_structure[end, 1] ≈ expected_density_rate
                            end
                        end
                    end
                end
            end
        end

        @testset "Boundary kernel in corrected pressure" begin
            for structure_type in structure_types,
                correction in (KernelCorrection(), MixedKernelGradientCorrection())
                fluid_system = create_fluid_system("WCSPH", (0.0, 0.0), 1005.0, nothing;
                                                   correction)
                # r=1.5 is inside the common boundary/fluid support of 2.0, but
                # outside the elastic TLSPH support of 2*0.4.
                structure_system = create_structure_system(structure_type, nothing;
                                                           boundary_density=PressureMirroring(),
                                                           elastic_kernel=WendlandC2Kernel{2}(),
                                                           elastic_smoothing_length=0.4)
                fluid, structure, arrays, semi,
                _ = initialize(fluid_system, structure_system)
                force_fluid, force_structure = pair_forces(fluid, structure, arrays, semi)
                # Independent cubic-spline gradients, including the fluid's kernel
                # correction. The collinear gradient-correction matrix is the identity.
                grad_s = SVector(-15 / (56pi), 0.0)
                gamma_f = fluid_density / 1005.0 * 10 / (7pi) + 5 / (112pi)
                dw_gamma_f = -grad_s / gamma_f
                grad_f = (-grad_s - 5 / (112pi) * dw_gamma_f) / gamma_f
                expected = 500.0 * fluid_density / 1005.0 * (grad_f - grad_s)
                @test force_structure ≈ expected
                @test force_fluid ≈ -expected

                # The elastic kernel may differ, but boundary/fluid support must match.
                structure = TrixiParticles.@set structure.boundary_model.smoothing_length = 0.5
                @test_throws ArgumentError Semidiscretization(fluid, structure)
            end
        end

        # The extra terms of the shifting techniques in the momentum equation act on the
        # fluid, but must not be applied to the structure.
        shifting_techniques = (ParticleShiftingTechnique(),
                               TransportVelocityAdami(background_pressure=1000.0))

        @testset "Shifting `$(nameof(typeof(shifting_technique)))`" for shifting_technique in
                                                                        shifting_techniques

            @testset "`$structure_type`" for structure_type in structure_types
                viscosity = ViscosityAdami(nu=0.1)
                fluid_system = create_fluid_system("WCSPH", (1.0, 0.5), 1005.0,
                                                   viscosity; shifting_technique)
                structure_system = create_structure_system(structure_type, viscosity)
                fluid, structure, arrays, semi,
                _ = initialize(fluid_system, structure_system)

                # Compare the forces without and with a shifting velocity
                fluid.cache.delta_v .= 0
                force_fluid, force_structure = pair_forces(fluid, structure, arrays, semi)

                fluid.cache.delta_v .= [0.5, 0.2]
                (force_fluid_shifting,
                 force_structure_shifting) = pair_forces(fluid, structure, arrays, semi)

                # Make sure that the shifting terms act on the fluid.
                @test !isapprox(force_fluid_shifting, force_fluid)

                # The shifting terms must not be applied to the structure.
                @test force_structure_shifting == force_structure
            end
        end
    end
end
