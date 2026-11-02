@testset "Cross-system update ordering" begin
    # Two overlapping fluids with different corrections, where one system's update
    # depends on the final density of the other. This verifies that the globally staged
    # update in `update_systems_and_nhs` produces the same results independent of the
    # order in which the systems are passed to the `Semidiscretization`.
    function ordered_correction_result(reverse_order; edac)
        spacing = 0.1
        smoothing_length = 2 * spacing
        smoothing_kernel = WendlandC6Kernel{2}()
        density = 1000.0
        velocity(pos) = SVector(0.1 + pos[1], -0.2 - pos[2])
        pressure(pos) = 1.0 + 2pos[1] - pos[2]

        # Offset the second block so that both systems interact through their
        # neighborhood search.
        gradient_initial = RectangularShape(spacing, (3, 3), (0.0, 0.0);
                                            density, velocity, pressure)
        shepard_initial = RectangularShape(spacing, (3, 3), (0.05, 0.025);
                                           density, velocity, pressure)

        if edac
            gradient_system = EntropicallyDampedSPHSystem(gradient_initial;
                                                          smoothing_kernel,
                                                          smoothing_length,
                                                          sound_speed=10.0,
                                                          pressure_acceleration=nothing,
                                                          density_calculator=ContinuityDensity(),
                                                          correction=GradientCorrection())
            shepard_system = EntropicallyDampedSPHSystem(shepard_initial;
                                                         smoothing_kernel,
                                                         smoothing_length,
                                                         sound_speed=10.0,
                                                         density_calculator=SummationDensity(),
                                                         correction=ShepardKernelCorrection())
        else
            state_equation = StateEquationCole(; sound_speed=10.0,
                                               reference_density=density, exponent=1)
            gradient_system = WeaklyCompressibleSPHSystem(gradient_initial;
                                                          smoothing_kernel,
                                                          smoothing_length,
                                                          state_equation,
                                                          density_calculator=ContinuityDensity(),
                                                          correction=GradientCorrection())
            shepard_system = WeaklyCompressibleSPHSystem(shepard_initial;
                                                         smoothing_kernel,
                                                         smoothing_length,
                                                         state_equation,
                                                         density_calculator=SummationDensity(),
                                                         correction=ShepardKernelCorrection())
        end

        # Vary the system order to check that correction staging is order-independent.
        systems = reverse_order ? (shepard_system, gradient_system) :
                  (gradient_system, shepard_system)
        semi = Semidiscretization(systems...; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        dv_ode = zero(v_ode)
        # Evaluate the RHS, which triggers the staged correction updates.
        TrixiParticles.kick!(dv_ode, v_ode, u_ode,
                             (; semi=ode.p.semi, split_integration_data=nothing), 0.0)

        # Recover the systems from the semidiscretization, since the order is swapped.
        gradient_system = only(system
                               for system in ode.p.semi.systems
                               if system.correction isa GradientCorrection)
        shepard_system = only(system
                              for system in ode.p.semi.systems
                              if system.correction isa ShepardKernelCorrection)
        GC.@preserve v_ode dv_ode begin
            v_gradient = TrixiParticles.wrap_v(v_ode, gradient_system, ode.p.semi)
            v_shepard = TrixiParticles.wrap_v(v_ode, shepard_system, ode.p.semi)
            dv_gradient = TrixiParticles.wrap_v(dv_ode, gradient_system, ode.p.semi)
            dv_shepard = TrixiParticles.wrap_v(dv_ode, shepard_system, ode.p.semi)

            return (;
                    gradient_density=copy(TrixiParticles.current_density(v_gradient,
                                                                         gradient_system)),
                    shepard_density=copy(TrixiParticles.current_density(v_shepard,
                                                                        shepard_system)),
                    gradient_pressure=copy(TrixiParticles.current_pressure(v_gradient,
                                                                           gradient_system)),
                    shepard_pressure=copy(TrixiParticles.current_pressure(v_shepard,
                                                                          shepard_system)),
                    correction_matrix=copy(gradient_system.cache.correction_matrix),
                    shepard_coefficient=copy(shepard_system.cache.kernel_correction_coefficient),
                    gradient_rhs=copy(dv_gradient), shepard_rhs=copy(dv_shepard))
        end
    end

    # Check all correction-coupled quantities for both WCSPH and EDAC systems.
    for edac in (false, true)
        forward = ordered_correction_result(false; edac)
        reverse = ordered_correction_result(true; edac)

        @test forward.gradient_density≈reverse.gradient_density rtol=5e-13 atol=5e-13
        @test forward.shepard_density≈reverse.shepard_density rtol=5e-13 atol=5e-13
        @test forward.gradient_pressure≈reverse.gradient_pressure rtol=5e-13 atol=5e-13
        @test forward.shepard_pressure≈reverse.shepard_pressure rtol=5e-13 atol=5e-13
        @test forward.correction_matrix≈reverse.correction_matrix rtol=5e-13 atol=5e-13
        @test forward.shepard_coefficient≈reverse.shepard_coefficient rtol=5e-13 atol=5e-13
        @test forward.gradient_rhs≈reverse.gradient_rhs rtol=1e-11 atol=1e-10
        @test forward.shepard_rhs≈reverse.shepard_rhs rtol=1e-11 atol=1e-10
    end

    # The mixed gradient/Shepard case above has only one system that mutates density and
    # therefore cannot expose sequential density correction. Use two overlapping Shepard
    # systems so reversing their declaration order would reveal coefficients assembled from
    # an already-corrected neighbor density.
    function ordered_shepard_result(reverse_order; edac)
        spacing = 0.1
        smoothing_length = 2 * spacing
        smoothing_kernel = WendlandC6Kernel{2}()
        density = 1000.0
        initial_a = RectangularShape(spacing, (3, 3), (0.0, 0.0); density)
        initial_b = RectangularShape(spacing, (3, 3), (0.05, 0.025); density)

        function make_system(initial_condition)
            if edac
                return EntropicallyDampedSPHSystem(initial_condition;
                                                   smoothing_kernel,
                                                   smoothing_length,
                                                   sound_speed=10.0,
                                                   density_calculator=SummationDensity(),
                                                   correction=ShepardKernelCorrection())
            end

            state_equation = StateEquationCole(; sound_speed=10.0,
                                               reference_density=density, exponent=1)
            return WeaklyCompressibleSPHSystem(initial_condition;
                                               smoothing_kernel,
                                               smoothing_length,
                                               state_equation,
                                               density_calculator=SummationDensity(),
                                               correction=ShepardKernelCorrection())
        end

        systems = reverse_order ? (make_system(initial_b), make_system(initial_a)) :
                  (make_system(initial_a), make_system(initial_b))
        semi = Semidiscretization(systems...; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode,
                             (; semi=ode.p.semi, split_integration_data=nothing), 0.0)

        # Compare the same physical particle clouds after reversing their tuple positions.
        index_a, index_b = reverse_order ? (2, 1) : (1, 2)
        system_a = ode.p.semi.systems[index_a]
        system_b = ode.p.semi.systems[index_b]
        v_a = TrixiParticles.wrap_v(v_ode, system_a, ode.p.semi)
        v_b = TrixiParticles.wrap_v(v_ode, system_b, ode.p.semi)
        dv_a = TrixiParticles.wrap_v(dv_ode, system_a, ode.p.semi)
        dv_b = TrixiParticles.wrap_v(dv_ode, system_b, ode.p.semi)

        return (;
                density_a=copy(TrixiParticles.current_density(v_a, system_a)),
                density_b=copy(TrixiParticles.current_density(v_b, system_b)),
                pressure_a=copy(TrixiParticles.current_pressure(v_a, system_a)),
                pressure_b=copy(TrixiParticles.current_pressure(v_b, system_b)),
                coefficient_a=copy(system_a.cache.kernel_correction_coefficient),
                coefficient_b=copy(system_b.cache.kernel_correction_coefficient),
                rhs_a=copy(dv_a), rhs_b=copy(dv_b))
    end

    # Coefficients and every quantity derived from corrected density must be independent of
    # system declaration order for both explicit pressure models.
    for edac in (false, true)
        forward = ordered_shepard_result(false; edac)
        reverse = ordered_shepard_result(true; edac)

        @test forward.density_a≈reverse.density_a rtol=5e-13 atol=5e-13
        @test forward.density_b≈reverse.density_b rtol=5e-13 atol=5e-13
        @test forward.pressure_a≈reverse.pressure_a rtol=5e-13 atol=5e-13
        @test forward.pressure_b≈reverse.pressure_b rtol=5e-13 atol=5e-13
        @test forward.coefficient_a≈reverse.coefficient_a rtol=5e-13 atol=5e-13
        @test forward.coefficient_b≈reverse.coefficient_b rtol=5e-13 atol=5e-13
        @test forward.rhs_a≈reverse.rhs_a rtol=1e-11 atol=1e-10
        @test forward.rhs_b≈reverse.rhs_b rtol=1e-11 atol=1e-10
    end
end

@testset "Boundary density before pressure" begin
    # Boundary pressure must be computed from the Shepard-corrected density, so the
    # density update has to run before pressure evaluation.
    n = 5
    particle_spacing = 1.0 / n
    smoothing_kernel = WendlandC6Kernel{2}()
    particles = RectangularShape(particle_spacing, (n, n), (0.0, 0.0); density=1000.0)
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

    # The boundary pressure must match the state equation evaluated at the
    # (Shepard-corrected) boundary density.
    @test boundary.boundary_model.pressure ≈
          state_equation.(boundary.boundary_model.cache.density)
end

@testset "Structure correction lifecycle" begin
    particle_spacing = 0.1
    smoothing_length = 2particle_spacing
    smoothing_kernel = WendlandC6Kernel{2}()
    density = 1000.0
    particles = RectangularShape(particle_spacing, (3, 3), (0.0, 0.0); density)
    state_equation = StateEquationCole(; sound_speed=10.0,
                                       reference_density=density, exponent=1)

    function structure_setup(correction)
        boundary_model = BoundaryModelDummyParticles(particles.density, particles.mass,
                                                     SummationDensity(), smoothing_kernel,
                                                     smoothing_length;
                                                     state_equation, correction)
        system = TotalLagrangianSPHSystem(particles; smoothing_kernel, smoothing_length,
                                          young_modulus=1.0e6, poisson_ratio=0.3,
                                          boundary_model)
        semi = Semidiscretization(system; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        system = first(ode.p.semi.systems)
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)

        return (; system, semi=ode.p.semi, v_ode, u_ode)
    end

    # Reconstruct the density numerator from the lifecycle result, rho_corrected * c, and
    # compare it with an independent summation. The second assertion makes the test sensitive
    # to the old behavior, where the unchanged initial density was normalized instead.
    shepard = structure_setup(ShepardKernelCorrection())
    v_shepard = TrixiParticles.wrap_v(shepard.v_ode, shepard.system, shepard.semi)
    u_shepard = TrixiParticles.wrap_u(shepard.u_ode, shepard.system, shepard.semi)
    raw_density = zeros(TrixiParticles.nparticles(shepard.system))
    TrixiParticles.summation_density!(shepard.system, shepard.semi, u_shepard,
                                      shepard.u_ode, raw_density)
    corrected_density = TrixiParticles.current_density(v_shepard, shepard.system)
    coefficient = shepard.system.boundary_model.cache.kernel_correction_coefficient
    @test corrected_density .* coefficient ≈ raw_density atol = 5e-13
    @test maximum(abs, raw_density .- density) > 1.0

    # The optimized TLSPH path assembles its density numerator alongside the Shepard
    # coefficient, but keeps the numerator in scratch storage until every system has
    # assembled its coefficient. Reverse a coupled fluid/TLSPH pair to ensure that this
    # fusion does not reintroduce declaration-order dependence.
    function ordered_fluid_structure_result(reverse_order)
        fluid_particles = RectangularShape(particle_spacing, (3, 3), (0.0, 0.0); density)
        fluid = WeaklyCompressibleSPHSystem(fluid_particles;
                                            smoothing_kernel, smoothing_length,
                                            state_equation,
                                            density_calculator=SummationDensity(),
                                            correction=ShepardKernelCorrection())

        structure_particles = RectangularShape(particle_spacing, (3, 2), (0.0, -0.15);
                                               density=1200.0)
        hydrodynamic_density = fill(density,
                                    TrixiParticles.nparticles(structure_particles))
        hydrodynamic_mass = fill(density * particle_spacing^2,
                                 TrixiParticles.nparticles(structure_particles))
        boundary_model = BoundaryModelDummyParticles(hydrodynamic_density,
                                                     hydrodynamic_mass,
                                                     SummationDensity(), smoothing_kernel,
                                                     smoothing_length;
                                                     state_equation,
                                                     correction=ShepardKernelCorrection())
        structure = TotalLagrangianSPHSystem(structure_particles;
                                             smoothing_kernel, smoothing_length,
                                             young_modulus=1.0e6, poisson_ratio=0.3,
                                             boundary_model)
        systems = reverse_order ? (structure, fluid) : (fluid, structure)
        semi = Semidiscretization(systems...; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode,
                             (; semi=ode.p.semi, split_integration_data=nothing), 0.0)

        fluid = only(system
                     for system in ode.p.semi.systems
                     if system isa WeaklyCompressibleSPHSystem)
        structure = only(system
                         for system in ode.p.semi.systems
                         if system isa TotalLagrangianSPHSystem)
        v_fluid = TrixiParticles.wrap_v(v_ode, fluid, ode.p.semi)
        v_structure = TrixiParticles.wrap_v(v_ode, structure, ode.p.semi)
        dv_fluid = TrixiParticles.wrap_v(dv_ode, fluid, ode.p.semi)
        dv_structure = TrixiParticles.wrap_v(dv_ode, structure, ode.p.semi)

        return (;
                fluid_density=copy(TrixiParticles.current_density(v_fluid, fluid)),
                structure_density=copy(TrixiParticles.current_density(v_structure,
                                                                      structure)),
                fluid_coefficient=copy(fluid.cache.kernel_correction_coefficient),
                structure_coefficient=copy(structure.boundary_model.cache.kernel_correction_coefficient),
                fluid_rhs=copy(dv_fluid), structure_rhs=copy(dv_structure))
    end

    forward = ordered_fluid_structure_result(false)
    reverse = ordered_fluid_structure_result(true)
    @test forward.fluid_density≈reverse.fluid_density rtol=5e-13 atol=5e-13
    @test forward.structure_density≈reverse.structure_density rtol=5e-13 atol=5e-13
    @test forward.fluid_coefficient≈reverse.fluid_coefficient rtol=5e-13 atol=5e-13
    @test forward.structure_coefficient≈reverse.structure_coefficient rtol=5e-13 atol=5e-13
    @test forward.fluid_rhs≈reverse.fluid_rhs rtol=1e-11 atol=1e-10
    @test forward.structure_rhs≈reverse.structure_rhs rtol=1e-11 atol=1e-10

    # TLSPH has a material gradient correction and a separate hydrodynamic boundary
    # correction. Verify that the ordinary structural gradient remains untouched while the
    # FSI-specific path consumes the nonidentity boundary correction matrix.
    gradient = structure_setup(GradientCorrection())
    particle = first(TrixiParticles.eachparticle(gradient.system))
    pos_diff = SVector(0.05, 0.025)
    distance = norm(pos_diff)
    raw_gradient = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, distance,
                                              smoothing_length)
    correction_matrix = TrixiParticles.correction_matrix(gradient.system, particle)
    hydrodynamic_gradient = TrixiParticles.hydrodynamic_smoothing_kernel_grad(gradient.system,
                                                                              pos_diff,
                                                                              distance,
                                                                              particle)

    @test norm(correction_matrix - I) > 1e-2
    @test TrixiParticles.smoothing_kernel_grad(gradient.system, pos_diff, distance,
                                               particle) ≈ raw_gradient
    @test hydrodynamic_gradient ≈ correction_matrix * raw_gradient

    # Rigid-body correction caches are not implemented. Reject both density and gradient
    # corrections at construction instead of allowing a later crash or stale normalization.
    error_message = "corrections in `BoundaryModelDummyParticles` are not supported " *
                    "for `RigidBodySystem`"
    for correction in (ShepardKernelCorrection(), GradientCorrection())
        boundary_model = BoundaryModelDummyParticles(particles.density, particles.mass,
                                                     SummationDensity(), smoothing_kernel,
                                                     smoothing_length;
                                                     state_equation, correction)
        @test_throws ArgumentError(error_message) RigidBodySystem(particles;
                                                                  boundary_model)
    end
end
