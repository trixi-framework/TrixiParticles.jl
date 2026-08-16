@testset "Cross-system update ordering" begin
    function ordered_correction_result(reverse_order; edac)
        spacing = 0.1
        smoothing_length = 2spacing
        smoothing_kernel = WendlandC6Kernel{2}()
        density = 1000.0
        velocity(pos) = SVector(0.1 + pos[1], -0.2 - pos[2])
        pressure(pos) = 1.0 + 2pos[1] - pos[2]

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

        systems = reverse_order ? (shepard_system, gradient_system) :
                  (gradient_system, shepard_system)
        semi = Semidiscretization(systems...; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode,
                             (; semi=ode.p.semi, split_integration_data=nothing), 0.0)

        gradient_system = only(system
                               for system in ode.p.semi.systems
                               if system.correction isa GradientCorrection)
        shepard_system = only(system
                              for system in ode.p.semi.systems
                              if system.correction isa ShepardKernelCorrection)
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
end

@testset "Boundary density before pressure" begin
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

    @test boundary.boundary_model.pressure ≈
          state_equation.(boundary.boundary_model.cache.density)
end
