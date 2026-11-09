using OrdinaryDiffEqLowStorageRK

@testset verbose=true "SplitIntegrationCallback" begin
    # The sub-steps of the split integration only contain the `TotalLagrangianSPHSystem`s,
    # but no fluid systems. The no-slip wall velocity in the boundary model of a structure
    # is interpolated from the fluid and must not be reset by the sub-steps.
    @testset verbose=true "No-Slip Wall Velocity: $(nameof(typeof(density_calculator)))" for density_calculator in
                                                                                             (PressureMirroring(),
                                                                                              AdamiPressureExtrapolation())
        particle_spacing = 0.1
        smoothing_length = 1.2 * particle_spacing * sqrt(2)
        smoothing_kernel = WendlandC2Kernel{2}()
        fluid_density = 1000.0
        viscosity = ViscosityAdami(nu=0.05)

        # Fluid moving along a clamped plate. Since the plate is at rest,
        # the wall velocity of the no-slip condition is `-v_fluid` (see Adami et al. 2012).
        fluid = RectangularShape(particle_spacing, (8, 4), (0.0, 0.0),
                                 density=fluid_density, velocity=(1.0, 0.0))
        plate = RectangularShape(particle_spacing, (8, 2), (0.0, -2 * particle_spacing),
                                 density=2700.0)

        fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length, sound_speed=20.0,
                                                   viscosity)

        hydrodynamic_density = fill(fluid_density, nparticles(plate))
        boundary_model = BoundaryModelDummyParticles(hydrodynamic_density,
                                                     hydrodynamic_density *
                                                     particle_spacing^2,
                                                     density_calculator, smoothing_kernel,
                                                     smoothing_length; viscosity)
        structure_system = TotalLagrangianSPHSystem(plate; smoothing_kernel,
                                                    smoothing_length, young_modulus=1e5,
                                                    poisson_ratio=0.3, boundary_model,
                                                    clamped_particles=1:8)

        semi = @test_logs (:info,
                           r"^To create the self-interaction neighborhood search of a `TotalLagrangianSPHSystem`") begin
            Semidiscretization(fluid_system, structure_system;
                               parallelization_backend=SerialBackend())
        end
        ode = semidiscretize(semi, (0.0, 2e-3))

        split_integration = SplitIntegrationCallback(CarpenterKennedy2N54(williamson_condition=false),
                                                     callback=StepsizeCallback(cfl=1.6),
                                                     dt=1.0)
        sol = solve(ode, RDPK3SpFSAL35(), dt=1e-5, abstol=1e-7, reltol=1e-5,
                    save_everystep=false, callback=split_integration)
        @test sol.t[end] == 2e-3

        # The final split integration is done after the last update of the wall velocity
        # in the large integrator, which has to remain available.
        wall_velocity = structure_system.boundary_model.cache.wall_velocity
        @test !iszero(wall_velocity)
    end
end
