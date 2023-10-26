using OrdinaryDiffEq

# Setup a single particle and calculate its density
@testset verbose=true "DensityCalculators" begin
    @testset verbose=true "SummationDensity" begin
        water_density = 1000.0

        # use reshape to create a matrix
        initial_condition = InitialCondition(zeros(2, 1), zeros(2, 1), [water_density],
                                            [water_density])

        smoothing_length = 1.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()

        state_equation = StateEquationCole(10, 7, water_density, 100000.0,
                                           background_pressure=100000.0)
        viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

        fluid_system = WeaklyCompressibleSPHSystem(initial_condition, SummationDensity(),
                                                   state_equation,
                                                   smoothing_kernel, smoothing_length,
                                                   viscosity=viscosity)

        (; cache) = fluid_system
        (; density) = cache # Density is in the cache for SummationDensity

        semi = Semidiscretization(fluid_system, neighborhood_search=GridNeighborhoodSearch,
                                  damping_coefficient=1e-5)

        tspan = (0.0, 0.01)
        ode = semidiscretize(semi, tspan)
        TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

        @test density[1] ===
              water_density * TrixiParticles.kernel(smoothing_kernel, 0.0, smoothing_length)
    end
end
