using OrdinaryDiffEq

# setup a single particle and calculate its density
@testset verbose=true "DensityCalculators" begin
    @testset verbose=true "SummationDensity" begin
        water_density = 1000.0

        # use reshape to create a matrix
        init = InitialCondition(reshape([0.0, 0.0], 2, 1), reshape([0.0, 0.0], 2, 1),
                                [water_density], [water_density])

        smoothing_length = 1.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()

        state_equation = StateEquationCole(10, 7, water_density, 100000.0,
                                           background_pressure=100000.0,
                                           clip_negative_pressure=false)
        viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

        fluid_system = WeaklyCompressibleSPHSystem(init, SummationDensity(), state_equation,
                                                   smoothing_kernel, smoothing_length,
                                                   viscosity=viscosity,
                                                   acceleration=(0.0, 0.0))

        (; cache) = fluid_system
        (; density) = cache # Density is in the cache for SummationDensity

        semi = Semidiscretization(fluid_system, neighborhood_search=GridNeighborhoodSearch,
                                  damping_coefficient=1e-5)

        tspan = (0.0, 0.01)
        ode = semidiscretize(semi, tspan)

        sol = solve(ode, RDPK3SpFSAL49(),
                    abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
                    reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
                    dtmax=1e-2, # Limit stepsize to prevent crashing
                    save_everystep=false)

        @test density[1] ===
              water_density * TrixiParticles.kernel(smoothing_kernel, 0.0, smoothing_length)
    end
end
