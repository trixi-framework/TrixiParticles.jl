using OrdinaryDiffEq

# Setup a single particle and calculate its density
@testset verbose = true "DensityCalculators" begin
    @testset verbose = true "SummationDensity" begin
        water_density = 1000.0

        initial_condition = InitialCondition(
            coordinates = zeros(2, 1),
            mass = water_density, density = water_density
        )

        smoothing_length = 1.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()

        state_equation = StateEquationCole(
            sound_speed = 10, reference_density = water_density,
            exponent = 7
        )
        viscosity = ArtificialViscosityMonaghan(alpha = 0.02, beta = 0.0)

        fluid_system = WeaklyCompressibleSPHSystem(
            initial_condition, SummationDensity(),
            state_equation,
            smoothing_kernel, smoothing_length,
            viscosity = viscosity
        )

        (; cache) = fluid_system
        (; density) = cache # Density is in the cache for SummationDensity

        semi = Semidiscretization(fluid_system)

        tspan = (0.0, 0.01)
        ode = semidiscretize(semi, tspan)
        TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)

        @test density[1] ===
            water_density * TrixiParticles.kernel(smoothing_kernel, 0.0, smoothing_length)
    end
end
