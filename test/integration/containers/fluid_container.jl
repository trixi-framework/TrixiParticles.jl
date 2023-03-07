@testset verbose=true "Constructors with Setups" begin
    setups = [
        RectangularShape(0.123, (2, 3), (-1.0, 0.1)),
        RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1)),
        RectangularTank(0.123, 1, 0.246, 0.246, 0.369, 0.369, 1020.0),
        RectangularTank(0.123, 2, 0.246, 0.246, 0.369, 0.492, 0.492, 0.492, 1020.0),
        CircularShape(0.52, 0.1, -0.2, 0.123)
    ]
    setup_names = [
        "RectangularShape 2D",
        "RectangularShape 3D",
        "RectangularTank 2D",
        "RectangularTank 3D",
        "CircularShape"
    ]
    NDIMS_ = [2, 3, 2, 3, 2]
    @testset "$(setup_names[i])" for i in eachindex(setups)
        setup = setups[i]
        NDIMS = NDIMS_[i]
        state_equation = Val(:state_equation)
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362
        density_calculator = SummationDensity()

        container = FluidParticleContainer(setup, density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length)

        @test container isa FluidParticleContainer{NDIMS}
        @test container.initial_coordinates == setup.coordinates
        @test container.initial_velocity == setup.velocities
        @test container.mass == setup.masses
        @test container.density_calculator == density_calculator
        @test container.state_equation == state_equation
        @test container.smoothing_kernel == smoothing_kernel
        @test container.smoothing_length == smoothing_length
        @test container.viscosity isa Pixie.NoViscosity
        @test container.acceleration == [0.0 for _ in 1:NDIMS]

        error_str = "Acceleration must be of length $NDIMS for a $(NDIMS)D problem"
        @test_throws ErrorException(error_str) FluidParticleContainer(setup,
                                                                      density_calculator,
                                                                      state_equation,
                                                                      smoothing_kernel,
                                                                      smoothing_length,
                                                                      acceleration=(0.0))
    end
end
