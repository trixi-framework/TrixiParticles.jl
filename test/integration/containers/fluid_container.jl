@testset verbose=true "Constructors with Setups" begin
    setups = [
        RectangularShape(0.123, (2, 3), (-1.0, 0.1), 1.0),
        RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1), 1.0),
        RectangularTank(0.123, (0.369, 0.246), (0.369, 0.369), 1020.0),
        RectangularTank(0.123, (0.369, 0.246, 0.246), (0.369, 0.492, 0.492), 1020.0),
        CircularShape(0.52, 0.1, (-0.2, 0.123), 1.0),
        RectangularShape(0.123, (2, 3), (-1.0, 0.1), 1.0),
        RectangularShape(0.123, (2, 3), (-1.0, 0.1), 1.0),
    ]
    setup_names = [
        "RectangularShape 2D",
        "RectangularShape 3D",
        "RectangularTank 2D",
        "RectangularTank 3D",
        "CircularShape",
        "RectangularShape 2D with KernelCorrection",
        "RectangularShape 2D with AkinciFreeSurfaceCorrection",
    ]
    NDIMS_ = [2, 3, 2, 3, 2, 2, 2]
    density_calculators = [
        SummationDensity(),
        ContinuityDensity(),
    ]
    correction = [
        NoCorrection(),
        NoCorrection(),
        NoCorrection(),
        NoCorrection(),
        NoCorrection(),
        KernelCorrection(),
        AkinciFreeSurfaceCorrection(),
    ]

    @testset "$(setup_names[i])" for i in eachindex(setups)
        setup = setups[i]
        NDIMS = NDIMS_[i]
        corr = correction[i]
        state_equation = Val(:state_equation)
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
        smoothing_length = 0.362

        @testset "$(typeof(density_calculator))" for density_calculator in density_calculators
            container = FluidParticleContainer(setup, density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, 1000.0, correction=corr)

            @test container isa FluidParticleContainer{NDIMS}
            @test container.initial_coordinates == setup.coordinates
            @test container.initial_velocity == setup.velocities
            @test container.mass == setup.masses
            @test container.density_calculator == density_calculator
            @test container.state_equation == state_equation
            @test container.smoothing_kernel == smoothing_kernel
            @test container.smoothing_length == smoothing_length
            @test container.viscosity isa TrixiParticles.NoViscosity
            @test container.acceleration == [0.0 for _ in 1:NDIMS]
            @test length(container.mass) == size(setup.coordinates, 2)

            if density_calculator isa SummationDensity
                @test length(container.cache.density) == size(setup.coordinates, 2)
            end

            if density_calculator isa ContinuityDensity
                @test length(container.cache.initial_density) == size(setup.coordinates, 2)
                @test container.cache.initial_density == setup.densities
            end
            if corr isa KernelCorrection
                @test length(container.cache.cw) == size(setup.coordinates, 2)
            end
        end
    end

    # wrong dimension of acceleration
    NDIMS_ = [2, 3]
    @testset "Wrong acceleration dimension" for i in eachindex(NDIMS_)
        setup = setups[i]
        NDIMS = NDIMS_[i]
        state_equation = Val(:state_equation)
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
        smoothing_length = 0.362

        @testset "$(typeof(density_calculator))" for density_calculator in density_calculators
            error_str = "Acceleration must be of length $NDIMS for a $(NDIMS)D problem!"
            @test_throws ArgumentError(error_str) FluidParticleContainer(setup,
                                                                         density_calculator,
                                                                         state_equation,
                                                                         smoothing_kernel,
                                                                         smoothing_length,
                                                                         1000.0,
                                                                         acceleration=(0.0))
        end
    end
end
