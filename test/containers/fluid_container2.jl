@testset verbose=true "Constructors with Setups" begin
    setups = [
        RectangularShape(0.123, (2, 3), (-1.0, 0.1), 1.0),
        RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1), 1.0),
        RectangularTank(0.123, (0.369, 0.246), (0.369, 0.369), 1020.0),
        RectangularTank(0.123, (0.369, 0.246, 0.246), (0.369, 0.492, 0.492), 1020.0),
        CircularShape(0.52, 0.1, (-0.2, 0.123), 1.0),
    ]
    setup_names = [
        "RectangularShape 2D",
        "RectangularShape 3D",
        "RectangularTank 2D",
        "RectangularTank 3D",
        "CircularShape",
    ]
    NDIMS_ = [2, 3, 2, 3, 2]
    density_calculators = [
        SummationDensity(),
        ContinuityDensity(),
    ]
    @testset "$(setup_names[i])" for i in eachindex(setups)
        setup = setups[i]
        NDIMS = NDIMS_[i]
        state_equation = Val(:state_equation)
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
        smoothing_length = 0.362

        @testset "$(typeof(density_calculator))" for density_calculator in density_calculators
            system = WeaklyCompressibleSPHSystem(setup, density_calculator,
                                                 state_equation, smoothing_kernel,
                                                 smoothing_length)

            @test system isa WeaklyCompressibleSPHSystem{NDIMS}
            @test system.initial_coordinates == setup.coordinates
            @test system.initial_velocity == setup.velocities
            @test system.mass == setup.masses
            @test system.density_calculator == density_calculator
            @test system.state_equation == state_equation
            @test system.smoothing_kernel == smoothing_kernel
            @test system.smoothing_length == smoothing_length
            @test system.viscosity isa TrixiParticles.NoViscosity
            @test system.acceleration == [0.0 for _ in 1:NDIMS]
            @test length(system.mass) == size(setup.coordinates, 2)

            if density_calculator isa SummationDensity
                @test length(system.cache.density) == size(setup.coordinates, 2)
            end

            if density_calculator isa ContinuityDensity
                @test length(system.cache.initial_density) == size(setup.coordinates, 2)
                @test system.cache.initial_density == setup.densities
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
            @test_throws ArgumentError(error_str) WeaklyCompressibleSPHSystem(setup,
                                                                              density_calculator,
                                                                              state_equation,
                                                                              smoothing_kernel,
                                                                              smoothing_length,
                                                                              acceleration=(0.0))
        end
    end
end
