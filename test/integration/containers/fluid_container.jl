@testset verbose=true "Constructors with Setups" begin
    setups = [
        RectangularShape(0.123, (2, 3), (-1.0, 0.1), density=1000.0),
        RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1), density=1000.0),
        RectangularTank(0.123, (0.369, 0.246), (0.369, 0.369), 1020.0),
        RectangularTank(0.123, (0.369, 0.246, 0.246), (0.369, 0.492, 0.492), 1020.0),
        CircularShape(0.52, 0.1, (-0.2, 0.123), density=1000.0),
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
            @test container.viscosity isa TrixiParticles.NoViscosity
            @test container.acceleration == [0.0 for _ in 1:NDIMS]

            if density_calculator isa SummationDensity
                @test length(container.cache.density) == size(setup.coordinates, 2)
                @test length(container.mass) == size(setup.coordinates, 2)
            end

            if density_calculator isa ContinuityDensity
                @test length(container.cache.initial_density) == size(setup.coordinates, 2)
                @test container.cache.initial_density == setup.densities
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
                                                                         acceleration=(0.0))
        end
    end

    # no density
    setups = [
        RectangularShape(0.123, (2, 3), (-1.0, 0.1)),
        RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1)),
    ]
    setup_names = [
        "RectangularShape 2D with no density",
        "RectangularShape 3D with no density",
    ]
    NDIMS_ = [2, 3]
    density_calculator = SummationDensity()

    @testset "$(setup_names[i])" for i in eachindex(setups)
        setup = setups[i]
        NDIMS = NDIMS_[i]
        state_equation = Val(:state_equation)
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
        smoothing_length = 0.362

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
        @test container.viscosity isa TrixiParticles.NoViscosity
        @test container.acceleration == [0.0 for _ in 1:NDIMS]

        if density_calculator isa SummationDensity
            @test length(container.cache.density) == size(setup.coordinates, 2)
            @test length(container.mass) == size(setup.coordinates, 2)
        end
    end

    struct MockShape{NDIMS, ELTYPE <: Real}
        coordinates :: Array{ELTYPE, 2}
        velocities  :: Array{ELTYPE, 2}
        masses      :: Vector{ELTYPE}
        densities   :: Vector{ELTYPE}

        function MockShape(n_particles, NDIMS; density=0.0)
            ELTYPE = Float64

            coordinates = Array{Float64, 2}(undef, NDIMS, n_particles)
            velocities = ones(ELTYPE, size(coordinates))

            densities = ones(ELTYPE, n_particles * (density > 0))
            masses = ones(ELTYPE, 0)

            return new{NDIMS, ELTYPE}(coordinates, velocities, masses, densities)
        end
    end

    setups = [
        MockShape(10, 2),
        MockShape(10, 3),
        MockShape(10, 3, density=1.0),
    ]
    setup_names = [
        "2D Shape no mass and no density",
        "3D Shape no mass and no density",
        "3D Shape no mass",
    ]
    NDIMS_ = [2, 3, 3]

    @testset "$(setup_names[i])" for i in eachindex(setups)
        setup = setups[i]
        NDIMS = NDIMS_[i]
        state_equation = Val(:state_equation)
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
        smoothing_length = 0.362
        @testset "$(typeof(density_calculator))" for density_calculator in density_calculators
            if density_calculator isa ContinuityDensity && length(setup.densities) == 0
                empty_density_err_str = "An initial density needs to be provided when using `ContinuityDensity`!"
                @test_throws ArgumentError(empty_density_err_str) FluidParticleContainer(setup,
                                                                                         density_calculator,
                                                                                         state_equation,
                                                                                         smoothing_kernel,
                                                                                         smoothing_length)
            elseif density_calculator isa SummationDensity && length(setup.masses) == 0
                empty_density_err_str = "An initial mass needs to be provided when using `$(typeof(density_calculator))`!"
                @test_throws ArgumentError(empty_density_err_str) FluidParticleContainer(setup,
                                                                                         density_calculator,
                                                                                         state_equation,
                                                                                         smoothing_kernel,
                                                                                         smoothing_length)
            end
        end
    end
end
