@testset verbose=true "FluidParticleContainer" begin
    @testset verbose=true "Constructors" begin
        coordinates_ = [
            [1.0 2.0
             1.0 2.0],
            [1.0 2.0
             1.0 2.0
             1.0 2.0],
        ]
        @testset "$(i+1)D" for i in 1:2
            NDIMS = i + 1
            coordinates = coordinates_[i]
            velocities = zero(coordinates)
            masses = [1.25, 1.5]
            densities = [990.0, 1000.0]
            state_equation = Val(:state_equation)
            smoothing_kernel = Val(:smoothing_kernel)
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = i + 1
            smoothing_kernel2 = Val(:smoothing_kernel2)
            # The wrong dimension. 2 -> 3, 3 -> 2.
            TrixiParticles.ndims(::Val{:smoothing_kernel2}) = i % 2 + 2
            smoothing_length = 0.362

            # SummationDensity
            density_calculator = SummationDensity()
            container = FluidParticleContainer(coordinates, velocities, masses,
                                               density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length)

            @test container isa FluidParticleContainer{NDIMS}
            @test container.initial_coordinates == coordinates
            @test container.initial_velocity == velocities
            @test container.mass == masses
            @test container.density_calculator == density_calculator
            @test container.state_equation == state_equation
            @test container.smoothing_kernel == smoothing_kernel
            @test container.smoothing_length == smoothing_length
            @test container.viscosity isa TrixiParticles.NoViscosity
            @test container.acceleration == [0.0 for _ in 1:NDIMS]

            error_str1 = "Acceleration must be of length $NDIMS for a $(NDIMS)D problem!"
            @test_throws ArgumentError(error_str1) FluidParticleContainer(coordinates,
                                                                          velocities,
                                                                          masses,
                                                                          density_calculator,
                                                                          state_equation,
                                                                          smoothing_kernel,
                                                                          smoothing_length,
                                                                          acceleration=(0.0))

            error_str2 = "Smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem!"
            @test_throws ArgumentError(error_str2) FluidParticleContainer(coordinates,
                                                                          velocities,
                                                                          masses,
                                                                          density_calculator,
                                                                          state_equation,
                                                                          smoothing_kernel2,
                                                                          smoothing_length)

            # ContinuityDensity
            density_calculator = ContinuityDensity()
            container = FluidParticleContainer(coordinates, velocities, masses, densities,
                                               density_calculator, state_equation,
                                               smoothing_kernel,
                                               smoothing_length)

            @test container isa FluidParticleContainer{NDIMS}
            @test container.initial_coordinates == coordinates
            @test container.initial_velocity == velocities
            @test container.mass == masses
            @test container.density_calculator == density_calculator
            @test container.state_equation == state_equation
            @test container.smoothing_kernel == smoothing_kernel
            @test container.smoothing_length == smoothing_length
            @test container.viscosity isa TrixiParticles.NoViscosity
            @test container.acceleration == [0.0 for _ in 1:NDIMS]

            @test_throws ArgumentError(error_str1) FluidParticleContainer(coordinates,
                                                                          velocities,
                                                                          masses,
                                                                          densities,
                                                                          density_calculator,
                                                                          state_equation,
                                                                          smoothing_kernel,
                                                                          smoothing_length,
                                                                          acceleration=(0.0))

            @test_throws ArgumentError(error_str2) FluidParticleContainer(coordinates,
                                                                          velocities,
                                                                          masses,
                                                                          densities,
                                                                          density_calculator,
                                                                          state_equation,
                                                                          smoothing_kernel2,
                                                                          smoothing_length)
        end
    end

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
                @test length(container.mass) == size(setup.coordinates, 2)

                if density_calculator isa SummationDensity
                    @test length(container.cache.density) == size(setup.coordinates, 2)
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
    end

    @testset verbose=true "show" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        velocities = zero(coordinates)
        masses = [1.25, 1.5]
        densities = [990.0, 1000.0]
        state_equation = Val(:state_equation)
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
        smoothing_length = 0.362
        density_calculator = SummationDensity()

        container = FluidParticleContainer(coordinates, velocities, masses,
                                           density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length)

        show_compact = "FluidParticleContainer{2}(SummationDensity(), " *
                       "Val{:state_equation}(), Val{:smoothing_kernel}(), " *
                       "TrixiParticles.NoViscosity(), [0.0, 0.0]) with 2 particles"
        @test repr(container) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ FluidParticleContainer{2}                                                                        │
        │ ═════════════════════════                                                                        │
        │ #particles: ………………………………………………… 2                                                                │
        │ density calculator: …………………………… SummationDensity                                                 │
        │ state equation: ……………………………………… Val                                                              │
        │ smoothing kernel: ………………………………… Val                                                              │
        │ viscosity: …………………………………………………… TrixiParticles.NoViscosity()                                     │
        │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", container) == show_box
    end

    @testset verbose=true "write_u0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        velocities = zero(coordinates)
        masses = [1.25, 1.5]
        densities = [990.0, 1000.0]
        state_equation = Val(:state_equation)
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362
        density_calculator = SummationDensity()

        container = FluidParticleContainer(coordinates, velocities, masses,
                                           density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length)

        u0 = zeros(TrixiParticles.u_nvariables(container),
                   TrixiParticles.n_moving_particles(container))
        TrixiParticles.write_u0!(u0, container)

        @test u0 == coordinates
    end

    @testset verbose=true "write_v0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        velocities = 2 * coordinates
        masses = [1.25, 1.5]
        densities = [990.0, 1000.0]
        state_equation = Val(:state_equation)
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362

        # SummationDensity
        container = FluidParticleContainer(coordinates, velocities, masses,
                                           SummationDensity(),
                                           state_equation, smoothing_kernel,
                                           smoothing_length)

        v0 = zeros(TrixiParticles.v_nvariables(container),
                   TrixiParticles.n_moving_particles(container))
        TrixiParticles.write_v0!(v0, container)

        @test v0 == velocities

        # ContinuityDensity
        container = FluidParticleContainer(coordinates, velocities, masses, densities,
                                           ContinuityDensity(),
                                           state_equation, smoothing_kernel,
                                           smoothing_length)

        v0 = zeros(TrixiParticles.v_nvariables(container),
                   TrixiParticles.n_moving_particles(container))
        TrixiParticles.write_v0!(v0, container)

        @test v0 == vcat(velocities, densities')
    end
end
