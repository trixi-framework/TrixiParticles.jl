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
                                               smoothing_length, 1000.0)

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
                                                                          acceleration=(0.0),
                                                                          1000.0)

            error_str2 = "Smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem!"
            @test_throws ArgumentError(error_str2) FluidParticleContainer(coordinates,
                                                                          velocities,
                                                                          masses,
                                                                          density_calculator,
                                                                          state_equation,
                                                                          smoothing_kernel2,
                                                                          smoothing_length,
                                                                          1000.0)

            # ContinuityDensity
            density_calculator = ContinuityDensity()
            container = FluidParticleContainer(coordinates, velocities, masses, densities,
                                               density_calculator, state_equation,
                                               smoothing_kernel,
                                               smoothing_length, 1000.0)

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
                                                                          1000.0,
                                                                          acceleration=(0.0))

            @test_throws ArgumentError(error_str2) FluidParticleContainer(coordinates,
                                                                          velocities,
                                                                          masses,
                                                                          densities,
                                                                          density_calculator,
                                                                          state_equation,
                                                                          smoothing_kernel2,
                                                                          smoothing_length,
                                                                          1000.0)
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
                                           smoothing_length, 1000.0)

        show_compact = "FluidParticleContainer{2}(SummationDensity(), NoCorrection(), " *
                       "Val{:state_equation}(), Val{:smoothing_kernel}(), " *
                       "TrixiParticles.NoViscosity(), [0.0, 0.0], 1000.0) with 2 particles"
        @test repr(container) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ FluidParticleContainer{2}                                                                        │
        │ ═════════════════════════                                                                        │
        │ #particles: ………………………………………………… 2                                                                │
        │ density calculator: …………………………… SummationDensity                                                 │
        │ correction method: ……………………………… NoCorrection                                                     │
        │ state equation: ……………………………………… Val                                                              │
        │ smoothing kernel: ………………………………… Val                                                              │
        │ viscosity: …………………………………………………… TrixiParticles.NoViscosity()                                     │
        │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
        │ rho0: ………………………………………………………………… 1000.0                                                           │
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
                                           smoothing_length, 1000)

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
                                           smoothing_length, 1000)

        v0 = zeros(TrixiParticles.v_nvariables(container),
                   TrixiParticles.n_moving_particles(container))
        TrixiParticles.write_v0!(v0, container)

        @test v0 == velocities

        # ContinuityDensity
        container = FluidParticleContainer(coordinates, velocities, masses, densities,
                                           ContinuityDensity(),
                                           state_equation, smoothing_kernel,
                                           smoothing_length, 1000)

        v0 = zeros(TrixiParticles.v_nvariables(container),
                   TrixiParticles.n_moving_particles(container))
        TrixiParticles.write_v0!(v0, container)

        @test v0 == vcat(velocities, densities')
    end
end
