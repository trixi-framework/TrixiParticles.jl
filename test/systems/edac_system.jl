@testset verbose=true "EntropicallyDampedSPHSystem" begin
    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Constructors" begin
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
            mass = [1.25, 1.5]
            density = [990.0, 1000.0]
            smoothing_kernel = Val(:smoothing_kernel)
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = i + 1
            smoothing_kernel2 = Val(:smoothing_kernel2)
            # The wrong dimension. 2 -> 3, 3 -> 2.
            TrixiParticles.ndims(::Val{:smoothing_kernel2}) = i % 2 + 2
            smoothing_length = 0.362
            sound_speed = 10.0

            initial_condition = InitialCondition(; coordinates, mass, density)

            system = EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                                 smoothing_length, sound_speed)

            @test system isa EntropicallyDampedSPHSystem{NDIMS}
            @test system.initial_condition == initial_condition
            @test system.mass == mass
            @test system.smoothing_kernel == smoothing_kernel
            @test system.smoothing_length == smoothing_length
            @test system.viscosity isa TrixiParticles.NoViscosity
            @test system.nu_edac == (0.5 * smoothing_length * sound_speed) / 8
            @test system.acceleration == [0.0 for _ in 1:NDIMS]

            error_str1 = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str1) EntropicallyDampedSPHSystem(initial_condition,
                                                                               smoothing_kernel,
                                                                               smoothing_length,
                                                                               sound_speed,
                                                                               acceleration=(0.0))

            error_str2 = "smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str2) EntropicallyDampedSPHSystem(initial_condition,
                                                                               smoothing_kernel2,
                                                                               smoothing_length,
                                                                               sound_speed)
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Constructors with Setups" begin
        setups = [
            RectangularShape(0.123, (2, 3), (-1.0, 0.1), density=1.0, pressure=1.0),
            RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1), density=1.0),
            RectangularTank(0.123, (0.369, 0.246), (0.369, 0.369), 1020.0).fluid,
            RectangularTank(0.123, (0.369, 0.246, 0.246), (0.369, 0.492, 0.492),
                            1020.0).fluid,
            SphereShape(0.52, 0.1, (-0.2, 0.123), 1.0),
        ]
        setup_names = [
            "RectangularShape 2D",
            "RectangularShape 3D",
            "RectangularTank 2D",
            "RectangularTank 3D",
            "SphereShape 2D",
        ]
        NDIMS_ = [2, 3, 2, 3, 2]

        @testset "$(setup_names[i])" for i in eachindex(setups)
            setup = setups[i]
            NDIMS = NDIMS_[i]
            smoothing_kernel = Val(:smoothing_kernel)
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
            smoothing_length = 0.362
            sound_speed = 10.0

            system = EntropicallyDampedSPHSystem(setup, smoothing_kernel, smoothing_length,
                                                 sound_speed)

            @test system isa EntropicallyDampedSPHSystem{NDIMS}
            @test system.initial_condition == setup
            @test system.mass == setup.mass
            @test system.smoothing_kernel == smoothing_kernel
            @test system.smoothing_length == smoothing_length
            @test system.viscosity isa TrixiParticles.NoViscosity
            @test system.nu_edac == (0.5 * smoothing_length * sound_speed) / 8
            @test system.acceleration == [0.0 for _ in 1:NDIMS]
            @test length(system.mass) == size(setup.coordinates, 2)
        end

        # wrong dimension of acceleration
        NDIMS_ = [2, 3]
        @testset "Wrong acceleration dimension" for i in eachindex(NDIMS_)
            setup = setups[i]
            NDIMS = NDIMS_[i]
            smoothing_kernel = Val(:smoothing_kernel)
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
            smoothing_length = 0.362
            sound_speed = 10.0

            error_str = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str) EntropicallyDampedSPHSystem(setup,
                                                                              smoothing_kernel,
                                                                              smoothing_length,
                                                                              sound_speed,
                                                                              acceleration=(0.0))
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "show" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
        smoothing_length = 0.362
        sound_speed = 10.0

        initial_condition = InitialCondition(; coordinates, mass, density)
        system = EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                             smoothing_length, sound_speed)

        show_compact = "EntropicallyDampedSPHSystem{2}(NoViscosity(), Val{:smoothing_kernel}(), [0.0, 0.0]) with 2 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ EntropicallyDampedSPHSystem{2}                                                                   │
        │ ══════════════════════════════                                                                   │
        │ #particles: ………………………………………………… 2                                                                │
        │ viscosity: …………………………………………………… NoViscosity                                                      │
        │ ν₍EDAC₎: ………………………………………………………… ≈ 0.226                                                          │
        │ smoothing kernel: ………………………………… Val                                                              │
        │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "write_u0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        pressure = [5.0, 7.8]
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362
        sound_speed = 10.0

        initial_condition = InitialCondition(; coordinates, mass, density, pressure)
        system = EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                             smoothing_length, sound_speed)

        u0 = zeros(TrixiParticles.u_nvariables(system),
                   TrixiParticles.n_moving_particles(system))
        TrixiParticles.write_u0!(u0, system)

        @test u0 == coordinates
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "write_v0!" begin
        coordinates = [0.5 2.0
                       1.0 2.0]
        velocity = 2 * coordinates
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        pressure = [5.0, 7.8]
        pressure_function(x) = 1.0 - 0.2 * ℯ^-((first(x) - 0.5)^2 / 0.001)
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362
        sound_speed = 10.0

        initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                             pressure)

        system = EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                             smoothing_length, sound_speed)

        v0 = zeros(TrixiParticles.v_nvariables(system),
                   TrixiParticles.n_moving_particles(system))
        TrixiParticles.write_v0!(v0, system)

        @test v0 == vcat(velocity, pressure')

        initial_condition = InitialCondition(; coordinates, velocity, mass, density,
                                             pressure=pressure_function)

        system = EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                             smoothing_length, sound_speed)

        v0 = zeros(TrixiParticles.v_nvariables(system),
                   TrixiParticles.n_moving_particles(system))
        TrixiParticles.write_v0!(v0, system)

        @test v0 == vcat(velocity, [0.8, 1.0]')
    end
end
