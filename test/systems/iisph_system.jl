@testset verbose=true "ImplicitIncompressibleSPHSystem" begin
    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Constructors" begin
        coordinates_ = [
            [1.0 2.0
             1.0 2.0],
            [1.0 2.0
             1.0 2.0
             1.0 2.0]
        ]
        omegas = [
            0.5,
            0.6
        ]
        max_errors = [
            0.1,
            0.05
        ]
        min_iterations_ = [
            2,
            10
        ]
        max_iterations_ = [
            20,
            30
        ]
        time_steps_ = [
            0.001,
            0.0001
        ]
        @testset "$(i+1)D" for i in 1:2
            NDIMS = i + 1
            coordinates = coordinates_[i]
            mass = [1.25, 1.5]
            density = [990.0, 1000.0]
            reference_density = 1000.0
            smoothing_kernel = Val(:smoothing_kernel)
            omega = omegas[i]
            max_error = max_errors[i]
            min_iterations = min_iterations_[i]
            max_iterations = max_iterations_[i]
            time_step = time_steps_[i]
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = i + 1
            smoothing_kernel2 = Val(:smoothing_kernel2)
            # The wrong dimension. 2 -> 3, 3 -> 2.
            TrixiParticles.ndims(::Val{:smoothing_kernel2}) = i % 2 + 2
            smoothing_length = 0.362

            initial_condition = InitialCondition(; coordinates, mass, density)
            system = ImplicitIncompressibleSPHSystem(initial_condition,
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     reference_density,
                                                     omega=omega,
                                                     max_error=max_error,
                                                     min_iterations=min_iterations,
                                                     max_iterations=max_iterations,
                                                     time_step=time_step)

            @test system isa ImplicitIncompressibleSPHSystem{NDIMS}
            @test system.initial_condition == initial_condition
            @test system.mass == mass
            @test system.reference_density == reference_density
            @test system.smoothing_kernel == smoothing_kernel
            @test system.omega == omega
            @test TrixiParticles.initial_smoothing_length(system) == smoothing_length
            @test system.viscosity === nothing
            @test system.acceleration == [0.0 for _ in 1:NDIMS]
            @test system.max_error == max_error
            @test system.min_iterations == min_iterations
            @test system.max_iterations == max_iterations
            @test system.time_step == time_step
            @test length(system.density) == size(coordinates, 2)

            error_str1 = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str1) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   density,
                                                                                   acceleration=(0.0),
                                                                                   time_step=0.001)

            error_str2 = "smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str2) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel2,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   time_step=0.001)

            error_str3 = "`reference_density` must be a positive number"
            @test_throws ArgumentError(error_str3) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   0.0,
                                                                                   time_step=0.001)

            error_str4 = "`max_error` is given in percentage, so it must be a number between 0 and 100"
            @test_throws ArgumentError(error_str4) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   max_error=0.0,
                                                                                   time_step=0.001)

            error_str5 = "`min_iterations` must be a positive number"
            @test_throws ArgumentError(error_str5) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   min_iterations=0,
                                                                                   time_step=0.001)

            error_str6 = "`min_iterations` must be smaller or equal to `max_iterations`"
            @test_throws ArgumentError(error_str6) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   min_iterations=10,
                                                                                   max_iterations=5,
                                                                                   time_step=0.001)

            error_str7 = "`time_step must be a positive number"
            @test_throws ArgumentError(error_str6) ImplicitIncompressibleSPHSystem(initial_condition,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   min_iterations=10,
                                                                                   max_iterations=5,
                                                                                   time_step=0)
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Constructors with Setups" begin
        setups = [
            RectangularShape(0.123, (2, 3), (-1.0, 0.1), density=1.0),
            RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1), density=1.0),
            RectangularTank(0.123, (0.369, 0.246), (0.369, 0.369), 1020.0).fluid,
            RectangularTank(0.123, (0.369, 0.246, 0.246), (0.369, 0.492, 0.492),
                            1020.0).fluid,
            SphereShape(0.52, 0.1, (-0.2, 0.123), 1.0)
        ]
        setup_names = [
            "RectangularShape 2D",
            "RectangularShape 3D",
            "RectangularTank 2D",
            "RectangularTank 3D",
            "SphereShape 2D"
        ]
        NDIMS_ = [2, 3, 2, 3, 2]
        reference_densities = [1.0, 1.0, 1020.0, 1020.0, 1.0]
        @testset "$(setup_names[i])" for i in eachindex(setups)
            setup = setups[i]
            NDIMS = NDIMS_[i]
            smoothing_kernel = Val(:smoothing_kernel)
            reference_density = reference_densities[i]
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
            smoothing_length = 0.362

            density_calculator = SummationDensity()

            system = ImplicitIncompressibleSPHSystem(setup,
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     reference_density,
                                                     time_step=0.001)

            @test system isa ImplicitIncompressibleSPHSystem{NDIMS}
            @test system.initial_condition == setup
            @test system.mass == setup.mass
            @test system.smoothing_kernel == smoothing_kernel
            @test system.reference_density == reference_density
            @test TrixiParticles.initial_smoothing_length(system) == smoothing_length
            @test system.viscosity === nothing
            @test system.acceleration == [0.0 for _ in 1:NDIMS]
            @test length(system.mass) == size(setup.coordinates, 2)
            @test length(system.density) == size(setup.coordinates, 2)
        end

        # wrong dimension of acceleration
        NDIMS_ = [2, 3]
        @testset "Wrong acceleration dimension" for i in eachindex(NDIMS_)
            setup = setups[i]
            NDIMS = NDIMS_[i]
            reference_density = 1.0
            smoothing_kernel = Val(:smoothing_kernel)
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
            smoothing_length = 0.362

            density_calculator = SummationDensity()

            error_str = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str) ImplicitIncompressibleSPHSystem(setup,
                                                                                  smoothing_kernel,
                                                                                  smoothing_length,
                                                                                  reference_density,
                                                                                  acceleration=(0.0),
                                                                                  time_step=0.001)
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "show" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        reference_density = 1000.0
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
        smoothing_length = 0.362

        initial_condition = InitialCondition(; coordinates, mass, density)
        system = ImplicitIncompressibleSPHSystem(initial_condition,
                                                 smoothing_kernel,
                                                 smoothing_length,
                                                 reference_density,
                                                 time_step=0.001)

        show_compact = "ImplicitIncompressibleSPHSystem{2}(1000.0, Val{:smoothing_kernel}(), nothing, [0.0, 0.0], 0.5, 0.1, 2, 20) with 2 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ ImplicitIncompressibleSPHSystem{2}                                                               │
        │ ══════════════════════════════════                                                               │
        │ #particles: ………………………………………………… 2                                                                │
        │ reference density: ……………………………… 1000.0                                                           │
        │ density calculator: …………………………… SummationDensity                                                 │
        │ smoothing kernel: ………………………………… Val                                                              │
        │ viscosity: …………………………………………………… nothing                                                          │
        │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
        │ omega: ……………………………………………………………… 0.5                                                              │
        │ max_error: …………………………………………………… 0.1                                                              │
        │ min_iterations: ……………………………………… 2                                                                │
        │ max_iterations: ……………………………………… 20                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end

    @testset verbose=true "write_u0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        reference_density = 1000.0
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362

        initial_condition = InitialCondition(; coordinates, mass, density)
        system = ImplicitIncompressibleSPHSystem(initial_condition,
                                                 smoothing_kernel,
                                                 smoothing_length, reference_density,
                                                 time_step=0.001)

        u0 = zeros(TrixiParticles.u_nvariables(system),
                   TrixiParticles.n_integrated_particles(system))
        TrixiParticles.write_u0!(u0, system)

        @test u0 == coordinates
    end

    @testset verbose=true "write_v0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        velocity = 2 * coordinates
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        reference_density = 1000.0
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)

        # SummationDensity (is always in use)
        system = ImplicitIncompressibleSPHSystem(initial_condition,
                                                 smoothing_kernel,
                                                 smoothing_length,
                                                 reference_density,
                                                 time_step=0.001)

        v0 = zeros(TrixiParticles.v_nvariables(system),
                   TrixiParticles.n_integrated_particles(system))
        TrixiParticles.write_v0!(v0, system)

        @test v0 == velocity
    end
end
