@testset verbose=true "WeaklyCompressibleSPHSystem" begin
    @testset verbose=true "Constructors" begin
        coordinates_ = [
            [1.0 2.0
             1.0 2.0],
            [1.0 2.0
             1.0 2.0
             1.0 2.0],
        ]
        density_calculators = [
            SummationDensity(),
            ContinuityDensity(),
        ]

        @testset "$(i+1)D" for i in 1:2
            @testset "$(typeof(density_calculator))" for density_calculator in density_calculators
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

                initial_condition = InitialCondition(coordinates, velocities, masses,
                                                     densities)
                system = WeaklyCompressibleSPHSystem(initial_condition,
                                                     density_calculator,
                                                     state_equation, smoothing_kernel,
                                                     smoothing_length)

                @test system isa WeaklyCompressibleSPHSystem{NDIMS}
                @test system.initial_condition == initial_condition
                @test system.mass == masses
                @test system.density_calculator == density_calculator
                @test system.state_equation == state_equation
                @test system.smoothing_kernel == smoothing_kernel
                @test system.smoothing_length == smoothing_length
                @test system.viscosity isa TrixiParticles.NoViscosity
                @test system.acceleration == [0.0 for _ in 1:NDIMS]

                if density_calculator isa SummationDensity
                    @test length(system.cache.density) == size(coordinates, 2)
                end

                error_str1 = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
                @test_throws ArgumentError(error_str1) WeaklyCompressibleSPHSystem(initial_condition,
                                                                                   density_calculator,
                                                                                   state_equation,
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   acceleration=(0.0))

                error_str2 = "smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"
                @test_throws ArgumentError(error_str2) WeaklyCompressibleSPHSystem(initial_condition,
                                                                                   density_calculator,
                                                                                   state_equation,
                                                                                   smoothing_kernel2,
                                                                                   smoothing_length)
            end
        end
    end

    @testset verbose=true "Constructors with Setups" begin
        setups = [
            RectangularShape(0.123, (2, 3), (-1.0, 0.1), 1.0),
            RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1), 1.0),
            RectangularTank(0.123, (0.369, 0.246), (0.369, 0.369), 1020.0).fluid,
            RectangularTank(0.123, (0.369, 0.246, 0.246), (0.369, 0.492, 0.492),
                            1020.0).fluid,
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
            "RectangularShape 2D with ShepardKernelCorrection",
            "RectangularShape 2D with AkinciFreeSurfaceCorrection",
        ]
        NDIMS_ = [2, 3, 2, 3, 2, 2, 2]
        density_calculators = [
            SummationDensity(),
            ContinuityDensity(),
        ]
        correction = [
            Nothing(),
            Nothing(),
            Nothing(),
            Nothing(),
            Nothing(),
            ShepardKernelCorrection(),
            AkinciFreeSurfaceCorrection(1000.0),
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
                system = WeaklyCompressibleSPHSystem(setup, density_calculator,
                                                     state_equation, smoothing_kernel,
                                                     smoothing_length,
                                                     correction=corr)

                @test system isa WeaklyCompressibleSPHSystem{NDIMS}
                @test system.initial_condition == setup
                @test system.mass == setup.mass
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
                if corr isa ShepardKernelCorrection
                    @test length(system.cache.kernel_correction_coefficient) ==
                          size(setup.coordinates, 2)
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
                error_str = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
                @test_throws ArgumentError(error_str) WeaklyCompressibleSPHSystem(setup,
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

        initial_condition = InitialCondition(coordinates, velocities, masses, densities)
        system = WeaklyCompressibleSPHSystem(initial_condition,
                                             density_calculator,
                                             state_equation, smoothing_kernel,
                                             smoothing_length)

        show_compact = "WeaklyCompressibleSPHSystem{2}(SummationDensity(), nothing, Val{:state_equation}(), Val{:smoothing_kernel}(), TrixiParticles.NoViscosity(), [0.0, 0.0]) with 2 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ WeaklyCompressibleSPHSystem{2}                                                                   │
        │ ══════════════════════════════                                                                   │
        │ #particles: ………………………………………………… 2                                                                │
        │ density calculator: …………………………… SummationDensity                                                 │
        │ correction method: ……………………………… Nothing                                                          │
        │ state equation: ……………………………………… Val                                                              │
        │ smoothing kernel: ………………………………… Val                                                              │
        │ viscosity: …………………………………………………… TrixiParticles.NoViscosity()                                     │
        │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
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

        initial_condition = InitialCondition(coordinates, velocities, masses, densities)
        system = WeaklyCompressibleSPHSystem(initial_condition,
                                             density_calculator,
                                             state_equation, smoothing_kernel,
                                             smoothing_length)

        u0 = zeros(TrixiParticles.u_nvariables(system),
                   TrixiParticles.n_moving_particles(system))
        TrixiParticles.write_u0!(u0, system)

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

        initial_condition = InitialCondition(coordinates, velocities, masses, densities)

        # SummationDensity
        system = WeaklyCompressibleSPHSystem(initial_condition,
                                             SummationDensity(),
                                             state_equation, smoothing_kernel,
                                             smoothing_length)

        v0 = zeros(TrixiParticles.v_nvariables(system),
                   TrixiParticles.n_moving_particles(system))
        TrixiParticles.write_v0!(v0, system)

        @test v0 == velocities

        # ContinuityDensity
        system = WeaklyCompressibleSPHSystem(initial_condition,
                                             ContinuityDensity(),
                                             state_equation, smoothing_kernel,
                                             smoothing_length)

        v0 = zeros(TrixiParticles.v_nvariables(system),
                   TrixiParticles.n_moving_particles(system))
        TrixiParticles.write_v0!(v0, system)

        @test v0 == vcat(velocities, densities')
    end
end
