# 2D
@testset verbose=true "Rectangular Shape 2D" begin
    @testset verbose=true "No Hydrostatic Pressure" begin
        @testset "Positions" begin
            # Rectangular shape
            particle_spacing = 0.01
            n_particles_per_dimension = (4, 3)

            positions = [(0.0, 0.0), (3.0, 4.0), (-3.0, -4.0), (-3.0, 4.0), (3.0, -4.0)]

            expected_coords = [
                [0.005 0.005 0.005 0.015 0.015 0.015 0.025 0.025 0.025 0.035 0.035 0.035;
                 0.005 0.015 0.025 0.005 0.015 0.025 0.005 0.015 0.025 0.005 0.015 0.025],
                [3.005 3.005 3.005 3.015 3.015 3.015 3.025 3.025 3.025 3.035 3.035 3.035;
                 4.005 4.015 4.025 4.005 4.015 4.025 4.005 4.015 4.025 4.005 4.015 4.025],
                [-2.995 -2.995 -2.995 -2.985 -2.985 -2.985 -2.975 -2.975 -2.975 -2.965 -2.965 -2.965;
                 -3.995 -3.985 -3.975 -3.995 -3.985 -3.975 -3.995 -3.985 -3.975 -3.995 -3.985 -3.975],
                [-2.995 -2.995 -2.995 -2.985 -2.985 -2.985 -2.975 -2.975 -2.975 -2.965 -2.965 -2.965;
                 4.005 4.015 4.025 4.005 4.015 4.025 4.005 4.015 4.025 4.005 4.015 4.025],
                [3.005 3.005 3.005 3.015 3.015 3.015 3.025 3.025 3.025 3.035 3.035 3.035;
                 -3.995 -3.985 -3.975 -3.995 -3.985 -3.975 -3.995 -3.985 -3.975 -3.995 -3.985 -3.975]
            ]

            @testset "Position $i" for i in eachindex(positions)
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, positions[i],
                                         density=1.0, loop_order=:x_first)

                @test shape.coordinates == expected_coords[i]
            end
        end

        @testset verbose=true "Loop Orders" begin
            loop_orders = [:y_first, :x_first]

            expected_coords = [
                [0.0 1.0 0.0 1.0;
                 0.0 0.0 1.0 1.0],
                [0.0 0.0 1.0 1.0;
                 0.0 1.0 0.0 1.0]
            ]

            @testset "$(loop_orders[i])" for i in eachindex(loop_orders)
                shape = RectangularShape(1.0, (2, 2), (0.0, 0.0), density=1.0,
                                         place_on_shell=true,
                                         loop_order=loop_orders[i])

                @test shape.coordinates == expected_coords[i]
            end
        end
    end

    # Only show all of these nested testsets in case of errors
    @testset verbose=false "Incompressible Hydrostatic Pressure" begin
        particle_spacing = 0.1

        pressure = [0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05;
                    0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05]

        @testset "Vertical Gravity" begin
            n_particles_per_dimension = (2, 10)

            @testset "Loop Order :y_first" begin
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, (0.0, 0.0),
                                         density=1000.0, acceleration=(0.0, -9.81))

                @test shape.pressure ≈ 9.81 * 1000.0 * vec(pressure)
                @test shape.density == 1000 * ones(20)
                @test shape.mass == 1000 * 0.1^2 * ones(20)
            end

            @testset "Loop Order :x_first" begin
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, (0.0, 0.0),
                                         density=1000.0, acceleration=(0.0, -9.81),
                                         loop_order=:x_first)

                # Transpose `pressure`
                @test shape.pressure ≈ 9.81 * 1000.0 * vec(pressure')
            end

            @testset "Positive Acceleration" begin
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, (0.0, 0.0),
                                         density=1000.0, acceleration=(0.0, 9.81))

                # Same as before, but revsere pressure
                @test shape.pressure ≈ 9.81 * 1000.0 * vec(reverse(pressure))
            end
        end

        @testset "Horizontal Gravity" begin
            n_particles_per_dimension = (10, 2)

            @testset "Loop Order :y_first" begin
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, (0.0, 0.0),
                                         density=1000.0, acceleration=(-3.73, 0.0))

                @test shape.pressure ≈ 3.73 * 1000.0 * vec(pressure')
                @test shape.density == 1000 * ones(prod(n_particles_per_dimension))
                @test shape.mass ==
                      1000 * particle_spacing^2 * ones(prod(n_particles_per_dimension))
            end

            @testset "Loop Order :x_first" begin
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, (0.0, 0.0),
                                         density=1000.0, acceleration=(-3.73, 0.0),
                                         loop_order=:x_first)

                # Don't transpose `pressure`
                @test shape.pressure ≈ 3.73 * 1000.0 * vec(pressure)
            end

            @testset "Positive Acceleration" begin
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, (0.0, 0.0),
                                         density=1000.0, acceleration=(4.71, 0.0))

                # Same as before, but revsere pressure
                @test shape.pressure ≈ 4.71 * 1000.0 * vec(reverse(pressure'))
            end
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Compressible Hydrostatic Pressure" begin
        particle_spacing = 0.1

        state_equation = Val(:state_equation)
        function TrixiParticles.inverse_state_equation(::Val{:state_equation}, pressure)
            return 1000.0 + 10pressure
        end

        pressure = [2300.0 1100.0 500.0 200.0 50.0;
                    2300.0 1100.0 500.0 200.0 50.0]

        # Only show all of these nested testsets in case of errors
        @testset verbose=false "Vertical Gravity" begin
            n_particles_per_dimension = (2, 5)

            @testset "Loop Order :y_first" begin
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, (0.0, 0.0),
                                         acceleration=(0.0, -1.0),
                                         state_equation=state_equation)

                @test shape.pressure ≈ vec(pressure)
                @test shape.density ==
                      TrixiParticles.inverse_state_equation.(Ref(state_equation),
                                                             shape.pressure)
                @test shape.mass == particle_spacing^2 * shape.density
            end

            @testset "Loop Order :x_first" begin
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, (0.0, 0.0),
                                         acceleration=(0.0, -1.0),
                                         state_equation=state_equation,
                                         loop_order=:x_first)

                # Transpose `pressure`
                @test shape.pressure ≈ vec(pressure')
            end

            @testset "Positive Acceleration" begin
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, (0.0, 0.0),
                                         acceleration=(0.0, 1.0),
                                         state_equation=state_equation)

                @test shape.pressure ≈ vec(reverse(pressure))
                @test shape.density ==
                      TrixiParticles.inverse_state_equation.(Ref(state_equation),
                                                             shape.pressure)
                @test shape.mass == particle_spacing^2 * shape.density
            end
        end

        @testset "Horizontal Gravity" begin
            n_particles_per_dimension = (5, 2)
            shape = RectangularShape(particle_spacing,
                                     n_particles_per_dimension, (0.0, 0.0),
                                     acceleration=(-1.0, 0.0),
                                     state_equation=state_equation)

            @test shape.pressure ≈ 1.0 * vec(pressure')
            @test shape.density ==
                  TrixiParticles.inverse_state_equation.(Ref(state_equation),
                                                         shape.pressure)
            @test shape.mass == particle_spacing^2 * shape.density
        end
    end
end

# 3D
@testset verbose=true "Rectangular Shape 3D" begin
    @testset verbose=true "No Hydrostatic Pressure" begin
        @testset "Positions" begin
            # Rectangular shape
            particle_spacing = 0.01
            n_particles_per_dimension = (4, 3, 5)

            positions = [(0.0, 0.0, 0.0), (3.0, 4.0, 2.0), (-3.0, -4.0, -2.0)]

            expected_coords = [
                [0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.015 0.015 0.015 0.015 0.015 0.015 0.015 0.015 0.015 0.015 0.015 0.015 0.015 0.015 0.015 0.025 0.025 0.025 0.025 0.025 0.025 0.025 0.025 0.025 0.025 0.025 0.025 0.025 0.025 0.025 0.035 0.035 0.035 0.035 0.035 0.035 0.035 0.035 0.035 0.035 0.035 0.035 0.035 0.035 0.035;
                 0.005 0.005 0.005 0.005 0.005 0.015 0.015 0.015 0.015 0.015 0.025 0.025 0.025 0.025 0.025 0.005 0.005 0.005 0.005 0.005 0.015 0.015 0.015 0.015 0.015 0.025 0.025 0.025 0.025 0.025 0.005 0.005 0.005 0.005 0.005 0.015 0.015 0.015 0.015 0.015 0.025 0.025 0.025 0.025 0.025 0.005 0.005 0.005 0.005 0.005 0.015 0.015 0.015 0.015 0.015 0.025 0.025 0.025 0.025 0.025;
                 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045 0.005 0.015 0.025 0.035 0.045],
                [3.005 3.005 3.005 3.005 3.005 3.005 3.005 3.005 3.005 3.005 3.005 3.005 3.005 3.005 3.005 3.015 3.015 3.015 3.015 3.015 3.015 3.015 3.015 3.015 3.015 3.015 3.015 3.015 3.015 3.015 3.025 3.025 3.025 3.025 3.025 3.025 3.025 3.025 3.025 3.025 3.025 3.025 3.025 3.025 3.025 3.035 3.035 3.035 3.035 3.035 3.035 3.035 3.035 3.035 3.035 3.035 3.035 3.035 3.035 3.035;
                 4.005 4.005 4.005 4.005 4.005 4.015 4.015 4.015 4.015 4.015 4.025 4.025 4.025 4.025 4.025 4.005 4.005 4.005 4.005 4.005 4.015 4.015 4.015 4.015 4.015 4.025 4.025 4.025 4.025 4.025 4.005 4.005 4.005 4.005 4.005 4.015 4.015 4.015 4.015 4.015 4.025 4.025 4.025 4.025 4.025 4.005 4.005 4.005 4.005 4.005 4.015 4.015 4.015 4.015 4.015 4.025 4.025 4.025 4.025 4.025;
                 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045 2.005 2.015 2.025 2.035 2.045],
                [-2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.995 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.985 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.975 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965 -2.965;
                 -3.995 -3.995 -3.995 -3.995 -3.995 -3.985 -3.985 -3.985 -3.985 -3.985 -3.975 -3.975 -3.975 -3.975 -3.975 -3.995 -3.995 -3.995 -3.995 -3.995 -3.985 -3.985 -3.985 -3.985 -3.985 -3.975 -3.975 -3.975 -3.975 -3.975 -3.995 -3.995 -3.995 -3.995 -3.995 -3.985 -3.985 -3.985 -3.985 -3.985 -3.975 -3.975 -3.975 -3.975 -3.975 -3.995 -3.995 -3.995 -3.995 -3.995 -3.985 -3.985 -3.985 -3.985 -3.985 -3.975 -3.975 -3.975 -3.975 -3.975;
                 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955 -1.995 -1.985 -1.975 -1.965 -1.955]
            ]

            @testset "Position $i" for i in eachindex(positions)
                shape = RectangularShape(particle_spacing,
                                         n_particles_per_dimension, positions[i],
                                         density=1.0, loop_order=:x_first)

                @test shape.coordinates == expected_coords[i]
            end
        end

        @testset verbose=true "Loop Orders" begin
            loop_orders = [:z_first, :y_first, :x_first]

            expected_coords = [
                [0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0;
                 0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0;
                 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0],
                [0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0;
                 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0;
                 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0],
                [0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0;
                 0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0;
                 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0]
            ]

            @testset "$(loop_orders[i])" for i in eachindex(loop_orders)
                shape = RectangularShape(1.0, (2, 2, 2), (0.0, 0.0, 0.0), density=1.0,
                                         place_on_shell=true, loop_order=loop_orders[i])

                @test shape.coordinates == expected_coords[i]
            end
        end
    end

    @testset "Incompressible Hydrostatic Pressure" begin
        particle_spacing = 0.2

        # Pressure distribution with gravity `(0.0, -1.0, 0.0)` and density `1.0`
        pressure = [0.9; 0.9;;
                    0.7; 0.7;;
                    0.5; 0.5;;
                    0.3; 0.3;;
                    0.1; 0.1;;;
                    0.9; 0.9;;
                    0.7; 0.7;;
                    0.5; 0.5;;
                    0.3; 0.3;;
                    0.1; 0.1;;;
                    0.9; 0.9;;
                    0.7; 0.7;;
                    0.5; 0.5;;
                    0.3; 0.3;;
                    0.1; 0.1]

        n_particles_per_dimension = (2, 5, 3)
        acceleration = (0.0, -9.81, 0.0)

        loop_orders = [:z_first, :y_first, :x_first]
        permutations = [
            [1, 2, 3], # Gravity in negative y-direction
            [2, 1, 3], # Gravity in negative x-direction
            [1, 3, 2] # Gravity in negative z-direction
        ]

        @testset "Permutation $permutation" for permutation in permutations
            @testset "Loop Order $loop_order" for loop_order in loop_orders
                n_particles_per_dimension_ = collect(n_particles_per_dimension)
                permute!(n_particles_per_dimension_, permutation)

                acceleration_ = collect(acceleration)
                permute!(acceleration_, permutation)

                shape = RectangularShape(particle_spacing,
                                         Tuple(n_particles_per_dimension_),
                                         (0.0, 0.0, 0.0), density=1000.0,
                                         loop_order=loop_order, acceleration=acceleration_)

                # Permute pressure with acceleration permutation
                permuted1 = permutedims(pressure, permutation)

                # Permute pressure with loop order permutation
                loop_order_permutation = TrixiParticles.loop_permutation(loop_order, Val(3))
                permuted2 = permutedims(permuted1, loop_order_permutation)

                @test shape.pressure ≈ 9.81 * 1000.0 * vec(permuted2)
                @test shape.density == 1000 * ones(prod(n_particles_per_dimension))
                @test shape.mass ==
                      1000 * particle_spacing^3 * ones(prod(n_particles_per_dimension))
            end
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Compressible Hydrostatic Pressure" begin
        particle_spacing = 0.1

        state_equation = Val(:state_equation)
        function TrixiParticles.inverse_state_equation(::Val{:state_equation}, pressure)
            return 1000.0 + 10pressure
        end

        # Pressure distribution with gravity `(0.0, -1.0, 0.0)`
        pressure = [2300.0; 2300.0;;
                    1100.0; 1100.0;;
                    500.0; 500.0;;
                    200.0; 200.0;;
                    50.0; 50.0;;;
                    2300.0; 2300.0;;
                    1100.0; 1100.0;;
                    500.0; 500.0;;
                    200.0; 200.0;;
                    50.0; 50.0;;;
                    2300.0; 2300.0;;
                    1100.0; 1100.0;;
                    500.0; 500.0;;
                    200.0; 200.0;;
                    50.0; 50.0]

        n_particles_per_dimension = (2, 5, 3)
        acceleration = (0.0, -1.0, 0.0)

        permutations = [
            [1, 2, 3], # Gravity in negative y-direction
            [2, 1, 3], # Gravity in negative x-direction
            [1, 3, 2] # Gravity in negative z-direction
        ]
        @testset "Permutation $permutation" for permutation in permutations
            n_particles_per_dimension_ = collect(n_particles_per_dimension)
            permute!(n_particles_per_dimension_, permutation)

            acceleration_ = collect(acceleration)
            permute!(acceleration_, permutation)

            shape = RectangularShape(particle_spacing,
                                     Tuple(n_particles_per_dimension_), (0.0, 0.0, 0.0),
                                     acceleration=acceleration_,
                                     state_equation=state_equation)

            @test shape.pressure ≈ vec(permutedims(pressure, permutation))
            density = TrixiParticles.inverse_state_equation.(Ref(state_equation),
                                                             shape.pressure)
            @test shape.density == density
            @test shape.mass == particle_spacing^3 * shape.density
        end
    end
end
