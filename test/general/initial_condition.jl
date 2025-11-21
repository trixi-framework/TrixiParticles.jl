@testset verbose=true "InitialCondition" begin
    disjoint_shapes_dict = Dict(
        "Rectangular Shapes" => (RectangularShape(0.1, (3, 4), (-1.0, 1.0),
                                                  density=1.0),
                                 RectangularShape(0.1, (4, 5), (0.0, 1.0), density=1.0,
                                                  velocity=(0.3, -0.5))),
        "Touching Rectangular Shapes" => (RectangularShape(0.1, (3, 4), (-1.0, 1.0),
                                                           density=1.0),
                                          RectangularShape(0.1, (4, 5), (0.0, 1.0),
                                                           density=1.0),
                                          RectangularShape(0.1, (2, 10), (0.0, 0.0),
                                                           density=1.0)),
        "Sphere Shapes" => (SphereShape(0.15, 0.5, (-1.0, 1.0), 1.0),
                            SphereShape(0.15, 0.2, (0.0, 1.0), 1.0),
                            SphereShape(0.15, 1.0, (0.0, -0.2), 1.0,
                                        sphere_type=RoundSphere())),
        "Touching Mixed Shapes" => (RectangularShape(0.1, (3, 10), (-1.0, 0.0),
                                                     density=1.0),
                                    SphereShape(0.1, 0.5, (-1.0, 1.5), 1000.0),
                                    SphereShape(0.1, 0.5, (1.0, 0.5), 1000.0,
                                                sphere_type=RoundSphere())))

    @testset verbose=true "Show" begin
        shape = RectangularShape(0.05, (20, 20), (0.0, 0.0), density=1.0)

        show_compact = "InitialCondition{Float64}()"
        @test repr(shape) == show_compact

        show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ InitialCondition{Float64}                                                                        │
            │ ═════════════════════════                                                                        │
            │ #dimensions: ……………………………………………… 2                                                                │
            │ #particles: ………………………………………………… 400                                                              │
            │ particle spacing: ………………………………… 0.05                                                             │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", shape) == show_box
    end

    @testset verbose=true "Constructors" begin
        @testset "Illegal Inputs" begin
            error_str = "`coordinates` and `velocities` must be of the same size"
            @test_throws ArgumentError(error_str) InitialCondition(coordinates=zeros(2, 3),
                                                                   velocity=zeros(2, 4),
                                                                   mass=ones(3),
                                                                   density=ones(3))

            error_str = "`velocity` must be 2-dimensional for 2-dimensional `coordinates`"
            @test_throws ArgumentError(error_str) InitialCondition(coordinates=zeros(2, 3),
                                                                   velocity=x -> (1, 2, 3),
                                                                   mass=ones(3),
                                                                   density=ones(3))

            error_str = """
                        Expected: length(mass) == size(coordinates, 2)
                        Got: size(coordinates, 2) = 2, length(mass) = 3"""
            @test_throws ArgumentError(error_str) InitialCondition(coordinates=zeros(2, 2),
                                                                   velocity=zeros(2, 2),
                                                                   mass=ones(3),
                                                                   density=ones(2))

            error_str = """
                        Expected: length(density) == size(coordinates, 2)
                        Got: size(coordinates, 2) = 2, length(density) = 3"""
            @test_throws ArgumentError(error_str) InitialCondition(coordinates=zeros(2, 2),
                                                                   velocity=zeros(2, 2),
                                                                   mass=ones(2),
                                                                   density=ones(3))

            error_str = """
                        Expected: length(pressure) == size(coordinates, 2)
                        Got: size(coordinates, 2) = 2, length(pressure) = 3"""
            @test_throws ArgumentError(error_str) InitialCondition(coordinates=zeros(2, 2),
                                                                   velocity=zeros(2, 2),
                                                                   mass=ones(2),
                                                                   density=ones(2),
                                                                   pressure=ones(3))

            error_str = "`mass` must be specified when not using `particle_spacing`"
            @test_throws ArgumentError(error_str) InitialCondition(coordinates=zeros(2, 2),
                                                                   velocity=zeros(2, 2),
                                                                   density=ones(2))

            error_str = "density must be positive and larger than `eps()`"
            @test_throws ArgumentError(error_str) InitialCondition(coordinates=zeros(2, 2),
                                                                   velocity=zeros(2, 2),
                                                                   density=zeros(2),
                                                                   mass=ones(2))
        end

        @testset "Constant Quantities" begin
            ic_actual1 = InitialCondition(coordinates=zeros(2, 5), velocity=(1.0, 2.0),
                                          mass=3.0, density=4.0, pressure=5.0)
            ic_actual2 = InitialCondition(coordinates=zeros(2, 5), velocity=[1.0, 2.0],
                                          mass=3.0, density=4.0, pressure=5.0)
            ic_expected = InitialCondition(coordinates=zeros(2, 5),
                                           velocity=(1, 2) .* ones(2, 5),
                                           mass=3 * ones(5), density=4 * ones(5),
                                           pressure=5 * ones(5))

            @test ic_actual1.coordinates == ic_actual2.coordinates ==
                  ic_expected.coordinates
            @test ic_actual1.velocity == ic_actual2.velocity == ic_expected.velocity
            @test ic_actual1.mass == ic_actual2.mass == ic_expected.mass
            @test ic_actual1.density == ic_actual2.density == ic_expected.density
            @test ic_actual1.pressure == ic_actual2.pressure == ic_expected.pressure
        end

        @testset "Automatic Mass Calculation" begin
            particle_spacing = 0.13
            coordinates = [88.3 10.4 5.2 48.3 58.9;
                           23.6 92.5 92.1 96.7 84.8;
                           77.5 44.1 18.2 30.5 44.0]
            ic_actual = InitialCondition(; coordinates, velocity=x -> 2x,
                                         density=x -> 4x[2], pressure=x -> 5x[3],
                                         particle_spacing)
            ic_expected = InitialCondition(; coordinates, velocity=2coordinates,
                                           mass=particle_spacing^3 * 4coordinates[2, :],
                                           density=4coordinates[2, :],
                                           pressure=5coordinates[3, :])

            @test ic_actual.coordinates == ic_expected.coordinates
            @test ic_actual.velocity == ic_expected.velocity
            @test ic_actual.mass == ic_expected.mass
            @test ic_actual.density == ic_expected.density
            @test ic_actual.pressure == ic_expected.pressure
        end

        @testset "Quantities as Functions" begin
            coordinates = [88.3 10.4 5.2 48.3 58.9;
                           23.6 92.5 92.1 96.7 84.8;
                           77.5 44.1 18.2 30.5 44.0]
            ic_actual = InitialCondition(; coordinates, velocity=x -> 2x, mass=x -> 3x[1],
                                         density=x -> 4x[2], pressure=x -> 5x[3])
            ic_expected = InitialCondition(; coordinates, velocity=2coordinates,
                                           mass=3coordinates[1, :],
                                           density=4coordinates[2, :],
                                           pressure=5coordinates[3, :])

            @test ic_actual.coordinates == ic_expected.coordinates
            @test ic_actual.velocity == ic_expected.velocity
            @test ic_actual.mass == ic_expected.mass
            @test ic_actual.density == ic_expected.density
            @test ic_actual.pressure == ic_expected.pressure
        end
    end

    @testset verbose=true "Union of Disjoint Shapes" begin
        @testset "$key" for key in keys(disjoint_shapes_dict)
            shapes = disjoint_shapes_dict[key]
            initial_condition = union(shapes...)

            # Number of particles should be the sum of the individual numbers of particles
            @test nparticles(initial_condition) ==
                  sum(nparticles(shape) for shape in shapes)
            @test initial_condition.particle_spacing == first(shapes).particle_spacing

            for i in eachindex(shapes)
                start_index = sum((nparticles(shapes[j]) for j in 1:(i - 1)), init=0) + 1
                end_index = start_index + nparticles(shapes[i]) - 1

                # All arrays are just appended
                @test view(initial_condition.coordinates, :, start_index:end_index) ==
                      shapes[i].coordinates
                @test view(initial_condition.velocity, :, start_index:end_index) ==
                      shapes[i].velocity
                @test view(initial_condition.mass, start_index:end_index) ==
                      shapes[i].mass
                @test view(initial_condition.density, start_index:end_index) ==
                      shapes[i].density
            end
        end
    end

    @testset "Union of Shapes with Different Spacing" begin
        shape1 = RectangularShape(0.13, (9, 10), (0.0, 0.0), density=1.0)
        shape2 = RectangularShape(0.1, (4, 5), (1.0, 1.0), density=1.0)

        error = ArgumentError("all passed initial conditions must have the same particle spacing")

        @test_throws error union(shape1, shape2)
    end

    @testset verbose=true "Union of Overlapping Shapes" begin
        @testset "Rectangular Shapes" begin
            shape1 = RectangularShape(0.13, (9, 10), (0.0, 0.0), density=1.0,
                                      loop_order=:x_first)
            shape2 = RectangularShape(0.13, (4, 5), (1.0, 1.0), density=1.0,
                                      loop_order=:x_first)

            initial_condition = union(shape1, shape2)

            expected_coords = [0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.065 1.065 1.065 1.195 1.195 1.195 1.195 1.195 1.325 1.325 1.325 1.325 1.325 1.455 1.455 1.455 1.455 1.455;
                               0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 1.325 1.455 1.585 1.065 1.195 1.325 1.455 1.585 1.065 1.195 1.325 1.455 1.585 1.065 1.195 1.325 1.455 1.585]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle with Added RoundSphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), density=1.0,
                                      loop_order=:x_first)
            shape2 = SphereShape(0.1, 0.3, (0.7, 0.6), 1.0, sphere_type=RoundSphere())

            initial_condition = union(shape1, shape2)

            expected_coords = [-0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.85 0.81490667 0.81490667 0.95 0.93096988 0.8767767 0.8767767 0.93096988;
                               0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.6 0.69641814 0.50358186 0.6 0.69567086 0.7767767 0.4232233 0.50432914]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "RoundSphere Inside Rectangle" begin
            # Same as above, but this will produce a `RoundSphere` inside the rectangle,
            # as the first shape in the union is prioritized.
            shape1 = SphereShape(0.1, 0.3, (0.7, 0.6), 1.0, sphere_type=RoundSphere())
            shape2 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), density=1.0,
                                      loop_order=:x_first)

            initial_condition = union(shape1, shape2)

            expected_coords = [0.75 0.675 0.675 0.85 0.81490667 0.72604723 0.625 0.55904611 0.55904611 0.625 0.72604723 0.81490667 0.95 0.93096988 0.8767767 0.79567086 0.7 0.60432914 0.5232233 0.46903012 0.45 0.46903012 0.5232233 0.60432914 0.7 0.79567086 0.8767767 0.93096988 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.75 0.75 0.75 0.75 0.75 0.75 0.75;
                               0.6 0.64330127 0.55669873 0.6 0.69641814 0.74772116 0.72990381 0.65130302 0.54869698 0.47009619 0.45227884 0.50358186 0.6 0.69567086 0.7767767 0.83096988 0.85 0.83096988 0.7767767 0.69567086 0.6 0.50432914 0.4232233 0.36903012 0.35 0.36903012 0.4232233 0.50432914 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25]
            @test initial_condition.coordinates ≈ expected_coords
        end
    end

    @testset verbose=true "Setdiff of Disjoint Shapes" begin
        @testset "$key" for key in keys(disjoint_shapes_dict)
            shapes = disjoint_shapes_dict[key]
            initial_condition = setdiff(shapes...)

            @test initial_condition.particle_spacing == first(shapes).particle_spacing
            @test initial_condition.coordinates == first(shapes).coordinates
            @test initial_condition.velocity == first(shapes).velocity
            @test initial_condition.mass == first(shapes).mass
            @test initial_condition.density == first(shapes).density
        end
    end

    @testset verbose=true "Setdiff of Overlapping Shapes" begin
        @testset "Rectangular Shapes" begin
            shape1 = RectangularShape(0.13, (9, 10), (0.0, 0.0), density=1.0,
                                      loop_order=:x_first)
            shape2 = RectangularShape(0.13, (4, 5), (1.0, 1.0), density=1.0,
                                      loop_order=:x_first)

            initial_condition = setdiff(shape1, shape2)

            expected_coords = [0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.105;
                               0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle without RoundSphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), density=1.0,
                                      loop_order=:x_first)
            shape2 = SphereShape(0.1, 0.35, (0.0, 0.6), 1.0, sphere_type=RoundSphere())

            initial_condition = setdiff(shape1, shape2)

            expected_coords = [-0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75;
                               0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 1.05 1.15 1.25 0.05 0.15 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle without Low-Res Sphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), density=1.0,
                                      loop_order=:x_first)
            shape2 = SphereShape(0.2, 0.35, (0.0, 0.6), 1.0)

            initial_condition = setdiff(shape1, shape2)

            expected_coords = [-0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75;
                               0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25]
            @test initial_condition.coordinates ≈ expected_coords
        end
    end

    @testset verbose=true "Intersect of Disjoint Shapes" begin
        @testset "$key" for key in keys(disjoint_shapes_dict)
            shapes = disjoint_shapes_dict[key]
            initial_condition = intersect(shapes...)

            @test initial_condition.particle_spacing == first(shapes).particle_spacing
            @test length(initial_condition.coordinates) == 0
            @test length(initial_condition.velocity) == 0
            @test length(initial_condition.mass) == 0
            @test length(initial_condition.density) == 0
        end
    end

    @testset verbose=true "Intersect of Overlapping Shapes" begin
        @testset "Rectangular Shapes" begin
            shape1 = RectangularShape(0.13, (9, 10), (0.0, 0.0), density=1.0)
            shape2 = RectangularShape(0.13, (4, 5), (1.0, 1.0), density=1.0)

            initial_condition = intersect(shape1, shape2)

            expected_coords = [1.105 1.105; 1.105 1.235]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle and RoundSphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), density=1.0,
                                      loop_order=:x_first)
            shape2 = SphereShape(0.1, 0.35, (0.0, 0.6), 1.0, sphere_type=RoundSphere())

            initial_condition = intersect(shape1, shape2)

            expected_coords = [-0.35 -0.35 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35;
                               0.55 0.65 0.35 0.45 0.55 0.65 0.75 0.85 0.35 0.45 0.55 0.65 0.75 0.85 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.35 0.45 0.55 0.65 0.75 0.85 0.35 0.45 0.55 0.65 0.75 0.85 0.55 0.65]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "RoundSphere and Rectangle" begin
            shape1 = SphereShape(0.1, 0.35, (0.7, 0.6), 1.0, sphere_type=RoundSphere())
            shape2 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), density=1.0)

            initial_condition = intersect(shape1, shape2)

            expected_coords = [0.8 0.75 0.65 0.6 0.65 0.75 0.81361295 0.72410734 0.62907902 0.55029785 0.50581164 0.50581164 0.55029785 0.62907902 0.72410734 0.81361295 0.77364565 0.6752262 0.57949137 0.49681553 0.43615787 0.40409161 0.40409161 0.43615787 0.49681553 0.57949137 0.6752262 0.77364565;
                               0.6 0.68660254 0.68660254 0.6 0.51339746 0.51339746 0.76459677 0.79854177 0.78700325 0.73262453 0.64786313 0.55213687 0.46737547 0.41299675 0.40145823 0.43540323 0.89082008 0.89897535 0.874732 0.82071717 0.74278422 0.64937838 0.55062162 0.45721578 0.37928283 0.325268 0.30102465 0.30917992]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle and Low-Res Sphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), density=1.0,
                                      loop_order=:x_first)
            shape2 = SphereShape(0.2, 0.35, (0.0, 0.6), 1.0)

            initial_condition = intersect(shape1, shape2)

            expected_coords = [-0.25 -0.25 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.25 0.25;
                               0.55 0.65 0.55 0.65 0.35 0.45 0.55 0.65 0.75 0.85 0.35 0.45 0.55 0.65 0.75 0.85 0.55 0.65 0.55 0.65]
            @test initial_condition.coordinates ≈ expected_coords
        end
    end
    @testset verbose=true "Move Particles to End" begin
        @testset verbose=true "InitialCondition Variant" begin
            @testset verbose=true "Valid Particle Movement" begin
                # Create a simple initial condition with known particle arrangement
                coordinates = [1.0 2.0 3.0 4.0 5.0;
                               1.0 2.0 3.0 4.0 5.0]
                velocity = [0.1 0.2 0.3 0.4 0.5;
                            0.1 0.2 0.3 0.4 0.5]
                mass = [1.1, 2.2, 3.3, 4.4, 5.5]
                density = [10.0, 20.0, 30.0, 40.0, 50.0]
                pressure = [100.0, 200.0, 300.0, 400.0, 500.0]

                ic = InitialCondition(coordinates=coordinates, velocity=velocity,
                                      mass=mass, density=density, pressure=pressure)

                # Move particles 2 and 4 to the end
                particle_ids_to_move = [2, 4]
                TrixiParticles.move_particles_to_end!(ic, particle_ids_to_move)

                # Expected order after moving: [1, 3, 5, 2, 4]
                expected_coordinates = [1.0 3.0 5.0 2.0 4.0;
                                        1.0 3.0 5.0 2.0 4.0]
                expected_velocity = [0.1 0.3 0.5 0.2 0.4;
                                     0.1 0.3 0.5 0.2 0.4]
                expected_mass = [1.1, 3.3, 5.5, 2.2, 4.4]
                expected_density = [10.0, 30.0, 50.0, 20.0, 40.0]
                expected_pressure = [100.0, 300.0, 500.0, 200.0, 400.0]

                @test ic.coordinates == expected_coordinates
                @test ic.velocity == expected_velocity
                @test ic.mass == expected_mass
                @test ic.density == expected_density
                @test ic.pressure == expected_pressure
            end

            @testset verbose=true "Duplicate Particle IDs" begin
                coordinates = [1.0 2.0 3.0 4.0; 1.0 2.0 3.0 4.0]
                velocity = [0.1 0.2 0.3 0.4; 0.1 0.2 0.3 0.4]
                mass = [1.1, 2.2, 3.3, 4.4]
                density = [10.0, 20.0, 30.0, 40.0]

                ic = InitialCondition(coordinates=coordinates, velocity=velocity,
                                      mass=mass, density=density)

                # Move particle 2 multiple times (should only move once)
                TrixiParticles.move_particles_to_end!(ic, [2, 2, 3])

                # Expected order: [1, 4, 2, 3] (2 and 3 moved to end)
                expected_coordinates = [1.0 4.0 2.0 3.0; 1.0 4.0 2.0 3.0]
                expected_mass = [1.1, 4.4, 2.2, 3.3]

                @test ic.coordinates == expected_coordinates
                @test ic.mass == expected_mass
            end

            @testset verbose=true "Out of Bounds Particle IDs" begin
                ic = rectangular_patch(0.1, (2, 4))

                # Test with invalid particle ID
                @test_throws BoundsError TrixiParticles.move_particles_to_end!(ic, [9])
                @test_throws BoundsError TrixiParticles.move_particles_to_end!(ic, [0])
                @test_throws BoundsError TrixiParticles.move_particles_to_end!(ic, [-1])
            end
        end

        @testset verbose=true "Vector Variant" begin
            @testset verbose=true "Valid Particle Movement" begin
                a = [1.0, 2.0, 3.0, 4.0, 5.0]
                returned = TrixiParticles.move_particles_to_end!(a, [2, 4])

                expected = [1.0, 3.0, 5.0, 2.0, 4.0]
                @test a == expected
                @test returned == expected
            end

            @testset verbose=true "Duplicate Particle IDs" begin
                a = [1, 2, 3, 4]
                TrixiParticles.move_particles_to_end!(a, [2, 2, 3])

                expected = [1, 4, 2, 3]
                @test a == expected
            end

            @testset verbose=true "Out of Bounds Particle IDs" begin
                a = collect(1:8)

                @test_throws BoundsError TrixiParticles.move_particles_to_end!(a, [9])
                @test_throws BoundsError TrixiParticles.move_particles_to_end!(a, [0])
                @test_throws BoundsError TrixiParticles.move_particles_to_end!(a, [-1])
            end

            @testset verbose=true "Empty IDs" begin
                a = [10, 20, 30]
                TrixiParticles.move_particles_to_end!(a, Int[])
                @test a == [10, 20, 30]
            end
        end
    end
end
