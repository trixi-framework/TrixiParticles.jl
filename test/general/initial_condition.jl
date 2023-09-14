@testset verbose=true "InitialCondition" begin
    @testset verbose=true "Union of Disjoint Shapes" begin
        shapes_dict = Dict(
            "Rectangular Shapes" => (RectangularShape(0.1, (3, 4), (-1.0, 1.0), 1.0),
                                     RectangularShape(0.1, (4, 5), (0.0, 1.0), 1.0,
                                                      init_velocity=(0.3, -0.5))),
            "Touching Rectangular Shapes" => (RectangularShape(0.1, (3, 4), (-1.0, 1.0),
                                                               1.0),
                                              RectangularShape(0.1, (4, 5), (0.0, 1.0),
                                                               1.0),
                                              RectangularShape(0.1, (2, 10), (0.0, 0.0),
                                                               1.0)),
            "Sphere Shapes" => (SphereShape(0.15, 0.5, (-1.0, 1.0), 1.0),
                                SphereShape(0.15, 0.2, (0.0, 1.0), 1.0),
                                SphereShape(0.15, 1.0, (0.0, -0.2), 1.0,
                                            sphere_type=RoundSphere())),
            "Touching Mixed Shapes" => (RectangularShape(0.1, (3, 10), (-1.0, 0.0), 1.0),
                                        SphereShape(0.1, 0.5, (-1.0, 1.5), 1000.0),
                                        SphereShape(0.1, 0.5, (1.0, 0.5), 1000.0,
                                                    sphere_type=RoundSphere())))

        @testset "$key" for key in keys(shapes_dict)
            shapes = shapes_dict[key]
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
        shape1 = RectangularShape(0.13, (9, 10), (0.0, 0.0), 1.0)
        shape2 = RectangularShape(0.1, (4, 5), (1.0, 1.0), 1.0)

        error = ArgumentError("all passed initial conditions must have the same particle spacing")

        @test_throws error union(shape1, shape2)
    end

    @testset verbose=true "Union of Overlapping Shapes" begin
        @testset "Rectangular Shapes" begin
            shape1 = RectangularShape(0.13, (9, 10), (0.0, 0.0), 1.0)
            shape2 = RectangularShape(0.13, (4, 5), (1.0, 1.0), 1.0)

            initial_condition = union(shape1, shape2)

            expected_coords = [0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.065 1.065 1.065 1.195 1.195 1.195 1.195 1.195 1.325 1.325 1.325 1.325 1.325 1.455 1.455 1.455 1.455 1.455;
                               0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 1.325 1.455 1.585 1.065 1.195 1.325 1.455 1.585 1.065 1.195 1.325 1.455 1.585 1.065 1.195 1.325 1.455 1.585]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle with Added RoundSphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), 1.0)
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
            shape2 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), 1.0)

            initial_condition = union(shape1, shape2)

            expected_coords = [0.75 0.675 0.675 0.85 0.81490667 0.72604723 0.625 0.55904611 0.55904611 0.625 0.72604723 0.81490667 0.95 0.93096988 0.8767767 0.79567086 0.7 0.60432914 0.5232233 0.46903012 0.45 0.46903012 0.5232233 0.60432914 0.7 0.79567086 0.8767767 0.93096988 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.75 0.75 0.75 0.75 0.75 0.75 0.75;
                               0.6 0.64330127 0.55669873 0.6 0.69641814 0.74772116 0.72990381 0.65130302 0.54869698 0.47009619 0.45227884 0.50358186 0.6 0.69567086 0.7767767 0.83096988 0.85 0.83096988 0.7767767 0.69567086 0.6 0.50432914 0.4232233 0.36903012 0.35 0.36903012 0.4232233 0.50432914 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25]
            @test initial_condition.coordinates ≈ expected_coords
        end
    end

    @testset verbose=true "Setdiff of Disjoint Shapes" begin
        shapes_dict = Dict(
            "Rectangular Shapes" => (RectangularShape(0.1, (3, 4), (-1.0, 1.0), 1.0),
                                     RectangularShape(0.1, (4, 5), (0.0, 1.0), 1.0,
                                                      init_velocity=(0.3, -0.5))),
            "Touching Rectangular Shapes" => (RectangularShape(0.1, (3, 4), (-1.0, 1.0),
                                                               1.0),
                                              RectangularShape(0.1, (4, 5), (0.0, 1.0),
                                                               1.0),
                                              RectangularShape(0.1, (2, 10), (0.0, 0.0),
                                                               1.0)),
            "Sphere Shapes" => (SphereShape(0.15, 0.5, (-1.0, 1.0), 1.0),
                                SphereShape(0.15, 0.2, (0.0, 1.0), 1.0),
                                SphereShape(0.15, 1.0, (0.0, -0.2), 1.0,
                                            sphere_type=RoundSphere())),
            "Touching Mixed Shapes" => (RectangularShape(0.1, (3, 10), (-1.0, 0.0), 1.0),
                                        SphereShape(0.1, 0.5, (-1.0, 1.5), 1000.0),
                                        SphereShape(0.1, 0.5, (1.0, 0.5), 1000.0,
                                                    sphere_type=RoundSphere())))

        @testset "$key" for key in keys(shapes_dict)
            shapes = shapes_dict[key]
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
            shape1 = RectangularShape(0.13, (9, 10), (0.0, 0.0), 1.0)
            shape2 = RectangularShape(0.13, (4, 5), (1.0, 1.0), 1.0)

            initial_condition = setdiff(shape1, shape2)

            expected_coords = [0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.065 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.195 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.325 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.455 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.585 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.715 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.845 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 0.975 1.105 1.105 1.105 1.105 1.105 1.105 1.105 1.105;
                               0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975 1.105 1.235 0.065 0.195 0.325 0.455 0.585 0.715 0.845 0.975]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle without RoundSphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), 1.0)
            shape2 = SphereShape(0.1, 0.35, (0.0, 0.6), 1.0, sphere_type=RoundSphere())

            initial_condition = setdiff(shape1, shape2)

            expected_coords = [-0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75;
                               0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 1.05 1.15 1.25 0.05 0.15 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle without Low-Res Sphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), 1.0)
            shape2 = SphereShape(0.2, 0.35, (0.0, 0.6), 1.0)

            initial_condition = setdiff(shape1, shape2)

            expected_coords = [-0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.75 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.65 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.55 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.45 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.35 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75;
                               0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 1.15 1.25]
            @test initial_condition.coordinates ≈ expected_coords
        end
    end

    @testset verbose=true "Intersect of Disjoint Shapes" begin
        shapes_dict = Dict(
            "Rectangular Shapes" => (RectangularShape(0.1, (3, 4), (-1.0, 1.0), 1.0),
                                     RectangularShape(0.1, (4, 5), (0.0, 1.0), 1.0,
                                                      init_velocity=(0.3, -0.5))),
            "Touching Rectangular Shapes" => (RectangularShape(0.1, (3, 4), (-1.0, 1.0),
                                                               1.0),
                                              RectangularShape(0.1, (4, 5), (0.0, 1.0),
                                                               1.0),
                                              RectangularShape(0.1, (2, 10), (0.0, 0.0),
                                                               1.0)),
            "Sphere Shapes" => (SphereShape(0.15, 0.5, (-1.0, 1.0), 1.0),
                                SphereShape(0.15, 0.2, (0.0, 1.0), 1.0),
                                SphereShape(0.15, 1.0, (0.0, -0.2), 1.0,
                                            sphere_type=RoundSphere())),
            "Touching Mixed Shapes" => (RectangularShape(0.1, (3, 10), (-1.0, 0.0), 1.0),
                                        SphereShape(0.1, 0.5, (-1.0, 1.5), 1000.0),
                                        SphereShape(0.1, 0.5, (1.0, 0.5), 1000.0,
                                                    sphere_type=RoundSphere())))

        @testset "$key" for key in keys(shapes_dict)
            shapes = shapes_dict[key]
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
            shape1 = RectangularShape(0.13, (9, 10), (0.0, 0.0), 1.0)
            shape2 = RectangularShape(0.13, (4, 5), (1.0, 1.0), 1.0)

            initial_condition = intersect(shape1, shape2)

            expected_coords = [1.105 1.105; 1.105 1.235]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle and RoundSphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), 1.0)
            shape2 = SphereShape(0.1, 0.35, (0.0, 0.6), 1.0, sphere_type=RoundSphere())

            initial_condition = intersect(shape1, shape2)

            expected_coords = [-0.35 -0.35 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35;
                               0.55 0.65 0.35 0.45 0.55 0.65 0.75 0.85 0.35 0.45 0.55 0.65 0.75 0.85 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.35 0.45 0.55 0.65 0.75 0.85 0.35 0.45 0.55 0.65 0.75 0.85 0.55 0.65]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "RoundSphere and Rectangle" begin
            shape1 = SphereShape(0.1, 0.35, (0.7, 0.6), 1.0, sphere_type=RoundSphere())
            shape2 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), 1.0)

            initial_condition = intersect(shape1, shape2)

            expected_coords = [0.8 0.75 0.65 0.6 0.65 0.75 0.81361295 0.72410734 0.62907902 0.55029785 0.50581164 0.50581164 0.55029785 0.62907902 0.72410734 0.81361295 0.77364565 0.6752262 0.57949137 0.49681553 0.43615787 0.40409161 0.40409161 0.43615787 0.49681553 0.57949137 0.6752262 0.77364565;
                               0.6 0.68660254 0.68660254 0.6 0.51339746 0.51339746 0.76459677 0.79854177 0.78700325 0.73262453 0.64786313 0.55213687 0.46737547 0.41299675 0.40145823 0.43540323 0.89082008 0.89897535 0.874732 0.82071717 0.74278422 0.64937838 0.55062162 0.45721578 0.37928283 0.325268 0.30102465 0.30917992]
            @test initial_condition.coordinates ≈ expected_coords
        end

        @testset "Rectangle and Low-Res Sphere" begin
            shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), 1.0)
            shape2 = SphereShape(0.2, 0.35, (0.0, 0.6), 1.0)

            initial_condition = intersect(shape1, shape2)

            expected_coords = [-0.25 -0.25 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.25 0.25;
                               0.55 0.65 0.55 0.65 0.35 0.45 0.55 0.65 0.75 0.85 0.35 0.45 0.55 0.65 0.75 0.85 0.55 0.65 0.55 0.65]
            @test initial_condition.coordinates ≈ expected_coords
        end
    end
end

@testset "System Buffer" begin
    buffer_particles = 5

    shape_1 = RectangularShape(0.1, (3, 6), (0.0, 0.0), 1.0)
    shape_2 = RectangularShape(0.1, (3, 6), (0.0, 0.0), 1.0, buffer=buffer_particles)
    shape_3 = RectangularShape(0.1, (4, 5), (0.0, 0.0), 1.0)

    initial_condition = InitialCondition(shape_1, shape_3, buffer=buffer_particles)

    @test size(shape_1.coordinates) == (2, 18)
    @test size(shape_2.coordinates) == (2, 18 + buffer_particles)
    @test size(shape_2.velocity) == (2, 18 + buffer_particles)
    @test size(shape_2.mass) == (18 + buffer_particles,)
    @test size(shape_2.density) == (18 + buffer_particles,)

    @test size(initial_condition.coordinates) == (2, 38 + buffer_particles)
    @test size(initial_condition.velocity) == (2, 38 + buffer_particles)
    @test size(initial_condition.mass) == (38 + buffer_particles,)
    @test size(initial_condition.density) == (38 + buffer_particles,)

    @test shape_1.buffer == nothing
    @test shape_2.buffer isa TrixiParticles.SystemBuffer
    @test initial_condition.buffer isa TrixiParticles.SystemBuffer

    error_str = "You have passed `buffer` before. Please pass `buffer` only here."
    @test_throws ArgumentError(error_str) InitialCondition(shape_1, shape_2)

    error_str = "invalid buffer: 1.0 of type Float64"
    @test_throws ArgumentError(error_str) RectangularShape(0.1, (2, 2), (0, 0), 1.0,
                                                           buffer=1.0)
end
