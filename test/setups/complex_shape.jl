@testset verbose=true "ComplexShape" begin
    data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
    validation_dir = pkgdir(TrixiParticles, "test", "preprocessing", "data")

    @testset verbose=true "2D" begin
        @testset verbose=true "Shifted Rectangle" begin
            algorithms = [:winding_horman, :winding_jacobson]
            shifts = [-0.5, 0.0, 0.5]
            particle_spacings = [0.03, 0.05]

            test_name(algorithm, shift,
                      particle_spacing) = "Algorithm: $(algorithm)" *
                                          ", Shift: $shift" *
                                          ", Particle Spacing: $particle_spacing"
            @testset verbose=true "$(test_name("$algorithm", shift,
            particle_spacing))" for algorithm in algorithms, shift in shifts,
                                    particle_spacing in particle_spacings
                points_rectangle = stack([[0.0, 0.0], [1.0, 0.0],
                                             [1.0, 0.5], [0.0, 0.5], [0.0, 0.0]]) .+ shift

                geometry = TrixiParticles.Polygon(points_rectangle)

                point_in_geometry_algorithm = algorithm == :winding_horman ?
                                              WindingNumberHormann() :
                                              WindingNumberJacobson(geometry;
                                                                    winding=NaiveWinding())

                grid_offset = 0.5 * particle_spacing
                shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                                             point_in_geometry_algorithm, grid_offset)

                min_corner = points_rectangle[:, 1] .+ 0.5particle_spacing
                max_corner = points_rectangle[:, 3]

                ranges_x = min_corner[1]:particle_spacing:max_corner[1]
                ranges_y = min_corner[2]:particle_spacing:max_corner[2]

                coords = hcat(collect.(Iterators.product(ranges_x, ranges_y))...)

                @test isapprox(shape_sampled.coordinates, coords)
            end
        end

        @testset verbose=true "Real World Data" begin
            files = ["hexagon", "circle", "inverted_open_curve"]
            algorithms = [:winding_horman, :winding_jacobson]
            algorithm_names = ["hormann", "jacobson"]

            @testset verbose=true "Algorithm: $(algorithms[i])" for i in 1:2
                @testset verbose=true "Test File `$(files[j])`" for j in eachindex(files)
                    geometry = load_geometry(joinpath(data_dir, files[j] * ".asc"))

                    point_in_geometry_algorithm = algorithms[i] == :winding_horman ?
                                                  WindingNumberHormann() :
                                                  WindingNumberJacobson(geometry;
                                                                        winding=NaiveWinding())

                    # Relaxed inside-outside segmentation for open geometry
                    if (i == 2 && j == 3)
                        point_in_geometry_algorithm = WindingNumberJacobson(geometry;
                                                                            winding=NaiveWinding(),
                                                                            winding_number_factor=0.4)
                    end

                    data = TrixiParticles.CSV.read(joinpath(validation_dir,
                                                            "coordinates_" *
                                                            algorithm_names[i] * "_" *
                                                            files[j] * ".csv"),
                                                   TrixiParticles.DataFrame)

                    # Access the field called `Points:0` of `data`. Since `data.Points:0` is not a valid
                    # identifier name, we need to use `var"name"`.
                    # See https://docs.julialang.org/en/v1/base/base/#var%22name%22
                    coords = vcat((data.var"Points:0")', (data.var"Points:1")')

                    shape_sampled = ComplexShape(geometry; particle_spacing=0.05,
                                                 density=1.0, point_in_geometry_algorithm)

                    @test isapprox(shape_sampled.coordinates, coords, atol=1e-2)
                end
            end
        end

        @testset verbose=true "Intersect of Overlapping Shapes and Geometries" begin
            shape = RectangularShape(0.1, (10, 10), (0.0, 0.0), density=1.0)
            geometry = load_geometry(joinpath(data_dir, "circle.asc"))

            initial_condition = intersect(shape, geometry)

            expected_coords = [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25;
                               0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.85 0.85 0.85 0.85 0.85 0.95 0.95 0.95]
            @test isapprox(initial_condition.coordinates, expected_coords)
        end

        @testset verbose=true "Setdiff of Overlapping Shapes and Geometries" begin
            shape = RectangularShape(0.1, (10, 10), (0.0, 0.0), density=1.0)
            geometry = load_geometry(joinpath(data_dir, "circle.asc"))

            initial_condition = setdiff(shape, geometry)

            expected_coords = [0.95 0.95 0.85 0.95 0.85 0.95 0.75 0.85 0.95 0.55 0.65 0.75 0.85 0.95 0.35 0.45 0.55 0.65 0.75 0.85 0.95;
                               0.35 0.45 0.55 0.55 0.65 0.65 0.75 0.75 0.75 0.85 0.85 0.85 0.85 0.85 0.95 0.95 0.95 0.95 0.95 0.95 0.95]
            @test isapprox(initial_condition.coordinates, expected_coords)
        end
    end

    @testset verbose=true "3D" begin
        @testset verbose=true "Real World Data" begin
            files = ["sphere", "bar"]
            particle_spacings = [0.1, 0.18]

            @testset verbose=true "Naive Winding" begin
                @testset verbose=true "Test File `$(files[i])`" for i in eachindex(files)
                    data = TrixiParticles.CSV.read(joinpath(validation_dir,
                                                            "coordinates_" * files[i] *
                                                            ".csv"),
                                                   TrixiParticles.DataFrame)

                    coords = vcat((data.var"Points:0")',
                                  (data.var"Points:1")',
                                  (data.var"Points:2")')

                    geometry = load_geometry(joinpath(data_dir, files[i] * ".stl"))

                    shape_sampled = ComplexShape(geometry; grid_offset=0.1,
                                                 particle_spacing=particle_spacings[i],
                                                 density=1.0)
                    @test isapprox(shape_sampled.coordinates, coords, atol=1e-3)
                end
            end
            @testset verbose=true "Hierarchical Winding" begin
                @testset verbose=true "Test File `$(files[i])`" for i in eachindex(files)
                    data = TrixiParticles.CSV.read(joinpath(validation_dir,
                                                            "coordinates_" * files[i] *
                                                            ".csv"),
                                                   TrixiParticles.DataFrame)

                    coords = vcat((data.var"Points:0")',
                                  (data.var"Points:1")',
                                  (data.var"Points:2")')

                    geometry = load_geometry(joinpath(data_dir, files[i] * ".stl"))

                    shape_sampled = ComplexShape(geometry;
                                                 particle_spacing=particle_spacings[i],
                                                 density=1.0, grid_offset=0.1,
                                                 point_in_geometry_algorithm=WindingNumberJacobson(geometry;
                                                                                                   winding_number_factor=0.1))
                    @test isapprox(shape_sampled.coordinates, coords, atol=1e-3)
                end
            end
        end
        @testset verbose=true "Intersect of Overlapping Shapes and Geometries" begin
            shape = RectangularShape(0.1, (10, 10, 10), (0.0, 0.0, 0.0), density=1.0)
            geometry = load_geometry(joinpath(data_dir, "sphere.stl"))

            initial_condition = intersect(shape, geometry)

            expected_coords = [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.05 0.15 0.25 0.35 0.45 0.55 0.05 0.15 0.25 0.35 0.45 0.55 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25 0.35 0.05 0.15 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.05 0.15 0.25 0.35 0.45 0.55 0.05 0.15 0.25 0.35 0.45 0.55 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25 0.35 0.05 0.15 0.05 0.15 0.25 0.35 0.45 0.55 0.05 0.15 0.25 0.35 0.45 0.55 0.05 0.15 0.25 0.35 0.45 0.55 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25 0.05 0.15 0.25 0.35 0.45 0.55 0.05 0.15 0.25 0.35 0.45 0.55 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25 0.35 0.05 0.15 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25 0.35 0.45 0.05 0.15 0.25 0.35 0.05 0.15 0.25 0.05 0.15 0.25 0.35 0.05 0.15 0.25 0.35 0.05 0.15 0.25 0.05 0.15 0.05 0.15 0.05 0.15;
                               0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.65 0.65 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.65 0.65 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.55 0.55 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.35 0.35 0.05 0.05 0.15 0.15;
                               0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65]
            @test isapprox(initial_condition.coordinates, expected_coords)
        end

        @testset verbose=true "Setdiff of Overlapping Shapes and Geometries" begin
            shape = RectangularShape(0.1, (10, 10, 10), (-0.3, -0.3, -0.3), density=1.0)
            geometry = load_geometry(joinpath(data_dir, "sphere.stl"))

            initial_condition = setdiff(shape, geometry)

            expected_coords = [0.65 0.65 0.65 0.65 0.65 0.65 0.55 0.65 0.55 0.65 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.65 0.65 0.65 0.55 0.65 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 0.65 0.65 0.65 0.55 0.65 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 0.65 0.65 0.65 0.55 0.65 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 0.65 0.65 0.65 0.55 0.65 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.55 0.65 0.55 0.65 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.55 0.65 0.65 0.65 0.65 0.65 0.55 0.65 0.55 0.65 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.55 0.65 0.55 0.65 0.55 0.65 0.55 0.65 0.55 0.65 0.55 0.65 0.45 0.55 0.65 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.35 0.45 0.55 0.65 0.45 0.55 0.65 0.45 0.55 0.65 0.45 0.55 0.65 0.45 0.55 0.65 0.35 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 -0.25 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65;
                               -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.35 0.45 0.45 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 -0.25 0.25 0.35 0.45 0.45 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 -0.25 0.25 0.35 0.45 0.45 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 -0.25 0.25 0.35 0.45 0.45 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 -0.25 0.25 0.35 0.45 0.45 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 0.35 0.45 0.45 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 -0.25 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.25 0.35 0.35 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 -0.25 -0.25 -0.15 -0.15 -0.05 -0.05 0.05 0.05 0.15 0.15 0.25 0.25 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65;
                               -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65]
            @test isapprox(initial_condition.coordinates, expected_coords)
        end
    end
end
