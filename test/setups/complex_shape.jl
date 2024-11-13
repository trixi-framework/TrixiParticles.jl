@testset verbose=true "ComplexShape" begin
    data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
    validation_dir = pkgdir(TrixiParticles, "test", "preprocessing", "data")

    @testset verbose=true "2D" begin
        @testset verbose=true "Shifted Rectangle" begin
            algorithms = [WindingNumberHormann(), WindingNumberJacobson()]
            shifts = [-0.5, 0.0, 0.5]
            particle_spacings = [0.03, 0.05]

            test_name(algorithm, shift, particle_spacing) = "Algorithm: $(TrixiParticles.type2string(algorithm))" *
                                                            ", Shift: $shift" *
                                                            ", Particle Spacing: $particle_spacing"
            @testset verbose=true "$(test_name(point_in_geometry_algorithm, shift,
            particle_spacing))" for point_in_geometry_algorithm in algorithms,
                                    shift in shifts,
                                    particle_spacing in particle_spacings

                points_rectangle = stack([[0.0, 0.0], [1.0, 0.0],
                                             [1.0, 0.5], [0.0, 0.5], [0.0, 0.0]]) .+ shift

                geometry = TrixiParticles.Polygon(points_rectangle)

                grid_offset = 0.5particle_spacing
                shape_sampled = ComplexShape(geometry; particle_spacing, density=1.0,
                                             point_in_geometry_algorithm, grid_offset)

                min_corner = points_rectangle[:, 1] .+ 0.5particle_spacing
                max_corner = points_rectangle[:, 3]

                ranges_x = min_corner[1]:particle_spacing:max_corner[1]
                ranges_y = min_corner[2]:particle_spacing:max_corner[2]

                coords = hcat(collect.(Iterators.product(ranges_x, ranges_y))...)

                @test isapprox(shape_sampled.initial_condition.coordinates, coords)
            end
        end

        @testset verbose=true "Real World Data" begin
            files = ["hexagon", "circle", "inverted_open_curve"]
            algorithms = [WindingNumberHormann(), WindingNumberJacobson()]
            algorithm_names = ["hormann", "jacobson"]

            @testset verbose=true "Algorithm: $(TrixiParticles.type2string(algorithms[i]))" for i in 1:2
                @testset verbose=true "Test File `$(files[j])`" for j in eachindex(files)
                    point_in_geometry_algorithm = algorithms[i]

                    # Relaxed inside-outside segmentation for open geometry
                    if (i == 2 && j == 3)
                        point_in_geometry_algorithm = WindingNumberJacobson(;
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

                    geometry = load_geometry(joinpath(data_dir, files[j] * ".asc"))

                    shape_sampled = ComplexShape(geometry; particle_spacing=0.05,
                                                 density=1.0, point_in_geometry_algorithm)

                    @test isapprox(shape_sampled.initial_condition.coordinates, coords,
                                   atol=1e-2)
                end
            end
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
                    @test isapprox(shape_sampled.initial_condition.coordinates, coords,
                                   atol=1e-3)
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
                                                 point_in_geometry_algorithm=WindingNumberJacobson(;
                                                                                                   geometry,
                                                                                                   winding_number_factor=0.1,
                                                                                                   hierarchical_winding=true))
                    @test isapprox(shape_sampled.initial_condition.coordinates, coords,
                                   atol=1e-3)
                end
            end
        end
    end
end
