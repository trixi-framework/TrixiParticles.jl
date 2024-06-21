@testset verbose=true "Complex Shapes" begin
    data_dir = pkgdir(TrixiParticles, "test", "preprocessing", "data")

    @testset verbose=true "Complex Shapes 2D" begin
        @testset verbose=true "Rectangular Shifted" begin
            algorithms = [WindingNumberHorman(), WindingNumberJacobson()]
            shifts = [-0.5, 0.0, 0.5]
            particle_spacings = [0.03, 0.05]

            test_name(algorithm, shift, particle_spacing) = "Algorithm: $(TrixiParticles.type2string(algorithm))" *
                                                            ", Shift: $shift" *
                                                            ", Particle Spacing: $particle_spacing"
            @testset verbose=true "$(test_name(point_in_shape_algorithm, shift,
            particle_spacing))" for point_in_shape_algorithm in algorithms,
                                    shift in shifts,
                                    particle_spacing in particle_spacings

                points_rectangular = [[0.0, 0.0] [1.0, 0.0] [1.0, 0.5] [0.0, 0.5]] .+ shift

                shape = TrixiParticles.Polygon(points_rectangular)

                seed = points_rectangular[:, 1] .- 0.5particle_spacing
                shape_sampled = ComplexShape(shape; particle_spacing, density=1.0,
                                             point_in_shape_algorithm, seed)

                min_corner = points_rectangular[:, 1] .+ 0.5particle_spacing
                max_corner = points_rectangular[:, 3]

                ranges_x = min_corner[1]:particle_spacing:max_corner[1]
                ranges_y = min_corner[2]:particle_spacing:max_corner[2]

                coords = hcat(collect.(Iterators.product(ranges_x, ranges_y))...)

                @test shape_sampled.coordinates == coords
            end
        end

        @testset verbose=true "Real World Data" begin
            files = ["hexagon", "circle", "inverted_open_curve"]
            algorithms = [WindingNumberHorman(), WindingNumberJacobson()]
            algorithm_names = ["horman", "jacobson"]

            @testset verbose=true "Algorithm: $(TrixiParticles.type2string(algorithms[i]))" for i in 1:2
                @testset verbose=true "Test File `$(files[j])`" for j in eachindex(files)
                    data = TrixiParticles.CSV.read(joinpath(data_dir,
                                                            "coordinates_" *
                                                            algorithm_names[i] * "_" *
                                                            files[j] * ".csv"),
                                                   TrixiParticles.DataFrame)

                    coords = vcat((data.var"Points:0")', (data.var"Points:1")')

                    shape = load_shape(joinpath(data_dir, files[j] * ".asc"))

                    shape_sampled = ComplexShape(shape; particle_spacing=0.05, density=1.0,
                                                 point_in_shape_algorithm=algorithms[i])

                    @test isapprox(shape_sampled.coordinates, coords, atol=1e-2)
                end
            end
        end
    end

    @testset verbose=true "Complex Shapes 3D" begin
        @testset verbose=true "Real World Data" begin
            files = ["sphere", "bar"]

            @testset verbose=true "Naive Winding" begin
                @testset verbose=true "Test File `$(files[i])`" for i in eachindex(files)
                    data = TrixiParticles.CSV.read(joinpath(data_dir,
                                                            "coordinates_" * files[i] *
                                                            ".csv"),
                                                   TrixiParticles.DataFrame)

                    coords = vcat((data.var"Points:0")',
                                  (data.var"Points:1")',
                                  (data.var"Points:2")')

                    shape = load_shape(joinpath(data_dir, files[i] * ".stl"))

                    shape_sampled = ComplexShape(shape; particle_spacing=0.2, density=1.0,
                                                 point_in_shape_algorithm=WindingNumberJacobson(;
                                                                                                shape,
                                                                                                winding_number_factor=0.1))
                    @test isapprox(shape_sampled.coordinates, coords, atol=1e-3)
                end
            end
            @testset verbose=true "Hierarchical Winding" begin
                @testset verbose=true "Test File `$(files[i])`" for i in eachindex(files)
                    data = TrixiParticles.CSV.read(joinpath(data_dir,
                                                            "coordinates_" * files[i] *
                                                            ".csv"),
                                                   TrixiParticles.DataFrame)

                    coords = vcat((data.var"Points:0")',
                                  (data.var"Points:1")',
                                  (data.var"Points:2")')

                    shape = load_shape(joinpath(data_dir, files[i] * ".stl"))

                    shape_sampled = ComplexShape(shape; particle_spacing=0.2, density=1.0,
                                                 point_in_shape_algorithm=WindingNumberJacobson(;
                                                                                                shape,
                                                                                                winding_number_factor=0.1,
                                                                                                hierarchical_winding=true))
                    @test isapprox(shape_sampled.coordinates, coords, atol=1e-3)
                end
            end
        end
    end
end
