@testset verbose=true "Shapes" begin
    @testset verbose=true "Rectangular Analytical" begin
        rot(vec, α) = [cos(α) -sin(α); sin(α) cos(α)] * vec

        @testset "Rotation Angle $(rot_angle)" for rot_angle in 0.0:(π / 8):(2π)
            v1 = rot([0.0, 0.0], rot_angle)
            v2 = rot([0.0, 0.5], rot_angle)
            v3 = rot([1.0, 0.5], rot_angle)
            v4 = rot([1.0, 0.0], rot_angle)

            points_rectangular_clockwise = [v1 v2 v3 v4]
            points_rectangular_counter_clockwise = [v1 v4 v3 v2]

            edge_vertices = [[v1, v2], [v2, v3], [v3, v4], [v4, v1]]

            normals_edge = [
                rot([-1.0, 0.0], rot_angle),
                rot([0.0, 1.0], rot_angle),
                rot([1.0, 0.0], rot_angle),
                rot([0.0, -1.0], rot_angle),
            ]

            normals_vertex = [
                rot([[-√2 / 2, -√2 / 2], [-√2 / 2, √2 / 2]], rot_angle), # edge 1
                rot([[-√2 / 2, √2 / 2], [√2 / 2, √2 / 2]], rot_angle),   # edge 2
                rot([[√2 / 2, √2 / 2], [√2 / 2, -√2 / 2]], rot_angle),   # edge 3
                rot([[√2 / 2, -√2 / 2], [-√2 / 2, -√2 / 2]], rot_angle), # edge 4
            ]

            shape_clockwise = TrixiParticles.Polygon(points_rectangular_clockwise)
            @test all(isapprox(shape_clockwise.edge_vertices, edge_vertices, atol=1e-14))
            @test all(isapprox(shape_clockwise.normals_edge, normals_edge, atol=1e-14))
            @test all(isapprox(shape_clockwise.normals_vertex, normals_vertex, atol=1e-14))

            shape_cclockwise = TrixiParticles.Polygon(points_rectangular_counter_clockwise)
            @test all(isapprox(shape_cclockwise.edge_vertices, edge_vertices, atol=1e-14))
            @test all(isapprox(shape_cclockwise.normals_edge, normals_edge, atol=1e-14))
            @test all(isapprox(shape_cclockwise.normals_vertex, normals_vertex, atol=1e-14))
        end
    end

    @testset verbose=true "Real World Data" begin
        data_dir = joinpath("..", "test", "preprocessing", "data")

        @testset verbose=true "2D" begin
            files = ["hexagon", "circle", "inverted_open_curve"]
            n_edges = [6, 63, 241]

            @testset "Test File `$(files[i])`" for i in eachindex(files)
                # Checked in ParaView with `trixi2vtk(shape)`
                data = TrixiParticles.CSV.read(joinpath(data_dir,
                                                        "normals_" * files[i] * ".csv"),
                                               TrixiParticles.DataFrame)

                normals_vertex = vcat((data.var"normals_vertex:0")',
                                      (data.var"normals_vertex:1")')

                points = vcat((data.var"Points:0")', (data.var"Points:1")')

                shape = load_shape(joinpath("examples", "preprocessing", files[i] * ".asc"))

                @test TrixiParticles.nfaces(shape) == n_edges[i]

                @testset "Normals $j" for j in eachindex(shape.vertices)
                    @test isapprox(shape.normals_vertex[j][1], normals_vertex[:, j],
                                   atol=1e-4)
                end
                @testset "Points $j" for j in eachindex(shape.vertices)
                    @test isapprox(shape.vertices[j], points[:, j], atol=1e-4)
                end
            end
        end
        @testset verbose=true "3D" begin
            files = ["sphere", "bar", "drive_gear"]
            n_faces = [192, 12, 1392]
            n_vertices = [98, 8, 696]

            @testset "Test File `$(files[i])`" for i in eachindex(files)
                # Checked in ParaView with `trixi2vtk(shape)`
                data = TrixiParticles.CSV.read(joinpath(data_dir,
                                                        "normals_" * files[i] * ".csv"),
                                               TrixiParticles.DataFrame)

                normals_vertex = vcat((data.var"normals_vertex:0")',
                                      (data.var"normals_vertex:1")',
                                      (data.var"normals_vertex:2")')

                points = vcat((data.var"Points:0")',
                              (data.var"Points:1")',
                              (data.var"Points:2")')

                shape = load_shape(joinpath("examples", "preprocessing", files[i] * ".stl"))

                @test TrixiParticles.nfaces(shape) == n_faces[i]
                @test length(shape.vertices) == n_vertices[i]

                @testset "Normals $j" for j in eachindex(shape.vertices)
                    @test isapprox(shape.normals_vertex[j], normals_vertex[:, j], atol=1e-5)
                end
                @testset "Points $j" for j in eachindex(shape.vertices)
                    @test isapprox(shape.vertices[j], points[:, j], atol=1e-5)
                end
            end
        end
    end
end
