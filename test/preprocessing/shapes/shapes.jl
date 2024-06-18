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

            shape = TrixiParticles.Polygon(points_rectangular_counter_clockwise)

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

            @test all(shape.edge_vertices .≈ edge_vertices)
            @test all(shape.normals_edge .≈ normals_edge)
            @test all(shape.normals_vertex .≈ normals_vertex)
        end
    end

    @testset verbose=true "Real World Data" begin
        @testset verbose=true "2D" begin
            files = ["hexagon", "circle", "inverted_open_curve"]
            n_edges = [6, 63, 241]
            spot_check = [3, 17, 137] # Random edge samples

            normals_edge = [
                [-0.5000001748438417, -0.8660253028382761],
                [-0.06234627263265082, -0.9980545788126094],
                [0.8255794226893319, -0.5642859353483385],
            ]

            @testset verbose=true "Test File `$(files[i])`" for i in eachindex(files)
                shape = load_shape(joinpath("examples", "preprocessing", files[i] * ".asc"))

                @test TrixiParticles.nfaces(shape) == n_edges[i]
                @test shape.normals_edge[spot_check[i]] ≈ normals_edge[i]
            end
        end
        @testset verbose=true "3D" begin
            files = ["sphere", "bar", "drive_gear"]
            n_faces = [192, 12, 1392]
            n_vertices = [98, 8, 696]
            spot_check = [37, 5, 329] # Random face samples

            normals_vertex = [
                [0.30382471570741304, -0.6736804313398639, -0.6736803533984385],
                [0.5773502691896257, -0.5773502691896258, -0.5773502691896258],
                [0.5715080389137815, -0.3430758125127828, -0.7454378232459212],
            ]

            @testset verbose=true "Test File `$(files[i])`" for i in eachindex(files)
                shape = load_shape(joinpath("examples", "preprocessing", files[i] * ".stl"))

                @test TrixiParticles.nfaces(shape) == n_faces[i]
                @test length(shape.vertices) == n_vertices[i]
                @test shape.normals_vertex[spot_check[i]] ≈ normals_vertex[i]
            end
        end
    end
end
