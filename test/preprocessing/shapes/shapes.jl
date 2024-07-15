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

            edge_normals = [
                rot([-1.0, 0.0], rot_angle),
                rot([0.0, 1.0], rot_angle),
                rot([1.0, 0.0], rot_angle),
                rot([0.0, -1.0], rot_angle),
            ]

            vertex_normals = [
                rot([[-√2 / 2, -√2 / 2], [-√2 / 2, √2 / 2]], rot_angle), # edge 1
                rot([[-√2 / 2, √2 / 2], [√2 / 2, √2 / 2]], rot_angle),   # edge 2
                rot([[√2 / 2, √2 / 2], [√2 / 2, -√2 / 2]], rot_angle),   # edge 3
                rot([[√2 / 2, -√2 / 2], [-√2 / 2, -√2 / 2]], rot_angle), # edge 4
            ]

            shape_clockwise = TrixiParticles.Polygon(points_rectangular_clockwise)
            @test all(isapprox(shape_clockwise.edge_vertices, edge_vertices, atol=1e-14))
            @test all(isapprox(shape_clockwise.edge_normals, edge_normals, atol=1e-14))
            @test all(isapprox(shape_clockwise.vertex_normals, vertex_normals, atol=1e-14))

            shape_cclockwise = TrixiParticles.Polygon(points_rectangular_counter_clockwise)
            @test all(isapprox(shape_cclockwise.edge_vertices, edge_vertices, atol=1e-14))
            @test all(isapprox(shape_cclockwise.edge_normals, edge_normals, atol=1e-14))
            @test all(isapprox(shape_cclockwise.vertex_normals, vertex_normals, atol=1e-14))
        end
    end

    @testset verbose=true "Real World Data" begin
        data_dir = pkgdir(TrixiParticles, "test", "preprocessing", "data")

        @testset verbose=true "2D" begin
            files = ["hexagon", "circle", "inverted_open_curve"]
            n_edges = [6, 63, 241]

            @testset "Test File `$(files[i])`" for i in eachindex(files)
                # Checked in ParaView with `trixi2vtk(shape)`
                data = TrixiParticles.CSV.read(joinpath(data_dir,
                                                        "normals_" * files[i] * ".csv"),
                                               TrixiParticles.DataFrame)

                vertex_normals = vcat((data.var"vertex_normals:0")',
                                      (data.var"vertex_normals:1")')

                points = vcat((data.var"Points:0")', (data.var"Points:1")')

                shape = load_shape(joinpath(data_dir, files[i] * ".asc"))

                @test TrixiParticles.nfaces(shape) == n_edges[i]

                @testset "Normals $j" for j in eachindex(shape.vertices)
                    @test isapprox(shape.vertex_normals[j][1], vertex_normals[:, j],
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

                vertex_normals = vcat((data.var"vertex_normals:0")',
                                      (data.var"vertex_normals:1")',
                                      (data.var"vertex_normals:2")')

                points = vcat((data.var"Points:0")',
                              (data.var"Points:1")',
                              (data.var"Points:2")')

                shape = load_shape(joinpath(data_dir, files[i] * ".stl"))

                @test TrixiParticles.nfaces(shape) == n_faces[i]
                @test length(shape.vertices) == n_vertices[i]

                @testset "Normals $j" for j in eachindex(shape.vertices)
                    @test isapprox(shape.vertex_normals[j], vertex_normals[:, j], atol=1e-5)
                end
                @testset "Points $j" for j in eachindex(shape.vertices)
                    @test isapprox(shape.vertices[j], points[:, j], atol=1e-5)
                end
            end
        end
    end
end

# ==========================================================================================
# ==== The following functions are only used for debugging yet

function save(filename, mesh; faces=eachface(mesh))
    save(File{format"STL_BINARY"}(filename), mesh; faces)
end

function save(fn::File{format"STL_BINARY"}, mesh::TriangleMesh; faces=eachface(mesh))
    open(fn, "w") do s
        save(s, mesh; faces)
    end
end

function save(f::Stream{format"STL_BINARY"}, mesh::TriangleMesh; faces)
    io = stream(f)
    points = mesh.face_vertices
    normals = mesh.face_normals

    # Implementation made according to https://en.wikipedia.org/wiki/STL_%28file_format%29#Binary_STL
    for i in 1:80 # Write empty header
        write(io, 0x00)
    end

    write(io, UInt32(length(faces))) # Write triangle count
    for i in faces
        n = SVector{3, Float32}(normals[i])
        triangle = points[i]

        for j in 1:3
            write(io, n[j])
        end

        for point in triangle, p in point
            write(io, Float32(p))
        end

        # After the 48 bytes of the normal and the vertices follows a 2-byte unsigned integer
        # that is the "attribute byte count" – in the standard format, this should be zero
        # because most software does not understand anything else.
        write(io, 0x0000) # 16 empty bits
    end
end
