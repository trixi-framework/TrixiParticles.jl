@testset verbose=true "Geometries" begin
    @testset verbose=true "Rectangular Analytical" begin
        rot(vec, alpha) = SVector{2}([cos(alpha) -sin(alpha); sin(alpha) cos(alpha)] * vec)

        @testset "Rotation Angle $(rot_angle)" for rot_angle in 0.0:(pi / 8):(2pi)
            v1 = rot([0.0, 0.0], rot_angle)
            v2 = rot([0.0, 0.5], rot_angle)
            v3 = rot([1.0, 0.5], rot_angle)
            v4 = rot([1.0, 0.0], rot_angle)

            points_rectangular_clockwise = [v1 v2 v3 v4 v1]
            points_rectangular_counter_clockwise = [v1 v4 v3 v2 v1]

            edge_vertices = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]

            edge_normals = [
                rot([-1.0, 0.0], rot_angle),
                rot([0.0, 1.0], rot_angle),
                rot([1.0, 0.0], rot_angle),
                rot([0.0, -1.0], rot_angle)
            ]

            sqrt2 = sqrt(2)
            vertex_normals = [
                rot.([[-sqrt2 / 2, -sqrt2 / 2], [-sqrt2 / 2, sqrt2 / 2]], rot_angle), # edge 1
                rot.([[-sqrt2 / 2, sqrt2 / 2], [sqrt2 / 2, sqrt2 / 2]], rot_angle),   # edge 2
                rot.([[sqrt2 / 2, sqrt2 / 2], [sqrt2 / 2, -sqrt2 / 2]], rot_angle),   # edge 3
                rot.([[sqrt2 / 2, -sqrt2 / 2], [-sqrt2 / 2, -sqrt2 / 2]], rot_angle) # edge 4
            ]

            geometry_clockwise = TrixiParticles.Polygon(points_rectangular_clockwise)

            for edge in eachindex(edge_vertices)
                @test all(isapprox.(geometry_clockwise.edge_vertices[edge],
                                    edge_vertices[edge], atol=1e-14))
            end
            for vertex in eachindex(vertex_normals)
                @test all(isapprox.(geometry_clockwise.vertex_normals[vertex],
                                    vertex_normals[vertex], atol=1e-14))
            end
            @test all(isapprox(geometry_clockwise.edge_normals, edge_normals, atol=1e-14))

            geometry_cclockwise = TrixiParticles.Polygon(Matrix(points_rectangular_counter_clockwise))

            for edge in eachindex(edge_vertices)
                @test all(isapprox.(geometry_cclockwise.edge_vertices[edge],
                                    edge_vertices[edge], atol=1e-14))
            end
            for vertex in eachindex(vertex_normals)
                @test all(isapprox.(geometry_cclockwise.vertex_normals[vertex],
                                    vertex_normals[vertex], atol=1e-14))
            end
            @test all(isapprox(geometry_cclockwise.edge_normals, edge_normals, atol=1e-14))
        end
    end

    @testset verbose=true "Real World Data" begin
        data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
        validation_dir = pkgdir(TrixiParticles, "test", "preprocessing", "data")

        @testset verbose=true "2D" begin
            files = ["hexagon", "circle", "inverted_open_curve"]
            n_edges = [6, 63, 240]
            volumes = [2.5980750000000006, 3.1363805763454, 2.6153740535469048]

            @testset "Test File `$(files[i])`" for i in eachindex(files)
                # Checked in ParaView with `trixi2vtk(geometry)`
                data = TrixiParticles.CSV.read(joinpath(validation_dir,
                                                        "normals_" * files[i] * ".csv"),
                                               TrixiParticles.DataFrame)

                vertex_normals = vcat((data.var"vertex_normals:0")',
                                      (data.var"vertex_normals:1")')

                points = vcat((data.var"Points:0")', (data.var"Points:1")')

                geometry = load_geometry(joinpath(data_dir, files[i] * ".asc"))

                @test TrixiParticles.nfaces(geometry) == n_edges[i]

                @testset "Normals $j" for j in eachindex(geometry.vertex_normals)
                    @test isapprox(geometry.vertex_normals[j][1], vertex_normals[:, j],
                                   atol=1e-4)
                end
                @testset "Points $j" for j in eachindex(geometry.vertices)[1:(end - 1)]
                    @test isapprox(geometry.vertices[j], points[:, j], atol=1e-4)
                end

                @test isapprox(TrixiParticles.volume(geometry), volumes[i])
            end
        end
        @testset verbose=true "3D" begin
            files = ["sphere", "bar", "gear"]
            n_faces = [3072, 12, 12406]
            n_vertices = [1538, 8, 6203]

            volumes = [1.3407718028525832, 24.727223299770113, 0.39679856862253504]
            @testset "Test File `$(files[i])`" for i in eachindex(files)
                # Checked in ParaView with `trixi2vtk(geometry)`
                data = TrixiParticles.CSV.read(joinpath(validation_dir,
                                                        "normals_" * files[i] * ".csv"),
                                               TrixiParticles.DataFrame)

                vertex_normals = reinterpret(reshape, SVector{3, Float64},
                                             vcat((data.var"vertex_normals:0")',
                                                  (data.var"vertex_normals:1")',
                                                  (data.var"vertex_normals:2")'))

                points = reinterpret(reshape, SVector{3, Float64},
                                     vcat((data.var"Points:0")',
                                          (data.var"Points:1")',
                                          (data.var"Points:2")'))

                geometry = load_geometry(joinpath(data_dir, files[i] * ".stl"))

                @test TrixiParticles.nfaces(geometry) == n_faces[i]
                @test length(geometry.vertices) == n_vertices[i]

                @testset "Normals $j" for j in eachindex(geometry.vertices)
                    @test isapprox(geometry.vertex_normals[j], vertex_normals[j], atol=1e-5)
                end

                @testset "Points $j" for j in eachindex(geometry.vertices)
                    @test isapprox(geometry.vertices[j], points[j], atol=1e-4)
                end

                @test isapprox(TrixiParticles.volume(geometry), volumes[i])
            end
        end
    end

    @testset verbose=true "Boundary Face" begin
        file = pkgdir(TrixiParticles, "test", "preprocessing", "data")
        planar_geometry = load_geometry(joinpath(file, "inflow_geometry.stl"))

        face, face_normal = planar_geometry_to_face(planar_geometry)

        expected_face = ([-0.10239515072676975, 0.2644994251485518, -0.36036119092034713],
                         [0.3064669575380171, 0.2392044626289733, -0.10866880239395837],
                         [-0.02275190052262935, 0.299506937268509, -0.034649329562556])
        ecpected_normal = [0.14372397390844055, 0.979596249614303, -0.14047991694743392]

        @test any(isapprox.(face, expected_face))
        @test isapprox(face_normal, ecpected_normal)
    end

    @testset verbose=true "Show" begin
        data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
        geometry = load_geometry(joinpath(data_dir, "circle.asc"))

        show_compact = "Polygon{2, Float64}()"
        @test repr(geometry) == show_compact

        show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ Polygon{2, Float64}                                                                              │
            │ ═══════════════════                                                                              │
            │ #edges: …………………………………………………………… 63                                                               │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", geometry) == show_box

        geometry = load_geometry(joinpath(data_dir, "sphere.stl"))

        show_compact = "TriangleMesh{3, Float64}()"
        @test repr(geometry) == show_compact

        show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ TriangleMesh{3, Float64}                                                                         │
            │ ════════════════════════                                                                         │
            │ #faces: …………………………………………………………… 3072                                                             │
            │ #vertices: …………………………………………………… 1538                                                             │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", geometry) == show_box
    end

    @testset verbose=true "Unique Sort" begin
        # Fixed seed to ensure reproducibility
        Random.seed!(1)

        x = rand(SVector{3, Float64}, 10_000)

        # Make sure that `x` is not unique
        for i in eachindex(x)[2:end]
            # Create duplicate values
            if rand(Bool)
                x[i] = x[i - 1]
            end
        end

        @test TrixiParticles.unique_sorted(copy(x)) == sort(unique(x))
    end
end
