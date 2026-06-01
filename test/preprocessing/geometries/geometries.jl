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

    @testset verbose=true "Open Polygon Closure" begin
        open_square = [1.0 2.0 2.0 1.0;
                       1.0 1.0 2.0 2.0]

        geometry = TrixiParticles.Polygon(open_square)

        @test TrixiParticles.nfaces(geometry) == 4
        @test first(geometry.vertices) == last(geometry.vertices)
        @test TrixiParticles.volume(geometry) ≈ 1.0

        mktempdir() do dir
            filename = joinpath(dir, "open_square.asc")
            open(filename, "w") do io
                println(io, "# ASCII")
                for vertex in eachcol(open_square)
                    println(io, vertex[1], " ", vertex[2])
                end
            end

            geometry_from_file = load_geometry(filename)

            @test TrixiParticles.nfaces(geometry_from_file) == 4
            @test first(geometry_from_file.vertices) == last(geometry_from_file.vertices)
            @test TrixiParticles.volume(geometry_from_file) ≈ 1.0
        end
    end

    @testset verbose=true "Closed Geometry Detection" begin
        open_square = [1.0 2.0 2.0 1.0;
                       1.0 1.0 2.0 2.0]

        closed_polygon = TrixiParticles.Polygon(open_square)
        open_polygon = TrixiParticles.Polygon(open_square; close_curve=false)
        partial_polygon = deleteat!(TrixiParticles.Polygon(open_square), 2)

        @test TrixiParticles.is_closed_geometry(closed_polygon)
        @test !TrixiParticles.is_closed_geometry(open_polygon)
        @test !TrixiParticles.is_closed_geometry(partial_polygon)

        shape = RectangularShape(0.5, (2, 2), (1.0, 1.0), density=1.0)
        @test_throws ArgumentError intersect(shape, open_polygon)
        @test_throws ArgumentError setdiff(shape, open_polygon)

        file = pkgdir(TrixiParticles, "test", "preprocessing", "data")
        planar_geometry = load_geometry(joinpath(file, "inflow_geometry.stl"))
        closed_mesh = extrude_geometry(planar_geometry, 0.8)
        open_mesh = extrude_geometry(planar_geometry, 0.8; omit_top_face=true)

        @test !TrixiParticles.is_closed_geometry(planar_geometry)
        @test TrixiParticles.is_closed_geometry(closed_mesh)
        @test !TrixiParticles.is_closed_geometry(open_mesh)
    end

    @testset verbose=true "Real World Data" begin
        data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
        validation_dir = pkgdir(TrixiParticles, "test", "preprocessing", "data")

        @testset verbose=true "2D" begin
            files = ["hexagon", "circle", "inverted_open_curve"]
            close_curves = [true, true, false]
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

                geometry = load_geometry(joinpath(data_dir, files[i] * ".asc");
                                         close_curve=close_curves[i])

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

    @testset verbose=true "Multiple Patches" begin
        file = pkgdir(TrixiParticles, "test", "preprocessing", "data")
        geometries = load_geometry(joinpath(file, "cuboid.stl"))

        @test isa(geometries, Vector{TrixiParticles.TriangleMesh})
        @test length(geometries) == 6  # six solids in the file
        total_faces = sum(TrixiParticles.nfaces(m) for m in geometries)
        @test total_faces == 12

        expected_normals = [
            SVector(0.0, 0.0, 1.0),   # front
            SVector(0.0, 0.0, -1.0),  # back
            SVector(-1.0, 0.0, 0.0),  # left
            SVector(1.0, 0.0, 0.0),   # right
            SVector(0.0, 1.0, 0.0),   # top
            SVector(0.0, -1.0, 0.0)   # bottom
        ]

        # Each patch should contain exactly two triangles and their normals should
        # match the facet normals.
        @testset "patch $i" for i in eachindex(geometries)
            mesh = geometries[i]
            @test TrixiParticles.nfaces(mesh) == 2
            @test all(isapprox.(mesh.face_normals, Ref(expected_normals[i]);
                                atol=1e-12))
        end
    end

    @testset verbose=true "Union" begin
        # Build a single geometry by uniting multiple STL patches (cuboid.stl contains separate solids).
        # The union should produce a closed volume.
        file = pkgdir(TrixiParticles, "test", "preprocessing", "data")
        geometries = load_geometry(joinpath(file, "cuboid.stl"))
        geometry_union = union(geometries...)

        # Employing `winding_number_factor = sqrt(eps())` serves as
        # a quantitative criterion for geometric watertightness.
        # Thus, we can check the resulting particle count with validated reference values.
        winding_number_factor = sqrt(eps())
        ic = ComplexShape(geometry_union; particle_spacing=0.05, density=1.0,
                          point_in_geometry_algorithm=WindingNumberJacobson(;
                                                                            geometry=geometry_union,
                                                                            winding_number_factor))

        @test nparticles(ic) == 9261
    end

    @testset verbose=true "Extrude Geometry" begin
        file = pkgdir(TrixiParticles, "test", "preprocessing", "data")
        geometry = load_geometry(joinpath(file, "inflow_geometry.stl"))

        @testset verbose=true "Watertightness" begin
            geometry_extruded = extrude_geometry(geometry, 0.8)

            # Employing `winding_number_factor = sqrt(eps())` serves as
            # a quantitative criterion for geometric watertightness.
            # Thus, we can check the resulting particle count with validated reference values.
            winding_number_factor = sqrt(eps())
            ic = ComplexShape(geometry_extruded; particle_spacing=0.03, density=1.0,
                              point_in_geometry_algorithm=WindingNumberJacobson(;
                                                                                geometry=geometry_extruded,
                                                                                winding_number_factor))

            @test nparticles(ic) == 2692
        end

        @testset verbose=true "Omitting Top/Bottom Face" begin
            geometry_extruded_1 = extrude_geometry(geometry, 0.8; omit_top_face=true)
            geometry_extruded_2 = extrude_geometry(geometry, 0.8; omit_bottom_face=true)
            geometry_extruded_3 = extrude_geometry(geometry, 0.8; omit_top_face=true,
                                                   omit_bottom_face=true)

            winding_number_factor = 0.2

            @test_throws ArgumentError ComplexShape(geometry_extruded_1;
                                                    particle_spacing=0.03, density=1.0,
                                                    point_in_geometry_algorithm=WindingNumberJacobson(;
                                                                                                      geometry=geometry_extruded_1,
                                                                                                      winding_number_factor))
            @test_throws ArgumentError ComplexShape(geometry_extruded_2;
                                                    particle_spacing=0.03, density=1.0,
                                                    point_in_geometry_algorithm=WindingNumberJacobson(;
                                                                                                      geometry=geometry_extruded_2,
                                                                                                      winding_number_factor))
            @test_throws ArgumentError ComplexShape(geometry_extruded_3;
                                                    particle_spacing=0.03, density=1.0,
                                                    point_in_geometry_algorithm=WindingNumberJacobson(;
                                                                                                      geometry=geometry_extruded_3,
                                                                                                      winding_number_factor))
        end
    end

    @testset verbose=true "Boundary Face" begin
        file = pkgdir(TrixiParticles, "test", "preprocessing", "data")
        planar_geometry = load_geometry(joinpath(file, "inflow_geometry.stl"))

        face, face_normal = planar_geometry_to_face(planar_geometry)

        expected_face = ([-0.02275190052262935, 0.299506937268509, -0.034649329562556],
                         [-0.10239515072676975, 0.2644994251485518, -0.36036119092034713],
                         [0.3064669575380171, 0.2392044626289733, -0.10866880239395837])
        ecpected_normal = [0.1475390528366601, 0.9789123605574772, -0.1412898376948919]

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
end;
