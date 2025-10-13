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

    @testset verbose=true "OrientedBoundingBox" begin
        @testset verbose=true "2D" begin
            @testset verbose=true "Manual Construction" begin
                # Create a 2D oriented bounding box and test its spanning vectors
                orientation_matrix = [cos(pi / 4) -sin(pi / 4);
                                      sin(pi / 4) cos(pi / 4)]
                box_2d = OrientedBoundingBox(; box_origin=[0.1, -2.0],
                                             orientation_matrix,
                                             edge_lengths=(2.0, 1.0))
                expected = [0.1+sqrt(2) 0.1-sqrt(2) / 2;
                            -2.0+sqrt(2) -2.0+sqrt(2) / 2]
                @test isapprox(stack(box_2d.spanning_vectors) .+ box_2d.box_origin,
                               expected)
            end

            @testset verbose=true "From Geometry" begin
                # Load 2D geometry from file and test oriented bounding box calculation
                data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
                geometry = load_geometry(joinpath(data_dir, "potato.asc"))

                box_1 = OrientedBoundingBox(geometry)
                expected = [1.925952836831006 0.07595980222547905;
                            -1.026422999553135 0.14290580311134243]
                @test isapprox(stack(box_1.spanning_vectors) .+ box_1.box_origin, expected)

                # Test oriented bounding box with custom local axis scale
                box_2 = OrientedBoundingBox(geometry, local_axis_scale=(1.5, 2))
                expected = [2.8819385332081096 -0.8800258941516246;
                            -1.3869998124508005 0.503482616009008]
                @test isapprox(stack(box_2.spanning_vectors) .+ box_2.box_origin, expected)

                box_3 = OrientedBoundingBox(geometry, local_axis_scale=(3, 0.5))
                expected = [1.3085086828079229 0.6934039562485623;
                            -1.8545287410598816 0.9710115446180893]
                @test isapprox(stack(box_3.spanning_vectors) .+ box_3.box_origin, expected)
            end

            @testset verbose=true "From Point-Cloud" begin
                # Create a point cloud from a spherical shape (quarter circle sector)
                shape = SphereShape(0.1, 1.5, (0.2, 0.4), 1.0, n_layers=4,
                                    sphere_type=RoundSphere(; start_angle=0,
                                                            end_angle=pi / 4))

                box = OrientedBoundingBox(shape.coordinates)
                expected = [1.3939339828220176 1.3264241636171004;
                            0.29393398282201755 1.4671898309959612]
                @test isapprox(stack(box.spanning_vectors) .+ box.box_origin, expected)
            end
        end

        @testset verbose=true "3D" begin
            @testset verbose=true "Manual Construction" begin
                # Create a 3D oriented bounding box and test its spanning vectors
                orientation_matrix = [1/sqrt(2) -1/sqrt(2) 0.0;
                                      1/sqrt(2) 1/sqrt(2) 0.0;
                                      0.0 0.0 1.0]
                box_3d = OrientedBoundingBox(; box_origin=[0.5, -0.2, 0.0],
                                             orientation_matrix,
                                             edge_lengths=(1.0, 2.0, 3.0))
                expected = [0.5+1 / sqrt(2) 0.5-2 / sqrt(2) 0.5;
                            -0.2+1 / sqrt(2) -0.2+2 / sqrt(2) -0.2;
                            0.0 0.0 3.0]
                @test isapprox(stack(box_3d.spanning_vectors) .+ box_3d.box_origin,
                               expected)
            end

            @testset verbose=true "From Geometry" begin
                # Load 3D geometry from file and test oriented bounding box calculation
                file = pkgdir(TrixiParticles, "test", "preprocessing", "data")
                geometry = load_geometry(joinpath(file, "inflow.stl"))
                box_1 = OrientedBoundingBox(geometry)
                expected = [-0.1095304728105212 -0.03438646676380468 0.2379628221247777;
                            0.16873336880727502 0.2999997960771134 0.20335870225239414;
                            -0.36117732916596945 -0.04334770439536223 -0.42937034485453873]
                @test isapprox(stack(box_1.spanning_vectors) .+ box_1.box_origin, expected)

                # Test oriented bounding box with custom local axis scale
                box_2 = OrientedBoundingBox(geometry, local_axis_scale=(1.5, 1.5, 0.5))
                expected = [-0.044939194583412626 0.06777681448666216 0.1427916288643849;
                            0.12137652133720928 0.3182761622419668 0.23147548411531255;
                            -0.4543422598660782 0.022402177289832625 -0.5018016853691041]
                @test isapprox(stack(box_2.spanning_vectors) .+ box_2.box_origin, expected)
            end
        end

        @testset verbose=true "`is_in_oriented_box`" begin
            @testset verbose=true "Manual Construction 2D" begin
                # Simple axis-aligned box
                orientation_matrix = [1.0 0.0;
                                      0.0 1.0]
                box_2d = OrientedBoundingBox(; box_origin=[0.0, 0.0],
                                             orientation_matrix,
                                             edge_lengths=(2.0, 1.0))
                # Test points
                query_points = Dict(
                    "Inside center" => ([1.0, 0.5], [1]),
                    "On corner" => ([0.0, 0.0], [1]),
                    "On edge" => ([1.0, 0.0], [1]),
                    "Just outside left" => ([-eps(), 0.5], []),
                    "Just outside right" => ([2.0 + eps(2.0), 0.5], []),
                    "Just outside bottom" => ([1.0, -eps()], []),
                    "Just outside top" => ([1.0, 1.0 + eps()], []),
                    "Far outside" => ([-5.0, -5.0], [])
                )
                @testset "$k" for k in keys(query_points)
                    (point, expected) = query_points[k]
                    @test expected == TrixiParticles.is_in_oriented_box(point, box_2d)
                end
            end

            @testset verbose=true "Rotated Box 2D" begin
                # 45° rotated box
                orientation_matrix = [1/sqrt(2) -1/sqrt(2);
                                      1/sqrt(2) 1/sqrt(2)]
                box_rotated = OrientedBoundingBox(; box_origin=[0.0, 0.0],
                                                  orientation_matrix,
                                                  edge_lengths=(sqrt(2), 1.0))
                v1_normalized = [1 / sqrt(2), 1 / sqrt(2)]  # First edge direction
                v2_normalized = [-1 / sqrt(2), 1 / sqrt(2)] # Second edge direction (perpendicular)

                # Test points
                query_points = Dict(
                    "Inside center" => ([0.5, 0.5], [1]),
                    "Origin" => ([0.0, 0.0], [1]),
                    "Just outside along v1" => ([0.0, 0.0] +
                                                (sqrt(2) + eps()) * v1_normalized, []),
                    "Just outside along v2 positive" => ([0.0, 0.0] +
                                                         (1.0 + eps()) * v2_normalized, []),
                    "Just outside along v2 negative" => ([0.0, 0.0] -
                                                         eps() * v2_normalized, []),
                    "Just inside along v1" => ([0.0, 0.0] + sqrt(2) * v1_normalized, [1]),
                    "Just inside along v1 negative" => ([0.0, 0.0] +
                                                        eps() * v1_normalized, [1]),
                    "Just inside along v2 positive" => ([0.0, 0.0] + v2_normalized, [1]),
                    "Just inside along v2 negative" => ([0.0, 0.0] +
                                                        eps() * v2_normalized, [1]),
                    "Far outside rotated" => ([2.0, 2.0], [])
                )

                @testset "$k" for k in keys(query_points)
                    (point, expected) = query_points[k]
                    @test expected == TrixiParticles.is_in_oriented_box(point, box_rotated)
                end
            end

            @testset verbose=true "Manual Construction 3D" begin
                # Simple axis-aligned box
                box_3d = OrientedBoundingBox(; box_origin=[0.0, 0.0, 0.0],
                                             orientation_matrix=I(3),
                                             edge_lengths=(2.0, 1.0, 0.5))

                # Test points
                query_points = Dict(
                    "Inside center" => ([1.0, 0.5, 0.25], [1]),
                    "On corner origin" => ([0.0, 0.0, 0.0], [1]),
                    "On corner opposite" => ([2.0, 1.0, 0.5], [1]),
                    "On face" => ([1.0, 0.0, 0.25], [1]),
                    "Just outside x-negative" => ([-eps(), 0.5, 0.25], []),
                    "Just outside x-positive" => ([2.0 + eps(2.0), 0.5, 0.25], []),
                    "Just outside y-negative" => ([1.0, -eps(), 0.25], []),
                    "Just outside y-positive" => ([1.0, 1.0 + eps(), 0.25], []),
                    "Just outside z-negative" => ([1.0, 0.5, -eps()], []),
                    "Just outside z-positive" => ([1.0, 0.5, 0.5 + eps()], []),
                    "Far outside" => ([-5.0, -5.0, -5.0], [])
                )

                @testset "$k" for k in keys(query_points)
                    (point, expected) = query_points[k]
                    @test expected == TrixiParticles.is_in_oriented_box(point, box_3d)
                end
            end

            @testset verbose=true "Rotated Box 3D" begin
                # Box oriented along space diagonal
                orientation_matrix = [1/sqrt(3) -1/sqrt(2) -1/sqrt(6);
                                      1/sqrt(3) 1/sqrt(2) -1/sqrt(6);
                                      1/sqrt(3) 0.0 2/sqrt(6)]
                box_rotated = OrientedBoundingBox(; box_origin=[0.0, 0.0, 0.0],
                                                  orientation_matrix,
                                                  edge_lengths=(2.0, 1.0, 0.5))

                # Test points
                query_points = Dict(
                    "Origin" => ([0.0, 0.0, 0.0], [1]),
                    "Inside diagonal center" => ([0.0, 0.0, 0.0] +
                                                 orientation_matrix * [1.0, 0.5, 0.25],
                                                 [1]),
                    "Just outside along diagonal" => ([0.0, 0.0, 0.0] +
                                                      orientation_matrix *
                                                      ([2.0, 1.0, 0.5] .+ eps()), []),
                    "Just inside along diagonal" => ([0.0, 0.0, 0.0] +
                                                     orientation_matrix * [2.0, 1.0, 0.5],
                                                     [1]),
                    "Far outside diagonal" => ([0.0, 0.0, 0.0] +
                                               orientation_matrix * [3.0, 2.0, 2.0], [])
                )

                @testset "$k" for k in keys(query_points)
                    (point, expected) = query_points[k]
                    @test expected == TrixiParticles.is_in_oriented_box(point, box_rotated)
                end
            end

            @testset verbose=true "Set Operations" begin
                shape = RectangularShape(0.1, (10, 10), (0.0, 0.0), density=1.0)
                box_1 = OrientedBoundingBox(box_origin=[0.0, 0.0],
                                            orientation_matrix=I(2),
                                            edge_lengths=(1.0, 1.0))
                box_2 = OrientedBoundingBox(box_origin=[0.0, 0.0],
                                            orientation_matrix=I(2),
                                            edge_lengths=(1.0, 0.5))

                @test nparticles(intersect(shape, box_1)) == 100
                @test nparticles(setdiff(shape, box_1)) == 0
                @test nparticles(intersect(shape, box_2)) == 50
                @test nparticles(setdiff(shape, box_2)) == 50
            end
        end

        @testset verbose=true "show" begin
            box_2d = OrientedBoundingBox(box_origin=[0.1, -2.0],
                                         orientation_matrix=I(2),
                                         edge_lengths=(2.0, 1.0))
            show_box = """
                ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
                │ OrientedBoundingBox (2D)                                                                         │
                │ ════════════════════════                                                                         │
                │ box origin: ………………………………………………… [0.1, -2.0]                                                      │
                │ edge lengths: …………………………………………… (2.0, 1.0)                                                       │
                └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
            @test repr("text/plain", box_2d) == show_box

            box_3d = OrientedBoundingBox(box_origin=[0.5, -0.2, 0.0],
                                         orientation_matrix=I(3),
                                         edge_lengths=(1.0, 2.0, 3.0))
            show_box = """
                ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
                │ OrientedBoundingBox (3D)                                                                         │
                │ ════════════════════════                                                                         │
                │ box origin: ………………………………………………… [0.5, -0.2, 0.0]                                                 │
                │ edge lengths: …………………………………………… (1.0, 2.0, 3.0)                                                  │
                └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
            @test repr("text/plain", box_3d) == show_box
        end
    end
end
