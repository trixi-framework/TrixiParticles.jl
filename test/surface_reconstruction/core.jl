@trixi_testset "surface reconstruction core" begin
    # Synthetic fields isolate the numerical stages from SPH time integration. Tests
    # here compare separate algorithms or exercise degenerate topology, not the recorded
    # reconstruction values checked in `reference_values.jl`.
    using MarchingCubes: MarchingCubes
    using Random: Random, randperm
    using LinearAlgebra: norm, cross, dot

    function cube_mesh(lower, upper; reverse=false)
        vertices = vec([TrixiParticles.SVector{3, Float32}(x, y, z)
                        for x in (lower, upper), y in (lower, upper), z in (lower, upper)])
        points = Float64.(reduce(hcat, vertices))
        faces = TrixiParticles.lattice_surface_topology(points).faces
        reverse && (faces = [TrixiParticles.Face(face[1], face[3], face[2])
                  for face in faces])
        return TrixiParticles.SurfaceMesh(vertices, faces)
    end

    function spherical_mesh(radius; latitude_segments=16, longitude_segments=32)
        vertices = TrixiParticles.SVector{3, Float32}[(0, 0, radius)]
        for latitude in 1:(latitude_segments - 1)
            theta = pi * latitude / latitude_segments
            for longitude in 0:(longitude_segments - 1)
                phi = 2pi * longitude / longitude_segments
                push!(vertices,
                      TrixiParticles.SVector{3, Float32}(radius * sin(theta) * cos(phi),
                                                         radius * sin(theta) * sin(phi),
                                                         radius * cos(theta)))
            end
        end
        push!(vertices, TrixiParticles.SVector{3, Float32}(0, 0, -radius))
        faces = TrixiParticles.Face[]
        function add_outward!(first, second, third)
            a, b,
            c = TrixiParticles.SVector{3, Float64}(vertices[first]),
                TrixiParticles.SVector{3, Float64}(vertices[second]),
                TrixiParticles.SVector{3, Float64}(vertices[third])
            face = dot(cross(b - a, c - a), a + b + c) > 0 ?
                   TrixiParticles.Face(first, second, third) :
                   TrixiParticles.Face(first, third, second)
            push!(faces, face)
        end
        ring(latitude,
             longitude) = 2 + (latitude - 1) * longitude_segments +
                          mod(longitude, longitude_segments)
        for longitude in 0:(longitude_segments - 1)
            add_outward!(1, ring(1, longitude), ring(1, longitude + 1))
        end
        for latitude in 1:(latitude_segments - 2), longitude in 0:(longitude_segments - 1)
            lower_left = ring(latitude, longitude)
            lower_right = ring(latitude, longitude + 1)
            upper_left = ring(latitude + 1, longitude)
            upper_right = ring(latitude + 1, longitude + 1)
            add_outward!(lower_left, upper_left, upper_right)
            add_outward!(lower_left, upper_right, lower_right)
        end
        south = length(vertices)
        for longitude in 0:(longitude_segments - 1)
            add_outward!(south, ring(latitude_segments - 1, longitude + 1),
                         ring(latitude_segments - 1, longitude))
        end
        return TrixiParticles.SurfaceMesh(vertices, faces)
    end

    function combine_meshes(meshes...)
        vertices = TrixiParticles.SVector{3, Float32}[]
        faces = TrixiParticles.Face[]
        for mesh in meshes
            offset = Int32(length(vertices))
            append!(vertices, mesh.vertices)
            append!(faces,
                    [TrixiParticles.Face(face[1] + offset, face[2] + offset,
                                         face[3] + offset)
                     for face in mesh.faces])
        end
        return TrixiParticles.SurfaceMesh(vertices, faces)
    end

    function winding_number(points, faces, point)
        solid_angle = 0.0
        for face in faces
            a = TrixiParticles.SVector{3, Float64}(points[:, face[1]]) - point
            b = TrixiParticles.SVector{3, Float64}(points[:, face[2]]) - point
            c = TrixiParticles.SVector{3, Float64}(points[:, face[3]]) - point
            denominator = norm(a) * norm(b) * norm(c) + dot(a, b) * norm(c) +
                          dot(b, c) * norm(a) + dot(c, a) * norm(b)
            solid_angle += 2atan(dot(a, cross(b, c)), denominator)
        end
        return solid_angle / (4pi)
    end

    @testset "Julia volume-CIC surface reconstruction" begin
        @testset "CIC preserves volume and first moment" begin
            points = [1.2 3.7; 1.4 2.3; 1.6 3.1]
            volumes = [0.35, 0.65]
            field = zeros(Float32, 7, 7, 7)
            stats = TrixiParticles.deposit_volume_cic!(field, points, volumes,
                                                       TrixiParticles.SVector{3, Float64}(0,
                                                                                          0,
                                                                                          0),
                                                       1.0)
            weighted_position = zeros(3)
            for index in CartesianIndices(field)
                weighted_position .+= Float64(field[index]) .* (Tuple(index) .- 1)
            end
            expected = vec(sum(points .* reshape(volumes, 1, :); dims=2)) / sum(volumes)
            @test stats.deposited_volume ≈ 1.0 atol=1.0e-7
            @test stats.particle_volume ≈ 1.0 atol=1.0e-14
            @test weighted_position / sum(field) ≈ expected atol=2.0e-7
        end

        @testset "sparse components receive explicit volume-equivalent fallback meshes" begin
            spacing = 0.05
            origin = TrixiParticles.SVector{3, Float64}(-0.75, -0.75, -0.75)
            points = [0.0 0.1 0.6;
                      0.0 0.0 0.0;
                      0.0 0.0 0.0]
            volumes = fill(0.001, 3)
            components = TrixiParticles.particle_components(points, 0.15)
            @test length(components) == 2
            @test sort(length.(components)) == [1, 2]

            field = zeros(Float32, 31, 31, 31)
            field[16, 16, 16] = 1
            constraint = fill(1.0f6, size(field))
            fallbacks = TrixiParticles.unresolved_sparse_components(field, constraint,
                                                                    points,
                                                                    volumes, components,
                                                                    Set{Int}(), origin,
                                                                    spacing, 0.5)

            @test length(fallbacks) == 1
            fallback = only(fallbacks)
            @test fallback.particles == Int32[3]
            @test fallback.center ≈ TrixiParticles.SVector{3, Float64}(0.6, 0, 0)
            @test abs(TrixiParticles.mesh_signed_volume(fallback.mesh)) ≈ 0.001 rtol=2.0e-5
            @test fallback.maximum_particle_distance == 0
            fallback_stats = TrixiParticles.mesh_geometry_stats(fallback.mesh)
            @test fallback_stats.n_connected_regions == 1
            @test fallback_stats.n_boundary_edges == 0
            @test fallback_stats.n_nonmanifold_edges == 0
            @test fallback_stats.n_degenerate_triangles == 0

            second_mesh,
            _ = TrixiParticles.equivalent_volume_sphere_mesh(TrixiParticles.SVector{3,
                                                                                    Float64}(-0.4,
                                                                                             0,
                                                                                             0),
                                                             0.002)
            combined = TrixiParticles.combine_surface_meshes([fallback.mesh, second_mesh])
            @test TrixiParticles.mesh_geometry_stats(combined).volume ≈ 0.003 rtol=2.0e-5
        end

        @testset "sparse fallback is volume-accounted in complete reconstruction" begin
            particle_spacing = 0.1
            voxel_size = 0.05
            origin = TrixiParticles.SVector{3, Float64}(-0.2, -0.2, -0.2)
            dimensions = (49, 49, 49)
            bulk = reduce(hcat,
                          vec([TrixiParticles.SVector{3, Float64}(0.5 + x, 0.5 + y,
                                                                  0.5 + z)
                               for z in -0.1:0.1:0.1, y in -0.1:0.1:0.1,
                                   x in -0.1:0.1:0.1]))
            isolated = reshape([1.5, 0.5, 0.5], 3, 1)
            points = hcat(bulk, isolated)
            volumes = fill(particle_spacing^3, size(points, 2))
            workspace = TrixiParticles.ReconstructionWorkspace(dimensions, origin,
                                                               voxel_size)
            options = TrixiParticles.SurfaceReconstructionOptions(isovalue=0.5,
                                                                  volume_tolerance_percent=0.1,
                                                                  volume_max_iterations=20,
                                                                  minimum_isovalue=0.1,
                                                                  maximum_isovalue=0.9,
                                                                  warm_start=false,
                                                                  sparse_component_fallback=true)

            support = TrixiParticles.deposition_support(points, origin, voxel_size,
                                                        dimensions)
            mesh,
            stats,
            _ = TrixiParticles._reconstruct!(workspace, points, volumes, [],
                                             origin, voxel_size,
                                             TrixiParticles.tank_domain(SVector(2.0,
                                                                                2.0,
                                                                                2.0)),
                                             particle_spacing, 1.8,
                                             0.0, support, options;
                                             initial_isovalue=options.isovalue)

            @test stats["n_sparse_source_components"] == 2
            @test stats["n_sparse_fallback_components"] == 1
            @test stats["n_sparse_fallback_particles"] == 1
            @test stats["sparse_fallback_volume"] ≈ particle_spacing^3 rtol=2.0e-5
            @test stats["sparse_primary_target_volume"] ≈ 27particle_spacing^3
            @test abs(100 * (stats["volume"] - sum(volumes)) / sum(volumes)) <= 0.1
            @test stats["n_connected_regions"] == 2
            @test stats["n_boundary_edges"] == 0
            @test stats["n_nonmanifold_edges"] == 0
            @test !isempty(mesh.vertices)
        end

        @testset "pseudonormal sign agrees with generalized winding" begin
            points = Matrix{Float64}(hcat([TrixiParticles.SVector{3, Float64}(x, y, z)
                                           for x in (0.0, 1.0), y in (0.0, 1.0),
                                               z in (0.0, 1.0)]...))
            topology = TrixiParticles.lattice_surface_topology(points)
            # Pull one corner inward to exercise concave edge and vertex signs.
            corner = findfirst(index -> points[:, index] == [1.0, 1.0, 1.0],
                               axes(points, 2))
            points[:, corner] .= (0.58, 0.62, 0.55)
            boundary = TrixiParticles.BoundaryMesh(points, topology)
            # A small grid covers face/edge/corner Voronoi regions; concentrated
            # probes below exercise the concave corner, where pseudonormals matter.
            probes = vec([TrixiParticles.SVector{3, Float64}(x, y, z)
                          for x in range(-0.15, 1.15; length=5),
                              y in range(-0.15, 1.15; length=5),
                              z in range(-0.15, 1.15; length=5)])
            append!(probes,
                    (TrixiParticles.SVector(0.54, 0.60, 0.51),
                     TrixiParticles.SVector(0.61, 0.57, 0.58),
                     TrixiParticles.SVector(0.69, 0.73, 0.53)))
            for point in probes
                distance = TrixiParticles.signed_distance(boundary.bvh, point)
                abs(distance) < 1.0e-9 && continue
                inside = abs(winding_number(points, topology.faces, point)) > 0.5
                @test (distance < 0) == inside
            end
        end

        @testset "near-duplicate cleanup preserves closed topology" begin
            vertices = TrixiParticles.SVector{3, Float32}[(0, 0, 0), (1, 0, 0), (0, 1, 0),
                                                          (0, 0, 1),
                                                          (1 + 2.0e-7, 0, 0)]
            faces = TrixiParticles.Face[(1, 3, 2), (1, 5, 4), (1, 4, 3), (5, 3, 4)]
            mesh,
            cleanup = TrixiParticles.clean_near_duplicate_vertices(TrixiParticles.SurfaceMesh(vertices,
                                                                                              faces))
            stats = TrixiParticles.mesh_geometry_stats(mesh)
            @test cleanup.merged_vertices == 1
            @test cleanup.rejected_nonmanifold_merges == 0
            @test cleanup.collapsed_faces == 0
            @test length(mesh.vertices) == 4
            @test stats.n_boundary_edges == 0
            @test stats.n_nonmanifold_edges == 0
            @test stats.n_degenerate_triangles == 0
            @test stats.volume ≈ 1 / 6 atol=1.0e-7
        end

        @testset "near-duplicate cleanup rejects a nonmanifold tetrahedron collapse" begin
            mesh = TrixiParticles.SurfaceMesh(TrixiParticles.SVector{3, Float32}[(0, 0, 0),
                                                                                 (5.0e-7, 0,
                                                                                  0),
                                                                                 (0, 1, 0),
                                                                                 (0, 0, 1)],
                                              TrixiParticles.Face[(1, 3, 2), (1, 2, 4),
                                                                  (1, 4, 3), (2, 3, 4)])
            cleaned, cleanup = TrixiParticles.clean_near_duplicate_vertices(mesh)
            stats = TrixiParticles.mesh_geometry_stats(cleaned)
            @test cleanup.merged_vertices == 0
            @test cleanup.rejected_nonmanifold_merges == 1
            @test cleanup.collapsed_faces == 0
            @test stats.n_boundary_edges == 0
            @test stats.n_nonmanifold_edges == 0
        end

        @testset "zero-volume shell remnants are discarded" begin
            flat_shell = TrixiParticles.SurfaceMesh(TrixiParticles.SVector{3, Float32}[(2,
                                                                                        0,
                                                                                        0),
                                                                                       (3,
                                                                                        0,
                                                                                        0),
                                                                                       (2,
                                                                                        1,
                                                                                        0),
                                                                                       (3,
                                                                                        1,
                                                                                        0)],
                                                    TrixiParticles.Face[(1, 3, 2),
                                                                        (1, 2, 4),
                                                                        (1, 4, 3),
                                                                        (2, 3, 4)])
            combined = TrixiParticles.combine_surface_meshes((cube_mesh(0.0, 1.0),
                                                              flat_shell))
            pruned, analysis, removal = TrixiParticles.prune_zero_volume_shells(combined)
            @test removal.removed_components == 1
            @test removal.removed_vertices == 4
            @test removal.removed_faces == 4
            @test length(pruned.vertices) == 8
            @test length(pruned.faces) == 12
            @test analysis.liquid_component_volumes ≈ [1.0] atol=1.0e-7
        end

        @testset "degenerate final triangle receives a manifold edge collapse" begin
            mesh = TrixiParticles.SurfaceMesh(TrixiParticles.SVector{3, Float32}[(0, 0, 0),
                                                                                 (2, 0, 0),
                                                                                 (1, 0, 0),
                                                                                 (0, 0, 1),
                                                                                 (2, 0, 1),
                                                                                 (1, 1, 1)],
                                              TrixiParticles.Face[(1, 2, 3), (1, 4, 2),
                                                                  (2, 4, 5), (2, 5, 3),
                                                                  (3, 5, 6), (3, 6, 1),
                                                                  (1, 6, 4), (4, 6, 5)])
            before = TrixiParticles.mesh_geometry_stats(mesh)
            cleaned, cleanup = TrixiParticles.collapse_degenerate_triangles(mesh)
            after = TrixiParticles.mesh_geometry_stats(cleaned)
            @test before.n_degenerate_triangles == 1
            @test cleanup.collapsed_edges == 1
            @test cleanup.removed_vertices == 1
            @test cleanup.removed_faces == 2
            @test cleanup.maximum_edge_length == 1
            @test after.n_boundary_edges == 0
            @test after.n_nonmanifold_edges == 0
            @test after.n_degenerate_triangles == 0
        end

        @testset "box-restricted marching cubes matches MarchingCubes.jl bitwise" begin
            function library_mesh(scalar, origin, spacing)
                dimensions = size(scalar)
                coordinates = ntuple(axis -> Float32.(range(origin[axis]; step=spacing,
                                                            length=dimensions[axis])), 3)
                mc = MarchingCubes.MC(copy(scalar), Int32; normal_sign=-1, x=coordinates[1],
                                      y=coordinates[2], z=coordinates[3])
                TrixiParticles.march_surface!(mc, 0.0f0)
                return TrixiParticles.surface_mesh(mc)
            end

            function port_mesh(scalar, origin, spacing, backend)
                workspace = TrixiParticles.ReconstructionWorkspace(size(scalar), origin,
                                                                   spacing;
                                                                   backend)
                workspace.field .= scalar
                workspace.constraint .= Inf32
                return TrixiParticles.contour!(workspace, 0.0; allow_empty=true)
            end

            x = range(-1, 1, length=31)
            blobs = ((TrixiParticles.SVector(0.3, -0.2, 0.1), 0.35),
                     (TrixiParticles.SVector(-0.4, 0.3, -0.2), 0.3),
                     (TrixiParticles.SVector(0.0, 0.0, 0.6), 0.25))
            blob_field = Float32[maximum(radius -
                                         norm(TrixiParticles.SVector(xi, yj, zk) - center)
                                         for (center, radius) in blobs)
                                 for xi in x, yj in x, zk in x]
            # A surface touching the grid boundary, and exact zeros / eps values (clamping path)
            slab_field = Float32[0.3 - abs(z - 0.5)
                                 for x in range(0, 1, 13), y in range(0, 1, 17),
                                     z in range(0, 1, 21)]
            clamping_field = zeros(Float32, 12, 12, 12)
            clamping_field[4:8, 4:8, 4:8] .= 1
            clamping_field[3, 3, 3] = eps(Float32) / 2
            clamping_field[10, 4, 7] = -eps(Float32)

            origin = TrixiParticles.SVector(-0.3, 0.1, 2.5)
            for scalar in (blob_field, slab_field, clamping_field),
                backend in (SerialBackend(), PolyesterBackend())
                expected = library_mesh(scalar, origin, 0.0137)
                mesh = port_mesh(scalar, origin, 0.0137, backend)
                @test !isempty(mesh.faces)
                @test mesh.vertices == expected.vertices
                @test mesh.faces == expected.faces
            end

            # Without any positive value there is no box and no surface
            workspace = TrixiParticles.ReconstructionWorkspace((5, 5, 5), origin, 0.1)
            workspace.field .= -1
            workspace.constraint .= Inf32
            @test TrixiParticles.contour!(workspace, 0.0; allow_empty=true) === nothing
        end

        @testset "grid-node candidates give identical vertex merges" begin
            Random.seed!(3)
            for (n, spacing) in ((40, 0.01), (33, 1.0e-5), (50, 0.2))
                workspace = TrixiParticles.ReconstructionWorkspace((n, n, n),
                                                                   TrixiParticles.SVector(0.37,
                                                                                          -1.2,
                                                                                          2.9),
                                                                   spacing)
                x = range(-1, 1, length=n)
                scalar = Float32[0.55 - sqrt(xi^2 + 0.8yj^2 + 1.3zk^2)
                                 for xi in x, yj in x, zk in x]
                # Values of magnitude ~eps at nodes near the surface put vertices within the
                # cleanup tolerance of grid nodes, so that real (and rejected) merges happen
                near = findall(value -> abs(value) < 0.08, scalar)
                for index in near[randperm(length(near))[1:min(200, end)]]
                    scalar[index] = rand((-1, 1)) * Float32(2 * eps(Float32))
                end
                workspace.field .= scalar
                workspace.constraint .= Inf32
                box, _ = TrixiParticles.scalar_and_active_box!(workspace, 0.0f0)
                mesh = TrixiParticles.marching_cubes_box!(workspace, box)

                full, full_stats = TrixiParticles.clean_near_duplicate_vertices(mesh)
                candidates = TrixiParticles.grid_node_candidates(mesh, workspace,
                                                                 TrixiParticles.MESH_CLEANUP_TOLERANCE_M)
                fast,
                fast_stats = TrixiParticles.clean_near_duplicate_vertices(mesh;
                                                                          candidates)
                @test full_stats.merged_vertices > 0
                @test count(candidates) < length(candidates)
                @test fast.vertices == full.vertices
                @test fast.faces == full.faces
                @test fast_stats == full_stats
            end
        end

        @testset "empty candidates skip the vertex cleanup" begin
            mesh = cube_mesh(0.0, 1.0)
            full, full_stats = TrixiParticles.clean_near_duplicate_vertices(mesh)
            fast,
            fast_stats = TrixiParticles.clean_near_duplicate_vertices(mesh;
                                                                      candidates=falses(length(mesh.vertices)))
            @test fast.vertices == full.vertices
            @test fast.faces == full.faces
            @test fast_stats == full_stats
            @test fast_stats.merged_vertices == 0
            @test fast_stats.collapsed_faces == 0
            # Degenerate faces still collapse even without candidates
            degenerate = TrixiParticles.SurfaceMesh(mesh.vertices,
                                                    [mesh.faces...,
                                                        TrixiParticles.Face(1, 1, 2)])
            collapsed,
            collapsed_stats = TrixiParticles.clean_near_duplicate_vertices(degenerate;
                                                                           candidates=falses(length(mesh.vertices)))
            @test collapsed_stats.collapsed_faces == 1
            @test length(collapsed.faces) == length(mesh.faces)
        end

        @testset "restricted Gaussian filter matches the full grid bitwise" begin
            points = [0.31 0.47 0.52 0.9; 0.22 0.35 0.61 0.4; 0.44 0.28 0.5 0.33]
            volumes = [1.0e-3, 2.0e-3, 1.5e-3, 1.2e-3]
            origin = TrixiParticles.SVector(0.0, 0.0, 0.0)
            spacing = 0.02
            dimensions = (70, 60, 50)
            kernel = TrixiParticles.gaussian_kernel(1.8)

            full = zeros(Float32, dimensions)
            TrixiParticles.deposit_volume_cic!(full, points, volumes, origin, spacing)
            restricted = copy(full)
            # Stale values in the buffers must not leak into the restricted result
            temporary, scratch = fill(7.0f0, dimensions), fill(7.0f0, dimensions)
            support = TrixiParticles.deposition_support(points, origin, spacing, dimensions)
            zeroed = copy(restricted)
            TrixiParticles.gaussian_filter!(full, similar(full), similar(full), kernel)
            TrixiParticles.gaussian_filter!(restricted, temporary, scratch, kernel; support)
            @test restricted == full
            @test all(axis -> length(support[axis]) < dimensions[axis], 1:3)

            # With zeroed buffers, the outputs are not zeroed outside their regions
            TrixiParticles.gaussian_filter!(zeroed, zeros(Float32, dimensions),
                                            zeros(Float32, dimensions), kernel;
                                            support, zeroed=true)
            @test zeroed == full
        end

        @testset "fused exclusion and support match the separate passes" begin
            grid = TrixiParticles.ReconstructionGrid(TrixiParticles.SVector{3, Float64}(0,
                                                                                        0,
                                                                                        0),
                                                     0.05, (30, 30, 30))
            inside = reduce(hcat,
                            vec([TrixiParticles.SVector{3, Float64}(x, y, z)
                                 for z in 0.1:0.1:0.3, y in 0.1:0.1:0.3,
                                     x in 0.1:0.1:0.3]))
            outside = Float64[-1.0 0.5 0.5; 0.5 -2.0 0.5; 0.5 0.5 5.0]
            function check_fused(points)
                volumes = fill(0.05^3, size(points, 2))
                separate_points, separate_volumes, separate_count,
                separate_excluded = TrixiParticles.exclude_outside_grid(points, volumes,
                                                                        grid)
                separate_support = TrixiParticles.deposition_support(separate_points,
                                                                     grid.origin,
                                                                     grid.spacing,
                                                                     grid.dimensions)
                fused = TrixiParticles.exclude_and_support(points, volumes, grid)
                @test fused[1] == separate_points
                @test fused[2] == separate_volumes
                @test fused[3] == separate_count
                @test fused[4] == separate_excluded
                @test fused[5] == separate_support
                return fused
            end
            # Nothing excluded: the input arrays are returned unchanged
            fused = check_fused(inside)
            @test fused[1] === inside
            @test fused[3] == 0
            @test all(axis -> length(fused[5][axis]) < grid.dimensions[axis], 1:3)
            # Some particles outside the grid
            fused = check_fused(hcat(inside, outside))
            @test fused[3] == 3
            # All particles outside the grid
            fused = check_fused(outside)
            @test fused[3] == 3 && isempty(fused[1])

            # Dimension-generic helpers must be inferred: a captured, reassigned tuple
            # once introduced per-particle boxing. Binary-exact coordinates also test
            # the last valid stencil versus the first excluded one in Float32/Float64.
            for dimension in (2, 3), scalar_type in (Float32, Float64)
                grid_case = TrixiParticles.ReconstructionGrid(SVector(ntuple(_ -> -0.25,
                                                                             dimension)),
                                                              0.125,
                                                              ntuple(_ -> 12, dimension))
                coords = fill(scalar_type(-0.1875), dimension, 5)
                coords[1, :] .= scalar_type.(-0.25 .+ 0.125 .* [-1, 0, 4.5, 10, 11])
                measures = collect(1.0:5.0)
                fused_case = @inferred TrixiParticles.exclude_and_support(coords, measures,
                                                                          grid_case)
                separate = TrixiParticles.exclude_outside_grid(coords, measures,
                                                               grid_case)
                @test fused_case[1:4] == separate
                @test fused_case[1] == coords[:, 2:4]
                @test fused_case[4] == 6.0
                @test fused_case[5] ==
                      TrixiParticles.deposition_support(separate[1], grid_case.origin,
                                                        grid_case.spacing,
                                                        grid_case.dimensions)
            end
        end

        @testset "parallel field reductions are deterministic" begin
            field = rand(Float32, 40, 30, 20)
            serial = TrixiParticles.field_sum_extrema(field; backend=SerialBackend())
            threaded = TrixiParticles.field_sum_extrema(field; backend=PolyesterBackend())
            @test serial == threaded
            @test serial[1] ≈ sum(Float64, field)
            @test serial[2] == minimum(field)
            @test serial[3] == maximum(field)

            # Restricted to a region outside of which the field is zero, the results are
            # bitwise identical, including the minimum of the zeros outside
            region = (5:31, 3:17, 8:12)
            masked = zeros(Float32, size(field))
            masked[region...] .= 0.5f0 .+ field[region...]
            restricted = TrixiParticles.field_sum_extrema(masked; region)
            @test restricted == TrixiParticles.field_sum_extrema(masked)
            @test restricted[2] == 0
            @test TrixiParticles.field_sum_extrema(field; region=axes(field)) == serial
        end

        @testset "boundary and nonmanifold edge counts" begin
            function reference_edge_counts(faces)
                uses = Dict{Tuple{Int32, Int32}, Int}()
                for face in faces
                    for (a, b) in ((face[1], face[2]), (face[2], face[3]),
                         (face[3], face[1]))
                        uses[minmax(a, b)] = get(uses, minmax(a, b), 0) + 1
                    end
                end
                return count(==(1), values(uses)), count(>(2), values(uses))
            end
            edge_counts(mesh) = TrixiParticles.count_boundary_and_nonmanifold_edges(mesh)
            Face = TrixiParticles.Face

            closed = cube_mesh(0.0, 1.0)
            @test edge_counts(closed) == (0, 0)
            open = TrixiParticles.SurfaceMesh(closed.vertices, closed.faces[2:end])
            @test edge_counts(open) == (3, 0)
            # Three triangles sharing the edge (1, 2)
            book = TrixiParticles.SurfaceMesh(fill(TrixiParticles.SVector{3, Float32}(0, 0,
                                                                                      0),
                                                   5),
                                              [Face(1, 2, 3), Face(1, 2, 4), Face(2, 1, 5)])
            @test edge_counts(book) == (6, 1)

            # A closed sphere with removed and duplicated faces
            sphere = spherical_mesh(1.0)
            faces = [face for (index, face) in enumerate(sphere.faces) if index % 7 != 0]
            append!(faces, sphere.faces[1:11:end])
            damaged = TrixiParticles.SurfaceMesh(sphere.vertices, faces)
            @test edge_counts(damaged) == reference_edge_counts(faces)
            @test all(>(0), edge_counts(damaged))
        end

        @testset "vertex winding flags" begin
            # Winding flags resolve the component roots once per vertex: uniform within
            # each component, distinguishing opposite windings
            two = combine_meshes(cube_mesh(0.0, 1.0), cube_mesh(2.0, 3.0; reverse=true))
            parent, reverse_winding = TrixiParticles.reflected_component_winding(two)
            flags = TrixiParticles.vertex_reverse_flags(parent, reverse_winding,
                                                        length(two.vertices))
            @test flags == [reverse_winding[TrixiParticles.find_root!(parent,
                                                             Int32(vertex))]
                   for vertex in 1:length(two.vertices)]
            @test all(==(flags[1]), flags[1:8]) && all(==(flags[9]), flags[9:16])
            @test flags[1] != flags[9]
        end

        @testset "geometry statistics reuse a precomputed analysis" begin
            mesh = cube_mesh(0.0, 1.0)
            analysis = TrixiParticles.mesh_liquid_analysis(mesh)
            @test TrixiParticles.mesh_geometry_stats(mesh; analysis) ==
                  TrixiParticles.mesh_geometry_stats(mesh)
        end

        @testset "parallel reductions agree with serial within tolerance" begin
            # Big enough to engage the parallel path (see `PARALLEL_REDUCTION_MIN_FACES`)
            n = 180
            spacing = 0.01
            workspace = TrixiParticles.ReconstructionWorkspace((n, n, n),
                                                               TrixiParticles.SVector(0.37,
                                                                                      -1.2,
                                                                                      2.9),
                                                               spacing)
            x = range(-1, 1, length=n)
            workspace.field .= Float32[0.55 - sqrt(xi^2 + 0.8yj^2 + 1.3zk^2)
                                       for xi in x, yj in x, zk in x]
            workspace.constraint .= Inf32
            box, _ = TrixiParticles.scalar_and_active_box!(workspace, 0.0f0)
            mesh = TrixiParticles.marching_cubes_box!(workspace, box)
            @test length(mesh.faces) > TrixiParticles.PARALLEL_REDUCTION_MIN_FACES

            serial_stats = TrixiParticles.mesh_geometry_stats(mesh)
            parallel_stats = TrixiParticles.mesh_geometry_stats(mesh;
                                                                backend=PolyesterBackend())
            # Everything but the float sums is exactly preserved: same topology,
            # components, integer counts, and bounds
            @test length(parallel_stats.shell_volumes) ==
                  length(serial_stats.shell_volumes)
            @test parallel_stats.n_connected_regions == serial_stats.n_connected_regions
            @test parallel_stats.n_surface_components ==
                  serial_stats.n_surface_components
            @test parallel_stats.n_cavity_regions == serial_stats.n_cavity_regions
            @test parallel_stats.n_boundary_edges == serial_stats.n_boundary_edges
            @test parallel_stats.n_nonmanifold_edges ==
                  serial_stats.n_nonmanifold_edges
            @test parallel_stats.n_degenerate_triangles ==
                  serial_stats.n_degenerate_triangles
            @test parallel_stats.lower == serial_stats.lower
            @test parallel_stats.upper == serial_stats.upper
            # Only the float sums reassociate, far below any tolerance
            @test isapprox(parallel_stats.volume, serial_stats.volume; rtol=1.0e-9)
            @test isapprox(parallel_stats.surface_area, serial_stats.surface_area;
                           rtol=1.0e-9)
            # The parallel path is deterministic: same threads, same bits
            repeat_stats = TrixiParticles.mesh_geometry_stats(mesh;
                                                              backend=PolyesterBackend())
            @test repeat_stats.volume == parallel_stats.volume
            @test repeat_stats.surface_area == parallel_stats.surface_area
        end

        @testset "nested shells define liquid domains and cavity winding" begin
            outer = cube_mesh(0.0, 1.0)
            cavity = cube_mesh(0.25, 0.75)
            mesh = combine_meshes(outer, cavity)
            analysis = TrixiParticles.mesh_liquid_analysis(mesh)
            stats = TrixiParticles.mesh_geometry_stats(mesh)
            @test analysis.shell_depths == Int32[0, 1]
            @test analysis.shell_parents == Int32[0, 1]
            @test stats.volume ≈ 0.875 atol=1.0e-7
            @test stats.region_volumes ≈ [0.875] atol=1.0e-7
            @test stats.n_connected_regions == 1
            @test stats.n_surface_components == 2
            @test stats.n_cavity_regions == 1
            @test stats.cavity_volume ≈ 0.125 atol=1.0e-7
            volume, centroid = TrixiParticles.mesh_volume_centroid(mesh)
            @test volume ≈ 0.875 atol=1.0e-7
            @test centroid ≈ TrixiParticles.SVector{3, Float64}(0.5, 0.5, 0.5) atol=1.0e-7

            parent, reversals = TrixiParticles.reflected_component_winding(mesh)
            @test length(parent) == length(mesh.vertices)
            for (shell_index, shell) in enumerate(analysis.shells)
                output_signed_volume = reversals[shell.root] ?
                                       -shell.signed_volume : shell.signed_volume
                @test sign(output_signed_volume) ==
                      (iseven(analysis.shell_depths[shell_index]) ? 1 : -1)
            end
        end

        @testset "mesh volume centroid is translation and winding invariant" begin
            first = cube_mesh(0.0, 1.0)
            second = cube_mesh(2.0, 3.0; reverse=true)
            mesh = combine_meshes(first, second)
            volume, centroid = TrixiParticles.mesh_volume_centroid(mesh)
            @test volume ≈ 2.0 atol=2.0e-7
            @test centroid ≈ TrixiParticles.SVector{3, Float64}(1.5, 1.5, 1.5) atol=2.0e-7

            tiny = cube_mesh(1.0, 1.0001)
            tiny_stats = TrixiParticles.mesh_geometry_stats(tiny)
            expected_tiny_volume = Float64(Float32(1.0001) - Float32(1.0))^3
            @test tiny_stats.n_connected_regions == 1
            @test tiny_stats.volume > eps(Float64)
            @test tiny_stats.volume ≈ expected_tiny_volume rtol=2.0e-6
        end

        @testset "disconnected liquid components ignore input shell winding" begin
            mesh = combine_meshes(cube_mesh(0.0, 1.0), cube_mesh(2.0, 3.0; reverse=true))
            stats = TrixiParticles.mesh_geometry_stats(mesh)
            @test stats.volume ≈ 2.0 atol=2.0e-7
            @test stats.region_volumes ≈ [1.0, 1.0] atol=2.0e-7
            @test stats.n_connected_regions == 2
            @test stats.n_surface_components == 2
            @test stats.n_cavity_regions == 0
        end

        @testset "three nesting levels separate liquid islands from cavities" begin
            mesh = combine_meshes(cube_mesh(0.0, 1.0), cube_mesh(0.1, 0.9),
                                  cube_mesh(0.3, 0.7))
            analysis = TrixiParticles.mesh_liquid_analysis(mesh)
            stats = TrixiParticles.mesh_geometry_stats(mesh)
            @test analysis.shell_depths == Int32[0, 1, 2]
            @test stats.region_volumes ≈ [0.488, 0.064] atol=2.0e-7
            @test stats.volume ≈ 0.552 atol=2.0e-7
            @test stats.cavity_volume ≈ 0.448 atol=2.0e-7
        end

        @testset "hollow spherical shells use nesting-aware volume" begin
            outer = spherical_mesh(1.0)
            inner = spherical_mesh(0.5)
            mesh = combine_meshes(outer, inner)
            stats = TrixiParticles.mesh_geometry_stats(mesh)
            polygonal_expected = abs(TrixiParticles.mesh_signed_volume(outer)) -
                                 abs(TrixiParticles.mesh_signed_volume(inner))
            @test stats.volume ≈ polygonal_expected atol=2.0e-6
            @test stats.volume ≈ (4pi / 3) * (1 - 0.5^3) rtol=0.025
            @test stats.n_cavity_regions == 1
        end

        @testset "marching cubes clears cached edge indices" begin
            field = Array{Float32}(undef, 9, 9, 9)
            for index in CartesianIndices(field)
                x, y, z = Tuple(index) .- 5
                field[index] = 8.5f0 - Float32(x^2 + y^2 + z^2)
            end
            contaminated = MarchingCubes.MC(copy(field), Int32; normal_sign=-1)
            fill!(contaminated.vert_indices, typemax(Int32))
            TrixiParticles.march_surface!(contaminated, 0.0f0; topology_aware=true)
            fresh = MarchingCubes.MC(copy(field), Int32; normal_sign=-1)
            TrixiParticles.march_surface!(fresh, 0.0f0; topology_aware=true)
            @test contaminated.vertices == fresh.vertices
            @test contaminated.triangles == fresh.triangles
            @test typemax(Int32) ∉ contaminated.vert_indices
        end

        @testset "volume correction accepts an empty upper bracket" begin
            spacing = 0.1
            workspace = TrixiParticles.ReconstructionWorkspace((25, 25, 25),
                                                               TrixiParticles.SVector{3,
                                                                                      Float64}(-1.2,
                                                                                               -1.2,
                                                                                               -1.2),
                                                               spacing)
            for index in CartesianIndices(workspace.field)
                point = TrixiParticles.SVector{3, Float64}((Tuple(index) .- 1) .* spacing .-
                                                           1.2)
                workspace.field[index] = Float32(0.8 - norm(point))
            end
            fill!(workspace.constraint, 1.0f6)
            target_volume = 4pi * 0.4^3 / 3
            options = TrixiParticles.SurfaceReconstructionOptions(isovalue=0.2,
                                                                  volume_tolerance_percent=0.1,
                                                                  volume_max_iterations=20,
                                                                  minimum_isovalue=0.1,
                                                                  maximum_isovalue=0.9,
                                                                  warm_start=false)
            mesh, effective_isovalue,
            evaluations,
            best_analysis = TrixiParticles.corrected_contour!(workspace,
                                                              options.isovalue,
                                                              target_volume,
                                                              options)
            @test !isnothing(best_analysis)
            final_error = only(evaluation["volume_error_percent"]
                               for evaluation in evaluations
                               if evaluation["isovalue"] == effective_isovalue)
            @test evaluations[2]["empty_mesh"]
            @test !isempty(mesh.vertices)
            @test 0.2 < effective_isovalue < 0.8
            @test abs(final_error) <= options.volume_tolerance_percent
        end
    end
end
