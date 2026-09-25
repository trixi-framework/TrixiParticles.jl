@trixi_testset "surface reconstruction correctness" begin
    # Analytic fields, independent mesh integrals, and library comparisons complement
    # the recorded reference values. These catch agreement with the wrong algorithm
    # that comparing two reconstruction entry points alone could miss.
    using MarchingCubes: MarchingCubes
    using Random: MersenneTwister, rand, randperm
    TP = TrixiParticles

    function cube(lower, upper)
        points = reduce(hcat,
                        vec([SVector{3, Float64}(x, y, z)
                             for x in (lower, upper), y in (lower, upper),
                                 z in (lower, upper)]))
        return SurfaceMesh(SVector{3, Float32}.(eachcol(points)),
                           lattice_surface_topology(points).faces)
    end

    # Independent divergence-theorem reference, with high-precision arithmetic on the
    # actual stored coordinates. No component grouping or production volume helper.
    function signed_volume(vertices, faces)
        volume = BigFloat(0)
        for face in faces
            a, b, c = (SVector{3, BigFloat}(vertices[i]) for i in face)
            volume += dot(a, cross(b, c)) / 6
        end
        return volume
    end

    @testset "CIC partition of unity and first moment" begin
        origin = SVector(-0.25, -0.5, 0.125)
        spacing = 0.125
        dimensions = (17, 19, 21)
        rng = MersenneTwister(907)
        points = origin .+ spacing .* (2 .+ 8 .* rand(rng, 3, 8))
        volumes = spacing^3 .* (0.3 .+ rand(rng, 8))
        for p in axes(points, 2)
            field = zeros(Float32, dimensions)
            TP.deposit_volume_cic!(field, points[:, p:p], volumes[p:p], origin, spacing)
            moments = TP.scalar_field_moments(field, origin, spacing)
            @test moments.integral ≈ volumes[p] rtol=8eps(Float32)
            @test moments.centroid≈points[:, p] atol=spacing*8eps(Float32) rtol=0
            @test all(>=(0), field)
            @test count(!iszero, field) <= 8
        end
    end

    @testset "Gaussian tensor-product reference and finite-grid mass" begin
        kernel = TP.gaussian_kernel(0.7)
        r = length(kernel) ÷ 2
        n = 2r + 5
        center = r + 3
        field = zeros(Float32, n, n, n)
        field[center, center, center] = 1
        TP.gaussian_filter!(field, similar(field), similar(field), kernel)
        expected = zeros(Float64, size(field))
        for k in eachindex(kernel), j in eachindex(kernel), i in eachindex(kernel)
            expected[center + i - r - 1, center + j - r - 1,
                     center + k - r - 1] = Float64(kernel[i]) * Float64(kernel[j]) *
                                           Float64(kernel[k])
        end
        @test field ≈ expected rtol=8eps(Float32)
        @test sum(Float64, field) ≈ 1 rtol=8eps(Float32)
        @test TP.scalar_field_moments(field, SVector(0.0, 0.0, 0.0), 1.0).centroid ≈
              fill(center - 1, 3) atol=1.0e-12

        # Zero padding at the finite grid boundary loses mass. It must not be claimed
        # to be a normalized convolution there.
        edge = zeros(Float32, n, n, n)
        edge[1, 1, 1] = 1
        TP.gaussian_filter!(edge, similar(edge), similar(edge), kernel)
        @test sum(Float64, edge) < 1
    end

    @testset "all classic marching-cubes cases match the reference port" begin
        dimensions = (6, 6, 6)
        origin = SVector(-0.7, 1.2, 0.3)
        spacing = 0.08
        workspace = TP.ReconstructionWorkspace(dimensions, origin, spacing)
        fill!(workspace.constraint, Inf32)
        coordinates = ntuple(axis -> Float32.(range(origin[axis]; step=spacing,
                                                    length=dimensions[axis])), 3)
        reference = MarchingCubes.MC(zeros(Float32, dimensions), Int32;
                                     x=coordinates[1], y=coordinates[2], z=coordinates[3])
        for case in 0:255
            fill!(workspace.field, -1.0f0)
            for p in 0:7
                workspace.field[3 + ((p ⊻ (p >> 1)) & 1), 3 + ((p >> 1) & 1),
                                3 + ((p >> 2) & 1)] = (case & (1 << p)) == 0 ? -0.7f0 :
                                                      0.9f0
            end
            box, _ = TP.scalar_and_active_box!(workspace, 0.0f0)
            # Keep the reference scalar separate from the port's buffers.
            reference.vol[] .= workspace.scratch
            TP.march_surface!(reference, 0.0f0)
            if isnothing(box)
                @test isempty(reference.vertices)
            else
                actual = TP.marching_cubes_box!(workspace, box)
                @test actual.vertices == reference.vertices
                @test actual.faces == reference.triangles
            end
        end
    end

    @testset "corrected Lewiner saddle topology versus classic" begin
        # Analytic domain: (-R, R)^3 intersected with xy + mu > 0. Positive mu
        # connects the two same-sign quadrants across the saddle; negative mu leaves
        # two components. The data at grid nodes is sampled directly from this field.
        R, offset = 1.5, 0.009
        lewiner_errors = Dict(-offset => Float64[], offset => Float64[])
        for n in (12, 26, 38), mu in (-offset, offset)
            coords = Float32.(range(-2R; stop=2R, length=n))
            field = Float32[min(x * y + mu, R - abs(x), R - abs(y), R - abs(z))
                            for x in coords, y in coords, z in coords]
            classic = MarchingCubes.MC(copy(field), Int32; normal_sign=-1,
                                       x=coords, y=coords, z=coords)
            lewiner = MarchingCubes.MC(copy(field), Int32; normal_sign=-1,
                                       x=coords, y=coords, z=coords)
            TP.march_surface!(classic, 0.0f0)
            TP.march_surface!(lewiner, 0.0f0; topology_aware=true)
            classic_mesh = TP.surface_mesh(classic)
            lewiner_mesh = TP.surface_mesh(lewiner)
            classic_stats = TP.mesh_geometry_stats(classic_mesh)
            lewiner_stats = TP.mesh_geometry_stats(lewiner_mesh)
            expected = mu < 0 ? 2 : 1
            @test lewiner_stats.n_connected_regions == expected
            @test classic_stats.n_connected_regions == (n == 38 && mu > 0 ? 1 : 2)
            @test lewiner_stats.n_boundary_edges ==
                  lewiner_stats.n_nonmanifold_edges ==
                  lewiner_stats.n_degenerate_triangles == 0
            @test classic_stats.n_boundary_edges == 0

            # Integrate xy + mu > 0 analytically over the square, then multiply by
            # the z extent. The polygonal approximation improves under refinement;
            # for the connected saddle, Lewiner is closer than classic at both n.
            m = abs(mu)
            exact = 2R * (2R^2 + sign(mu) * 2m * (1 + log(R^2 / m)))
            push!(lewiner_errors[mu], abs(lewiner_stats.volume - exact))
            if mu > 0 && n < 38
                @test abs(lewiner_stats.volume - exact) < abs(classic_stats.volume - exact)
            end

            if n == 12 && mu > 0
                # The two polygonal tilings disagree by a finite volume at the
                # saddle transition even though the analytic enclosed volume is
                # continuous. Isovalue correction cannot necessarily hit a target
                # in this gap within its configured 0.1% tolerance.
                function lewiner_volume(level)
                    lewiner.vol[] .= field .- Float32(level)
                    TP.march_surface!(lewiner, 0.0f0; topology_aware=true)
                    return TP.mesh_geometry_stats(TP.surface_mesh(lewiner)).volume
                end
                above = lewiner_volume(0.00899)
                below = lewiner_volume(0.00901)
                @test above - below > 0.4
                # Shifting the isovalue moves both the hyperbola and the box walls.
                # Include the changing box extent in the analytic comparison.
                analytic_volume(level) = begin
                    halfwidth, margin = R - level, mu - level
                    2halfwidth * (2halfwidth^2 +
                     sign(margin) * 2abs(margin) *
                     (1 + log(halfwidth^2 / abs(margin))))
                end
                analytic_change = analytic_volume(0.00899) - analytic_volume(0.00901)
                @test 0 < analytic_change < 0.003
                target = (above + below) / 2
                @test 100 * min(above - target, target - below) / target > 0.1
                lewiner.vol[] .= field
                TP.march_surface!(lewiner, 0.0f0; topology_aware=true)
            end

            # Reuse across changes of sign while poisoning every old edge index.
            # The reset wrapper must reproduce the fresh calculation bit-for-bit.
            other = Float32[min(x * y - mu, R - abs(x), R - abs(y), R - abs(z))
                            for x in coords, y in coords, z in coords]
            lewiner.vol[] .= other
            fill!(lewiner.vert_indices, typemax(Int32))
            TP.march_surface!(lewiner, 0.0f0; topology_aware=true)
            lewiner.vol[] .= field
            fill!(lewiner.vert_indices, typemax(Int32))
            TP.march_surface!(lewiner, 0.0f0; topology_aware=true)
            @test lewiner.vertices == lewiner_mesh.vertices
            @test lewiner.triangles == lewiner_mesh.faces
        end
        @test all(all(diff(errors) .< 0) for errors in values(lewiner_errors))
    end

    @testset "torus genus and volume converge under grid refinement" begin
        major_radius, minor_radius = 0.9, 0.32
        exact_volume = 2pi^2 * major_radius * minor_radius^2
        errors = Float64[]
        for n in (22, 38, 62)
            xy = Float32.(range(-1.6; stop=1.6, length=n))
            z = Float32.(range(-0.7; stop=0.7, length=n))
            field = Float32[minor_radius^2 -
                            (hypot(x, y) - major_radius)^2 - zz^2
                            for x in xy, y in xy, zz in z]
            candidate_volumes = Float64[]
            for topology_aware in (false, true)
                mc = MarchingCubes.MC(copy(field), Int32; normal_sign=-1,
                                      x=xy, y=xy, z=z)
                TP.march_surface!(mc, 0.0f0; topology_aware)
                mesh = TP.surface_mesh(mc)
                geometry = TP.mesh_geometry_stats(mesh)
                # For a closed triangular manifold E = 3F/2 and chi = V - E + F.
                # A single torus has genus one, hence chi = 0.
                @test geometry.n_connected_regions == 1
                @test geometry.n_boundary_edges == geometry.n_nonmanifold_edges == 0
                @test length(mesh.vertices) - length(mesh.faces) ÷ 2 == 0
                push!(candidate_volumes, geometry.volume)
            end
            @test candidate_volumes[1] == candidate_volumes[2]
            push!(errors, abs(candidate_volumes[2] / exact_volume - 1))
        end
        @test all(diff(errors) .< 0)
    end

    @testset "independent threaded slab flags" begin
        workspace = TP.ReconstructionWorkspace((4, 4, 257), SVector(0.0, 0.0, 0.0), 0.1;
                                               backend=ThreadsDynamicBackend())
        fill!(workspace.constraint, Inf32)
        for k in (1, 32, 33, 64, 65, 128, 129, 257)
            fill!(workspace.field, 0)
            workspace.field[2, 2, k] = 1
            box, positive = TP.scalar_and_active_box!(workspace, 0.5f0)
            @test positive
            @test box == (1:3, 1:3, max(1, k - 1):min(257, k + 1))
        end
        # Adjacent slabs share a storage word in a BitVector. Repeat the competing
        # writes to catch accidental reintroduction of a packed, threaded flag array.
        for _ in 1:10
            fill!(workspace.field, 0)
            workspace.field[2, 2, 32] = 1
            workspace.field[2, 2, 33] = 1
            box, positive = TP.scalar_and_active_box!(workspace, 0.5f0)
            @test positive
            @test box == (1:3, 1:3, 31:34)
        end
    end

    @testset "union-find roots before parallel accumulation" begin
        # A balanced merge tree in reverse leaf order leaves paths longer than two.
        # Path halving alone does not make every starting vertex a direct child of root.
        faces = TP.Face[]
        for stride in (1, 2, 4, 8, 16, 32, 64, 128)
            for first in reverse(1:(2stride):256)
                push!(faces, TP.Face(first, first + stride, first))
            end
        end
        mesh = SurfaceMesh(fill(SVector{3, Float32}(0, 0, 0), 256), faces)
        pruned, analysis, removal = TP.prune_zero_volume_shells(mesh)
        @test isempty(pruned.faces)
        @test isnothing(analysis)
        @test removal.removed_components == 1

        # Nondegenerate, closed surface with shuffled connectivity; compare the volume
        # against the independent high-precision oriented-face integral.
        sphere = TP.triangulated_sphere_mesh(SVector(0.7, -1.1, 0.4), 0.6)
        shuffled = SurfaceMesh(sphere.vertices,
                               sphere.faces[randperm(MersenneTwister(81),
                                                     length(sphere.faces))])
        @test TP.mesh_geometry_stats(shuffled).volume ≈
              Float64(signed_volume(shuffled.vertices, shuffled.faces)) rtol=1.0e-12
    end

    @testset "nested normals and serialized winding in simulation coordinates" begin
        outer, cavity = cube(-1, 1), cube(-0.5, 0.5)
        mesh = TP.combine_surface_meshes([outer, cavity])
        normals, _, _ = TP.surface_vertex_normals(mesh)
        @test all(dot(normals[i], mesh.vertices[i]) > 0 for i in 1:8)
        @test all(dot(normals[i], mesh.vertices[i]) < 0 for i in 9:16)
        unused = SurfaceMesh(vcat(mesh.vertices, [SVector{3, Float32}(3, 3, 3)]),
                             mesh.faces)
        @test TP.surface_vertex_normals(unused)[1][1:16] == normals

        mktempdir() do dir
            write_ply(mesh, joinpath(dir, "surface.ply"))
            ply_points,
            ply_faces = open(joinpath(dir, "surface.ply")) do io
                while readline(io) != "end_header"
                end
                vertices = read!(io, Matrix{Float32}(undef, 6, length(mesh.vertices)))
                faces = TP.Face[]
                for _ in mesh.faces
                    @test read(io, UInt8) == 3
                    push!(faces, TP.Face(ntuple(_ -> read(io, Int32) + 1, 3)))
                end
                @test eof(io)
                return SVector{3, Float32}.(eachcol(vertices[1:3, :])), faces
            end
            @test signed_volume(ply_points, ply_faces) ≈ 7 atol=1.0e-30
            for level in (false, 1, true)
                file = trixi2vtk(mesh; output_directory=dir, filename="surface",
                                 overwrite=true, compress=level)
                vtk = TP.ReadVTK.VTKFile(file * ".vtp")
                vertices = TP.ReadVTK.get_points(vtk)
                primitives = TP.ReadVTK.get_primitives(vtk, "Polys")
                faces = TP.Face.(eachcol(reshape(primitives.connectivity, 3, :)))
                @test vertices == reduce(hcat, mesh.vertices)
                @test faces == ply_faces
                @test signed_volume(SVector{3, Float64}.(eachcol(vertices)), faces) ≈ 7 atol=1.0e-30
                @test TP.ReadVTK.get_data(TP.ReadVTK.get_point_data(vtk)["Normals"]) ==
                      reduce(hcat, normals)
            end
        end
        # Float64 vertices / Int64 face indices are supported by the generic mesh API.
        generic = SurfaceMesh(SVector{3, Float64}.(outer.vertices),
                              SVector{3, Int64}.(outer.faces))
        @test TP.surface_vertex_normals(generic)[1] == TP.surface_vertex_normals(outer)[1]
        geometry = TP.TriangleMesh(mesh)
        boundary = BoundaryMesh(geometry)
        @test TP.signed_distance(boundary.bvh, SVector(0.0, 0.0, 0.0)) > 0
        @test TP.signed_distance(boundary.bvh, SVector(0.75, 0.0, 0.0)) < 0
    end

    @testset "lattice and boundary input validation" begin
        mesh = cube(0, 1)
        points = Float64.(reduce(hcat, mesh.vertices))
        duplicate = copy(points)
        duplicate[:, end] = duplicate[:, 1]
        @test_throws ArgumentError lattice_surface_topology(duplicate)
        @test_throws ArgumentError lattice_surface_topology(points[:, 1:4])
        @test_throws ArgumentError TP.build_triangle_bvh(points, mesh.faces[2:end])
        inconsistent = copy(mesh.faces)
        a, b, c = inconsistent[1]
        inconsistent[1] = TP.Face(a, c, b)
        @test_throws ArgumentError TP.build_triangle_bvh(points, inconsistent)
        @test_throws ArgumentError TP.build_triangle_bvh(points, [TP.Face(1, 2, 100)])
        @test_throws ArgumentError TP.build_triangle_bvh(points, mesh.faces; leaf_size=0)
        boundary = BoundaryMesh(points, lattice_surface_topology(points))
        @test BoundaryMesh(Float32.(points), lattice_surface_topology(points)).points ==
              points
        # A surface topology may refer to only the boundary vertices of a point set.
        # An unused interior point must not change signed distances.
        with_unused = hcat(points, reshape([0.5, 0.5, 0.5], 3, 1))
        unused_boundary = BoundaryMesh(with_unused, lattice_surface_topology(points))
        @test TP.signed_distance(unused_boundary.bvh, SVector(0.5, 0.5, 0.5)) ≈
              -0.5 atol=2.0e-15
        # Analytic SDF of a box, including exterior edge and corner Voronoi regions.
        for point in (SVector(0.3, 0.2, 0.4), SVector(1.2, 0.5, 0.5),
             SVector(1.2, -0.2, 0.5), SVector(-0.3, 1.4, 1.2))
            q = abs.(point .- 0.5) .- 0.5
            expected = norm(max.(q, 0)) + min(maximum(q), 0)
            @test TP.signed_distance(boundary.bvh, point) ≈ expected atol=2.0e-15
        end
        # A dimensionful absolute "tie" tolerance must not replace the nearest
        # triangle with a farther one when the geometry is small.
        for scale in (1.0e-6, 1.0e-7)
            scaled = scale .* points
            small = BoundaryMesh(scaled, lattice_surface_topology(scaled))
            @test TP.signed_distance(small.bvh, scale .* SVector(0.1, 0.2, 0.3)) ≈
                  -0.1scale rtol=2.0e-14
        end
    end

    @testset "final-volume postcondition and finite parameters" begin
        geometry = TP.mesh_geometry_stats(cube(0, 1))
        @test isnothing(TP.validate_reconstructed_geometry(geometry, 1.0, 0.1))
        @test_throws ErrorException TP.validate_reconstructed_geometry(geometry, 1.01, 0.1)
        @test_throws ErrorException TP.validate_reconstructed_geometry(merge(geometry,
                                                                             (n_boundary_edges=1,)),
                                                                       1.0, 0.1)
        for kwargs in ((volume_tolerance_percent=Inf,), (boundary_clearance=-1.0,),
             (boundary_clearance=NaN,), (maximum_isovalue=Inf,))
            @test_throws ArgumentError SurfaceReconstruction(; particle_spacing=0.1,
                                                             kwargs...)
        end
        rec = SurfaceReconstruction(; particle_spacing=0.1)
        @test_throws ArgumentError reconstruct_surface!(rec, zeros(3, 0), Float64[])
        @test_throws ArgumentError reconstruct_surface!(rec, zeros(3, 1, 2), [0.001])
        grid = TP.ReconstructionGrid(SVector(0.0, 0.0, 0.0), 0.1, (10, 10, 10))
        excluded = TP.exclude_and_support(fill(-100.0, 3, 1), [1.0], grid)
        @test excluded[3] == 1
        @test excluded[5] === (1:0, 1:0, 1:0)
        for isovalue in (NaN, Inf, -1.0, 1.0)
            @test_throws ArgumentError reconstruct_surface!(rec, zeros(3, 1), [0.001];
                                                            initial_isovalue=isovalue)
        end
    end
end
