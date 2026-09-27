@trixi_testset "surface reconstruction reference values" begin
    # Recorded results for the production configuration: volume-CIC deposition,
    # particle-volume correction, voxel size h/2, and Gaussian sigma 0.9h.
    # Tight tolerances allow cross-platform roundoff. The benchmark's
    # `surface_reconstruction/determinism.jl` script checks bitwise agreement across threads.
    #
    # Provenance: fixed-tank values were recorded from the production implementation on
    # 2026-09-23. Adaptive values were recorded after correcting an origin-clipping bug:
    # previously, the domain's lower corner was pinned at world origin and clipped the
    # fluid on three faces. Adaptive reconstruction is now translation-invariant and
    # agrees with an equivalent far-wall tank to seven digits.
    @testset verbose=true "reference cases" begin
        spacing = 0.1
        # `vec` makes `reduce(hcat, ...)` return a `Matrix` instead of a slow-to-compile
        # `SMatrix`; results are bitwise identical for both input types
        points = reduce(hcat,
                        vec([TrixiParticles.SVector{3, Float64}(x, y, z) * spacing
                             for z in 0:6, y in 0:6, x in 0:6]))
        volumes = fill(spacing^3, size(points, 2))

        cube = reduce(hcat,
                      vec([TrixiParticles.SVector{3, Float64}(0.2, 0.2, 0.2) +
                           0.3 * TrixiParticles.SVector{3, Float64}(i, j, k)
                           for i in (0.0, 1.0), j in (0.0, 1.0), k in (0.0, 1.0)]))
        cube_topo = TrixiParticles.lattice_surface_topology(cube)
        cube_mesh = TrixiParticles.BoundaryMesh(cube, cube_topo)
        enclosed_block = TrixiParticles.enclosed_particles(points, [cube_mesh])
        retained = findall(iszero, enclosed_block)

        points_sparse = hcat(points, reshape([0.9, 0.9, 0.9], 3, 1))
        volumes_sparse = vcat(volumes, spacing^3)

        function check_reference_values(mesh, stats; isovalue, volume, area, vertices,
                                        faces, components=1, cavities=0, inside=0)
            @test isapprox(stats["effective_isovalue"], isovalue; rtol=1.0e-10)
            @test isapprox(stats["volume"], volume; rtol=1.0e-10)
            @test isapprox(stats["surface_area"], area; rtol=1.0e-10)
            @test length(mesh.vertices) == vertices
            @test length(mesh.faces) == faces
            @test stats["n_boundary_edges"] == 0
            @test stats["n_nonmanifold_edges"] == 0
            @test stats["n_degenerate_triangles"] == 0
            @test stats["n_connected_regions"] == components
            @test stats["n_cavity_regions"] == cavities
            @test stats["n_vertices_inside_boundaries"] == inside
            # Every closed component must hit the production volume gate
            @test abs(100 * (stats["volume"] - stats["particle_volume"]) /
                      stats["particle_volume"]) <= 0.1
        end

        @testset "volume-cic production defaults" begin
            mesh,
            stats = reconstruct_surface(points, volumes;
                                        particle_spacing=spacing,
                                        voxel_size=spacing / 2)
            check_reference_values(mesh, stats;
                                   isovalue=0.40896298462440717, volume=0.34278756422335027,
                                   area=2.510226552487736, vertices=1152, faces=2300)
        end

        @testset "volume-cic production defaults + fixed tank" begin
            mesh,
            stats = reconstruct_surface(points, volumes;
                                        particle_spacing=spacing,
                                        voxel_size=spacing / 2,
                                        tank_size=(1.0, 1.0, 1.0))
            check_reference_values(mesh, stats;
                                   isovalue=0.2158546563506937, volume=0.3432752718463588,
                                   area=2.678826788009569, vertices=1170, faces=2336)
        end

        @testset "boundary cavity production defaults" begin
            mesh,
            stats = reconstruct_surface(points[:, retained], volumes[retained],
                                        [cube_mesh];
                                        particle_spacing=spacing,
                                        voxel_size=spacing / 2,
                                        boundary_clearance=0.0)
            check_reference_values(mesh, stats;
                                   isovalue=0.38029438737128535, volume=0.3347313775727149,
                                   area=3.08222206393927, vertices=1512, faces=3016,
                                   cavities=1)
        end

        @testset "sparse component fallback" begin
            mesh,
            stats = reconstruct_surface(points_sparse, volumes_sparse;
                                        particle_spacing=spacing,
                                        voxel_size=spacing / 2,
                                        sparse_component_fallback=true)
            check_reference_values(mesh, stats;
                                   isovalue=0.4086214230064432, volume=0.3439986415524085,
                                   area=2.55973822852343, vertices=1634, faces=3260,
                                   components=2)
            @test stats["n_sparse_fallback_components"] == 1
        end

        @testset "adaptive grid is translation-invariant" begin
            base_mesh,
            base_stats = reconstruct_surface(points, volumes;
                                             particle_spacing=spacing,
                                             voxel_size=spacing / 2)
            # Half-voxel, incommensurate, and negative shifts exercise grid movement
            # (the old adaptive path clipped at the world origin).
            for shift in (spacing / 2, 0.013, -0.7)
                mesh,
                stats = reconstruct_surface(points .+ shift, volumes;
                                            particle_spacing=spacing,
                                            voxel_size=spacing / 2)
                @test stats["effective_isovalue"] ≈ base_stats["effective_isovalue"] rtol=1.0e-7
                @test stats["volume"] ≈ base_stats["volume"] rtol=1.0e-7
                @test length(mesh.vertices) == length(base_mesh.vertices)
                @test length(mesh.faces) == length(base_mesh.faces)
                translation = SVector{3, Float32}(shift, shift, shift)
                # Same vertex set up to Float32 rounding of the shifted grid coordinates
                mismatch = maximum(minimum(sum(abs, (vertex + translation) - other)
                                           for other in mesh.vertices)
                                   for vertex in base_mesh.vertices)
                @test mismatch < 1.0e-5
            end

            # Far from every wall, a fixed tank and the adaptive grid agree
            far_mesh,
            far_stats = reconstruct_surface(points .+ 1.0, volumes;
                                            particle_spacing=spacing,
                                            voxel_size=spacing / 2,
                                            tank_size=(2.6, 2.6, 2.6))
            @test far_stats["effective_isovalue"] ≈ base_stats["effective_isovalue"] rtol=1.0e-6
            @test length(far_mesh.vertices) == length(base_mesh.vertices)
        end
    end
end
