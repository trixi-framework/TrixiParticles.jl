@testset verbose=true "FaceNeighborhoodSearch" begin
    @testset verbose=true "2D Cell Bounding Box" begin
        triangle = [0.0 1.0 0.5 0.0;
                    0.0 0.0 0.7 0.0]

        # Only use the third edge of the triangle, i.e. the edge from [0.1, 0.0] to [0.0, 0.0]
        edge_aligned = deleteat!(TrixiParticles.Polygon(triangle), [1, 2])
        edge_id = 1 # Only one edge in `Polygon`

        cell_sizes = [1.0 + sqrt(eps()), 0.1]

        expected_ncells_bbox = [(1, 1), (11, 1)]
        # One padding layer in each direction around the bounding box
        expected_ncells_filled = map(x -> prod(x .+ 2), expected_ncells_bbox)

        @testset verbose=true "Axis Aligned Edge: cell size $(cell_size)" for (i,
                                                                               cell_size) in
                                                                              enumerate(cell_sizes)
            nhs = TrixiParticles.FaceNeighborhoodSearch{2}(; search_radius=cell_size)

            TrixiParticles.initialize!(nhs, edge_aligned)

            @test expected_ncells_bbox[i] ==
                  size(collect(TrixiParticles.bounding_box(edge_id, edge_aligned, nhs)))

            @test expected_ncells_filled[i] == length(nhs.neighbors.hashtable)
        end

        # Only use the first edge of the triangle, i.e. the edge from [0.0, 0.0] to [0.5, 0.7]
        edge_arbitrary = deleteat!(TrixiParticles.Polygon(triangle), [2, 3])
        edge_id = 1 # Only one edge in `Polygon`

        expected_ncells_bbox = [(1, 1), (6, 7)]
        # One padding layer in each direction around the bounding box
        expected_ncells_filled = map(x -> prod(x .+ 2), expected_ncells_bbox)

        @testset verbose=true "Arbitrary Edge: cell size $(cell_size)" for (i, cell_size) in
                                                                           enumerate(cell_sizes)
            nhs = TrixiParticles.FaceNeighborhoodSearch{2}(; search_radius=cell_size)

            TrixiParticles.initialize!(nhs, edge_arbitrary)

            @test expected_ncells_bbox[i] ==
                  size(collect(TrixiParticles.bounding_box(edge_id, edge_arbitrary, nhs)))

            @test expected_ncells_filled[i] == length(nhs.neighbors.hashtable)
        end
    end

    # Tested with Paraview and `save("triangle.stl", triangle)`, `trixi2vtk(stack([min_corner, max_corner]))`
    @testset verbose=true "3D Cell Bounding Box" begin
        # Axis aligned triangle
        A = SVector(0.0, 1.0, 0.0)
        B = zero(A)
        C = SVector(1.0, 1.0, 0.0)

        triangle_aligned = TrixiParticles.TriangleMesh([(A, B, C)],
                                                       [TrixiParticles.cross(B - A, C - A)],
                                                       [A, B, C])
        face_id = 1 # Only one face in `TriangleMesh`

        cell_sizes = [1.0 + sqrt(eps()), 0.1]

        expected_ncells_bbox = [(1, 1, 1), (11, 11, 1)]
        # One padding layer in each direction around the bounding box
        expected_ncells_filled = map(x -> prod(x .+ 2), expected_ncells_bbox)

        @testset verbose=true "Axis Aligned Triangle: cell size $(cell_size)" for (i,
                                                                                   cell_size) in
                                                                                  enumerate(cell_sizes)
            nhs = TrixiParticles.FaceNeighborhoodSearch{3}(; search_radius=cell_size)

            TrixiParticles.initialize!(nhs, triangle_aligned)

            @test expected_ncells_bbox[i] ==
                  size(collect(TrixiParticles.bounding_box(face_id, triangle_aligned, nhs)))

            @test expected_ncells_filled[i] == length(nhs.neighbors.hashtable)
        end

        # Arbitrary triangle
        A = SVector(0.42, -0.16, 0.81)
        B = SVector(-0.58, 0.74, -0.02)
        C = SVector(0.87, 1.02, 0.37)

        triangle_arbitrary = TrixiParticles.TriangleMesh([(A, B, C)],
                                                         [TrixiParticles.cross(B - A,
                                                                               C - A)],
                                                         [A, B, C])
        face_id = 1 # Only one face in `TriangleMesh`
        cell_sizes = [1.0 + sqrt(eps()), 0.1]

        expected_ncells_bbox = [(2, 3, 2), (15, 13, 10)]
        # One padding layer in each direction around the bounding box
        expected_ncells_filled = map(x -> prod(x .+ 2), expected_ncells_bbox)

        @testset verbose=true "Arbitrary Triangle: cell size $(cell_size)" for (i,
                                                                                cell_size) in
                                                                               enumerate(cell_sizes)
            nhs = TrixiParticles.FaceNeighborhoodSearch{3}(; search_radius=cell_size)

            TrixiParticles.initialize!(nhs, triangle_arbitrary)

            @test expected_ncells_bbox[i] ==
                  size(collect(TrixiParticles.bounding_box(face_id, triangle_arbitrary,
                                                           nhs)))

            @test expected_ncells_filled[i] == length(nhs.neighbors.hashtable)
        end
    end
end
