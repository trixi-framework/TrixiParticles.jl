@testset verbose=true "FaceNeighborhoodSearch" begin
    @testset verbose=true "2D Cell Intersection" begin
        triangle = [0.0 1.0 0.5 0.0;
                    0.0 0.0 0.7 0.0]

        # Only use the third edge of the triangle, i.e. the edge from [0.5, 0.7] to [0.0, 0.0]
        edge_aligned = deleteat!(TrixiParticles.Polygon(triangle), [1, 2])
        edge_id = 1 # Only one edge in `Polygon`

        search_radii = [1.0, 0.1]

        cell_coords = [(0.0, 0.0), (-0.2, -0.2), (0.0, sqrt(eps())), (0.4, -0.05)]
        intersections = [[true, true], [true, false], [false, false], [true, true]]

        @testset verbose=true "Axis Aligned Edge: cell $i, search radius $j" for i in eachindex(cell_coords),
                                                                                 j in eachindex(search_radii)

            nhs = TrixiParticles.FaceNeighborhoodSearch{2}(; search_radius=search_radii[j])

            @test intersections[i][j] ==
                  TrixiParticles.face_intersects_cell(edge_id, edge_aligned,
                                                      cell_coords[i] ./ search_radii[j],
                                                      nhs)
        end

        edge_arbitrary = deleteat!(TrixiParticles.Polygon(triangle), [2, 3])
        edge_id = 1 # Only one edge in `Polygon`

        search_radii = [1.0, 0.1]

        cell_coords = [(0.0, 0.0), (0.2, 0.2), (0.5, 0.0)]
        intersections = [[true, true], [true, true], [true, false]]

        @testset verbose=true "Arbitrary Edge: cell $i, search radius $j" for i in eachindex(cell_coords),
                                                                              j in eachindex(search_radii)

            nhs = TrixiParticles.FaceNeighborhoodSearch{2}(; search_radius=search_radii[j])

            @test intersections[i][j] ==
                  TrixiParticles.face_intersects_cell(edge_id, edge_arbitrary,
                                                      cell_coords[i] ./ search_radii[j],
                                                      nhs)
        end
    end

    # Tested with Paraview and `save("triangle.stl", triangle)`, `trixi2vtk(stack([min_corner, max_corner]))`
    @testset verbose=true "3D Cell Intersection" begin
        # Axis aligned triangle
        A = SVector(0.0, 1.0, 0.0)
        B = zero(A)
        C = SVector(1.0, 1.0, 0.0)

        triangle_aligned = TrixiParticles.TriangleMesh([(A, B, C)],
                                                       [TrixiParticles.cross(B - A, C - A)],
                                                       [A, B, C])
        face_id = 1 # Only one face in `TriangleMesh`

        search_radii = [1.0, 0.1, 2.0]

        cell_coords = [
            (-0.1, -0.1, 0.0),
            (-0.1, -0.1, 0.1),
            (-0.1, -0.1, -sqrt(eps())),
            (15.0, 15.0, -0.5)
        ]

        intersections = [
            [true, true, true],
            [false, false, false],
            [true, true, true],
            # This returns `true' incorrectly, because the triangle plane intersects
            # the cell but the triangle itself does not.
            [true, true, true]
        ]

        @testset verbose=true "Axis Aligned Triangle: cell $i, search radius $j" for i in eachindex(cell_coords),
                                                                                     j in eachindex(search_radii)

            nhs = TrixiParticles.FaceNeighborhoodSearch{3}(; search_radius=search_radii[j])

            @test intersections[i][j] ==
                  TrixiParticles.face_intersects_cell(face_id, triangle_aligned,
                                                      cell_coords[i], nhs)
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
        search_radii = [1.0, 0.1, 5.0]

        cell_coords = [(-0.1, -0.1, 0.0), (-0.5, -0.5, -0.1), (1.8, 3.3, 4.2)]

        intersections = [[true, false, true], [true, false, true], [false, true, false]]

        @testset verbose=true "Arbitrary Triangle: cell $i, search radius $j" for i in eachindex(cell_coords),
                                                                                  j in eachindex(search_radii)

            nhs = TrixiParticles.FaceNeighborhoodSearch{3}(; search_radius=search_radii[j])

            @test intersections[i][j] ==
                  TrixiParticles.face_intersects_cell(face_id, triangle_arbitrary,
                                                      cell_coords[i], nhs)
        end
    end
end
