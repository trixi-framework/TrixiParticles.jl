@testset verbose=true "GridNeighborhoodSearch" begin
    @testset "Coordinate Limits" begin
        # Test the threshold for very large and very small coordinates.
        coords1 = [Inf, -Inf]
        coords2 = [NaN, 0]
        coords3 = [typemax(Int) + 1.0, -typemax(Int) - 1.0]

        @test TrixiParticles.cell_coords(coords1, 1.0, nothing) ==
              (typemax(Int), typemin(Int))
        @test TrixiParticles.cell_coords(coords2, 1.0, nothing) == (typemax(Int), 0)
        @test TrixiParticles.cell_coords(coords3, 1.0, nothing) ==
              (typemax(Int), typemin(Int))
    end

    @testset "Rectangular Point Cloud 2D" begin
        #### Setup
        # Rectangular filled with equidistant spaced particles
        # from (x, y) = (-0.25, -0.25) to (x, y) = (0.35, 0.35)
        range = -0.25:0.1:0.35
        coordinates1 = hcat(collect.(Iterators.product(range, range))...)
        nparticles = size(coordinates1, 2)

        particle_position1 = [0.05, 0.05]
        particle_spacing = 0.1
        radius = particle_spacing

        # Create neighborhood search
        nhs1 = GridNeighborhoodSearch{2}(radius, nparticles)

        coords_fun(i) = coordinates1[:, i]
        TrixiParticles.initialize!(nhs1, coords_fun)

        # Get each neighbor for `particle_position1`
        neighbors1 = sort(collect(TrixiParticles.eachneighbor(particle_position1, nhs1)))

        # Move particles
        coordinates2 = coordinates1 .+ [1.4, -3.5]

        # Update neighborhood search
        coords_fun2(i) = coordinates2[:, i]
        TrixiParticles.update!(nhs1, coords_fun2)

        # Get each neighbor for updated NHS
        neighbors2 = sort(collect(TrixiParticles.eachneighbor(particle_position1, nhs1)))

        # Change position
        particle_position2 = particle_position1 .+ [1.4, -3.5]

        # Get each neighbor for `particle_position2`
        neighbors3 = sort(collect(TrixiParticles.eachneighbor(particle_position2, nhs1)))

        # Double search radius
        nhs2 = GridNeighborhoodSearch{2}(2 * radius, size(coordinates1, 2))
        TrixiParticles.initialize!(nhs2, coords_fun)

        # Get each neighbor in double search radius
        neighbors4 = sort(collect(TrixiParticles.eachneighbor(particle_position1, nhs2)))

        # Move particles
        coordinates2 = coordinates1 .+ [0.4, -0.4]

        # Update neighborhood search
        TrixiParticles.update!(nhs2, coords_fun2)

        # Get each neighbor in double search radius
        neighbors5 = sort(collect(TrixiParticles.eachneighbor(particle_position1, nhs2)))

        #### Verification
        @test neighbors1 == [17, 18, 19, 24, 25, 26, 31, 32, 33]

        @test neighbors2 == Int[]

        @test neighbors3 == [17, 18, 19, 24, 25, 26, 31, 32, 33]

        @test neighbors4 == [9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 24, 25,
            26, 27, 28, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47,
            48, 49]

        @test neighbors5 == [36, 37, 38, 43, 44, 45]
    end

    @testset "Rectangular Point Cloud 3D" begin
        #### Setup
        # Rectangular filled with equidistant spaced particles
        # from (x, y, z) = (-0.25, -0.25, -0.25) to (x, y) = (0.35, 0.35, 0.35)
        range = -0.25:0.1:0.35
        coordinates1 = hcat(collect.(Iterators.product(range, range, range))...)
        nparticles = size(coordinates1, 2)

        particle_position1 = [0.05, 0.05, 0.05]
        particle_spacing = 0.1
        radius = particle_spacing

        # Create neighborhood search
        nhs1 = GridNeighborhoodSearch{3}(radius, nparticles)

        coords_fun(i) = coordinates1[:, i]
        TrixiParticles.initialize!(nhs1, coords_fun)

        # Get each neighbor for `particle_position1`
        neighbors1 = sort(collect(TrixiParticles.eachneighbor(particle_position1, nhs1)))

        # Move particles
        coordinates2 = coordinates1 .+ [1.4, -3.5, 0.8]

        # Update neighborhood search
        coords_fun2(i) = coordinates2[:, i]
        TrixiParticles.update!(nhs1, coords_fun2)

        # Get each neighbor for updated NHS
        neighbors2 = sort(collect(TrixiParticles.eachneighbor(particle_position1, nhs1)))

        # Change position
        particle_position2 = particle_position1 .+ [1.4, -3.5, 0.8]

        # Get each neighbor for `particle_position2`
        neighbors3 = sort(collect(TrixiParticles.eachneighbor(particle_position2, nhs1)))

        #### Verification
        @test neighbors1 ==
              [115, 116, 117, 122, 123, 124, 129, 130, 131, 164, 165, 166, 171, 172,
            173, 178, 179, 180, 213, 214, 215, 220, 221, 222, 227, 228, 229]

        @test neighbors2 == Int[]

        @test neighbors3 ==
              [115, 116, 117, 122, 123, 124, 129, 130, 131, 164, 165, 166, 171, 172,
            173, 178, 179, 180, 213, 214, 215, 220, 221, 222, 227, 228, 229]
    end

    @testset verbose=true "Periodicity 2D" begin
        @testset "Clean Example" begin
            coords = [-0.08 0.0 0.18 0.1 -0.08
                      -0.12 -0.05 -0.09 0.15 0.39]

            # 3 x 6 cells
            nhs = GridNeighborhoodSearch{2}(0.1, size(coords, 2),
                                            min_corner=[-0.1, -0.2], max_corner=[0.2, 0.4])

            TrixiParticles.initialize!(nhs, coords)

            neighbors = [sort(collect(TrixiParticles.eachneighbor(coords[:, i], nhs)))
                         for i in 1:5]

            # Note that (1, 2) and (2, 3) are not neighbors, but they are in neighboring cells
            @test neighbors[1] == [1, 2, 3, 5]
            @test neighbors[2] == [1, 2, 3]
            @test neighbors[3] == [1, 2, 3]
            @test neighbors[4] == [4]
            @test neighbors[5] == [1, 5]

            neighbors_loop = [Int[] for _ in axes(coords, 2)]

            TrixiParticles.for_particle_neighbor(nothing, nothing,
                                                 coords, coords, nhs,
                                                 particles=axes(coords, 2)) do particle,
                                                                               neighbor,
                                                                               pos_diff,
                                                                               distance
                append!(neighbors_loop[particle], neighbor)
            end

            @test sort(neighbors_loop[1]) == [1, 3, 5]
            @test sort(neighbors_loop[2]) == [2]
            @test sort(neighbors_loop[3]) == [1, 3]
            @test sort(neighbors_loop[4]) == [4]
            @test sort(neighbors_loop[5]) == [1, 5]
        end

        @testset "Offset Domain Triggering Split Cells" begin
            # This used to trigger a "split cell bug", where the left and right boundary
            # cells were only partially contained in the domain.
            # The left particle was placed inside a ghost cells, which causes it to not
            # see the right particle, even though it is within the search distance.
            # The domain size is an integer multiple of the cell size, but the NHS did not
            # offset the grid based on the domain position.
            # See https://github.com/trixi-framework/TrixiParticles.jl/pull/211 for a more
            # detailed explanation.
            coords = [-1.4 1.9
                      0.0 0.0]

            # 5 x 1 cells
            nhs = GridNeighborhoodSearch{2}(1.0, size(coords, 2),
                                            min_corner=[-1.5, 0.0], max_corner=[2.5, 3.0])

            TrixiParticles.initialize!(nhs, coords)

            neighbors = [sort(unique(collect(TrixiParticles.eachneighbor(coords[:, i], nhs))))
                         for i in 1:2]

            @test neighbors[1] == [1, 2]
            @test neighbors[2] == [1, 2]
        end
    end

    @testset verbose=true "Periodicity 3D" begin
        coords = [-0.08 0.0 0.18 0.1 -0.08
                  -0.12 -0.05 -0.09 0.15 0.39
                  0.14 0.34 0.12 0.06 0.13]

        # 3 x 6 x 3 cells
        nhs = GridNeighborhoodSearch{3}(0.1, size(coords, 2),
                                        min_corner=[-0.1, -0.2, 0.05],
                                        max_corner=[0.2, 0.4, 0.35])

        TrixiParticles.initialize!(nhs, coords)

        neighbors = [sort(collect(TrixiParticles.eachneighbor(coords[:, i], nhs)))
                     for i in 1:5]

        # Note that (1, 2) and (2, 3) are not neighbors, but they are in neighboring cells
        @test neighbors[1] == [1, 2, 3, 5]
        @test neighbors[2] == [1, 2, 3]
        @test neighbors[3] == [1, 2, 3]
        @test neighbors[4] == [4]
        @test neighbors[5] == [1, 5]

        neighbors_loop = [Int[] for _ in axes(coords, 2)]

        TrixiParticles.for_particle_neighbor(coords, coords,
                                             nhs) do particle, neighbor, pos_diff, distance
            append!(neighbors_loop[particle], neighbor)
        end

        @test sort(neighbors_loop[1]) == [1, 3, 5]
        @test sort(neighbors_loop[2]) == [2]
        @test sort(neighbors_loop[3]) == [1, 3]
        @test sort(neighbors_loop[4]) == [4]
        @test sort(neighbors_loop[5]) == [1, 5]
    end
end
