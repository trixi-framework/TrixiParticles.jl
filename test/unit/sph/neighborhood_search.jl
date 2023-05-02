@testset verbose=true "Neighborhood Search" begin
    @testset verbose=true "TrivialNeighborhoodSearch" begin
        # Setup with 5 particles
        nhs = TrixiParticles.TrivialNeighborhoodSearch(Base.OneTo(5))

        # Get each neighbor for arbitrary coordinates
        neighbors = collect(TrixiParticles.eachneighbor([1.0, 2.0], nhs))

        #### Verification
        @test neighbors == [1, 2, 3, 4, 5]
    end

    @testset verbose=true "SpatialHashingSearch" begin
        @testset "Coordinate Limits" begin
            # Test the threshold for very large and very small coordinates.
            nhs = SpatialHashingSearch{2}(1.0, 0)
            coords1 = [Inf, -Inf]
            coords2 = [NaN, 0]
            coords3 = [typemax(Int) + 1.0, -typemax(Int) - 1.0]

            @test TrixiParticles.cell_coords(coords1, nhs) == (typemax(Int), typemin(Int))
            @test TrixiParticles.cell_coords(coords2, nhs) == (typemax(Int), 0)
            @test TrixiParticles.cell_coords(coords3, nhs) == (typemax(Int), typemin(Int))
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
            nhs1 = SpatialHashingSearch{2}(radius, nparticles)

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
            nhs2 = SpatialHashingSearch{2}(2 * radius, size(coordinates1, 2))
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
            nhs1 = SpatialHashingSearch{3}(radius, nparticles)

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
    end
end
