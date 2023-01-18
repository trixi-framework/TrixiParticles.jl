@testset "Neighborhood search" begin
    # Test the threshold for very large and very small coordinates.
    NHS = SpatialHashingSearch{2}(1.0)
    coords1 = [Inf, -Inf]
    coords2 = [NaN, 0]
    coords4 = [typemax(Int) + 1.0, -typemax(Int) - 1.0]

    @test Pixie.get_cell_coords(coords1, NHS) == (typemax(Int), typemin(Int))
    @test Pixie.get_cell_coords(coords2, NHS) == (typemax(Int), 0)
    @test Pixie.get_cell_coords(coords4, NHS) == (typemax(Int), typemin(Int))

    @testset "SpatialHashingSearch" begin
        #### Setup
        # Rectangular filled with equidistant spaced particles
        # from (x, y) = (-0.25, -0.25) to (x, y) = (0.35, 0.35)
        coordinates1 = [-0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.15 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35
                        -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35 -0.25 -0.15 -0.05 0.05 0.15 0.25 0.35]

        particle_position1 = [0.05; 0.05]
        particle_spacing = 0.1
        radius = particle_spacing

        #### Mocking
        # Mock the container
        container = Val(:mock_container)

        # create neighborhood search
        NHS1 = SpatialHashingSearch{2}(radius)
        Pixie.eachparticle(container) = Base.OneTo(size(coordinates1, 2))
        Pixie.ndims(container) = size(coordinates1, 1)

        Pixie.initialize!(NHS1, coordinates1, container)

        # get each neighbor for `particle_position1`
        neighbors1 = collect(Pixie.eachneighbor(particle_position1, NHS1))

        # move particles
        coordinates2 = coordinates1 .+ [1.4, -3.5]

        # update neighborhood search
        Pixie.update!(NHS1, coordinates2, container)

        # get each neighbor for updated NHS
        neighbors2 = collect(Pixie.eachneighbor(particle_position1, NHS1))

        # change position
        particle_position2 = particle_position1 .+ [1.4, -3.5]

        # get each neighbor for `particle_position2`
        neighbors3 = collect(Pixie.eachneighbor(particle_position2, NHS1))

        # double search radius
        NHS2 = SpatialHashingSearch{2}(2 * radius)
        Pixie.initialize!(NHS2, coordinates1, container)

        # get each neighbor in double search radius
        neighbors4 = collect(Pixie.eachneighbor(particle_position1, NHS2))

        #### Verification
        @test neighbors1 == [17
                             24
                             31
                             18
                             25
                             32
                             19
                             26
                             33]

        @test neighbors2 == Int[]

        @test neighbors1 == [17
                             24
                             31
                             18
                             25
                             32
                             19
                             26
                             33]

        @test neighbors4 == [9
               10
               16
               17
               23
               24
               30
               31
               37
               38
               44
               45
               11
               12
               18
               19
               25
               26
               32
               33
               39
               40
               46
               47
               13
               14
               20
               21
               27
               28
               34
               35
               41
               42
               48
               49]
    end
end
