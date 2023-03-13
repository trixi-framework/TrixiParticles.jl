@testset "Rectangular Shape" begin

    # Rectangular shape
    particle_spacing = 0.01
    n_particles_per_dimension = (4, 3)

    position = [[0 0], [3.0 4.0], [-3.0 -4.0], [-3.0 4.0], [3.0 -4]]

    expected_coords = [
        [0.0 0.0 0.0 0.01 0.01 0.01 0.02 0.02 0.02 0.03 0.03 0.03;
         0.0 0.01 0.02 0.0 0.01 0.02 0.0 0.01 0.02 0.0 0.01 0.02],
        [3.0 3.0 3.0 3.01 3.01 3.01 3.02 3.02 3.02 3.03 3.03 3.03;
         4.0 4.01 4.02 4.0 4.01 4.02 4.0 4.01 4.02 4.0 4.01 4.02],
        [-3.0 -3.0 -3.0 -2.99 -2.99 -2.99 -2.98 -2.98 -2.98 -2.97 -2.97 -2.97;
         -4.0 -3.99 -3.98 -4.0 -3.99 -3.98 -4.0 -3.99 -3.98 -4.0 -3.99 -3.98],
        [-3.0 -3.0 -3.0 -2.99 -2.99 -2.99 -2.98 -2.98 -2.98 -2.97 -2.97 -2.97;
         4.0 4.01 4.02 4.0 4.01 4.02 4.0 4.01 4.02 4.0 4.01 4.02],
        [3.0 3.0 3.0 3.01 3.01 3.01 3.02 3.02 3.02 3.03 3.03 3.03;
         -4.0 -3.99 -3.98 -4.0 -3.99 -3.98 -4.0 -3.99 -3.98 -4.0 -3.99 -3.98],
    ]

    @testset "Position $i" for i in eachindex(position)
        shape = RectangularShape(particle_spacing,
                                 (n_particles_per_dimension[1],
                                  n_particles_per_dimension[2]),
                                 (position[i][1],
                                  position[i][2]))

        @test shape.coordinates == expected_coords[i]
    end
end
