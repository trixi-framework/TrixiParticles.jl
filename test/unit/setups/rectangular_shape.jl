# 2D
@testset "Rectangular Shape 2D" begin

    # Rectangular shape
    particle_spacing = 0.01
    n_particles_per_dimension = (4, 3)

    positions = [(0.0, 0.0), (3.0, 4.0), (-3.0, -4.0), (-3.0, 4.0), (3.0, -4.0)]

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

    @testset "Position $i" for i in eachindex(positions)
        shape = RectangularShape(particle_spacing,
                                 n_particles_per_dimension, positions[i], 0.0)

        @test shape.coordinates == expected_coords[i]
    end
end

# 3D
@testset "Rectangular Shape 3D" begin

    # Rectangular shape
    particle_spacing = 0.01
    n_particles_per_dimension = (4, 3, 5)

    positions = [(0.0, 0.0, 0.0), (3.0, 4.0, 2.0), (-3.0, -4.0, -2.0)]

    expected_coords = [
        [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03;
         0.0 0.0 0.0 0.0 0.0 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.0 0.0 0.0 0.0 0.0 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.0 0.0 0.0 0.0 0.0 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.0 0.0 0.0 0.0 0.0 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02;
         0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04 0.0 0.01 0.02 0.03 0.04],
        [3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.01 3.01 3.01 3.01 3.01 3.01 3.01 3.01 3.01 3.01 3.01 3.01 3.01 3.01 3.01 3.02 3.02 3.02 3.02 3.02 3.02 3.02 3.02 3.02 3.02 3.02 3.02 3.02 3.02 3.02 3.03 3.03 3.03 3.03 3.03 3.03 3.03 3.03 3.03 3.03 3.03 3.03 3.03 3.03 3.03;
         4.0 4.0 4.0 4.0 4.0 4.01 4.01 4.01 4.01 4.01 4.02 4.02 4.02 4.02 4.02 4.0 4.0 4.0 4.0 4.0 4.01 4.01 4.01 4.01 4.01 4.02 4.02 4.02 4.02 4.02 4.0 4.0 4.0 4.0 4.0 4.01 4.01 4.01 4.01 4.01 4.02 4.02 4.02 4.02 4.02 4.0 4.0 4.0 4.0 4.0 4.01 4.01 4.01 4.01 4.01 4.02 4.02 4.02 4.02 4.02;
         2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04 2.0 2.01 2.02 2.03 2.04],
        [-3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -3.0 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.99 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.98 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97 -2.97;
         -4.0 -4.0 -4.0 -4.0 -4.0 -3.99 -3.99 -3.99 -3.99 -3.99 -3.98 -3.98 -3.98 -3.98 -3.98 -4.0 -4.0 -4.0 -4.0 -4.0 -3.99 -3.99 -3.99 -3.99 -3.99 -3.98 -3.98 -3.98 -3.98 -3.98 -4.0 -4.0 -4.0 -4.0 -4.0 -3.99 -3.99 -3.99 -3.99 -3.99 -3.98 -3.98 -3.98 -3.98 -3.98 -4.0 -4.0 -4.0 -4.0 -4.0 -3.99 -3.99 -3.99 -3.99 -3.99 -3.98 -3.98 -3.98 -3.98 -3.98;
         -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96 -2.0 -1.99 -1.98 -1.97 -1.96],
    ]

    @testset "Position $i" for i in eachindex(positions)
        shape = RectangularShape(particle_spacing,
                                 n_particles_per_dimension, positions[i], 0.0)

        @test shape.coordinates == expected_coords[i]
    end
end
