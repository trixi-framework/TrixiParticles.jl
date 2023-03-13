@testset "Circular Shape" begin
    position = [[0 0], [3.0 4.0], [-3.0 -4.0], [-3.0 4.0], [3.0 -4]]

    # Circular shape
    radius = 2.0
    particle_spacing = 0.8

    expected_coords1 = [
        [-0.8 0.0 0.8 -1.6 -0.8 0.0 0.8 1.6 -1.6 -0.8 0.0 0.8 1.6 -1.6 -0.8 0.0 0.8 1.6 -0.8 0.0 0.8
         -1.6 -1.6 -1.6 -0.8 -0.8 -0.8 -0.8 -0.8 0.0 0.0 0.0 0.0 0.0 0.8 0.8 0.8 0.8 0.8 1.6 1.6 1.6],
        [2.2 3.0 3.8 1.4 2.2 3.0 3.8 4.6 1.4 2.2 3.0 3.8 4.6 1.4 2.2 3.0 3.8 4.6 2.2 3.0 3.8
         2.4 2.4 2.4 3.2 3.2 3.2 3.2 3.2 4.0 4.0 4.0 4.0 4.0 4.8 4.8 4.8 4.8 4.8 5.6 5.6 5.6],
        [-3.8 -3.0 -2.2 -4.6 -3.8 -3.0 -2.2 -1.4 -4.6 -3.8 -3.0 -2.2 -1.4 -4.6 -3.8 -3.0 -2.2 -1.4 -3.8 -3.0 -2.2
         -5.6 -5.6 -5.6 -4.8 -4.8 -4.8 -4.8 -4.8 -4.0 -4.0 -4.0 -4.0 -4.0 -3.2 -3.2 -3.2 -3.2 -3.2 -2.4 -2.4 -2.4],
        [-3.8 -3.0 -2.2 -4.6 -3.8 -3.0 -2.2 -1.4 -4.6 -3.8 -3.0 -2.2 -1.4 -4.6 -3.8 -3.0 -2.2 -1.4 -3.8 -3.0 -2.2
         2.4 2.4 2.4 3.2 3.2 3.2 3.2 3.2 4.0 4.0 4.0 4.0 4.0 4.8 4.8 4.8 4.8 4.8 5.6 5.6 5.6],
        [2.2 3.0 3.8 1.4 2.2 3.0 3.8 4.6 1.4 2.2 3.0 3.8 4.6 1.4 2.2 3.0 3.8 4.6 2.2 3.0 3.8
         -5.6 -5.6 -5.6 -4.8 -4.8 -4.8 -4.8 -4.8 -4.0 -4.0 -4.0 -4.0 -4.0 -3.2 -3.2 -3.2 -3.2 -3.2 -2.4 -2.4 -2.4],
    ]

    expected_coords2 = [
        [-0.8 0.0 0.8 -1.6 -0.8 0.0 0.8 1.6 -1.6 -0.8 -1.6 -0.8 0.0 0.8 1.6 -0.8 0.0 0.8
         -1.6 -1.6 -1.6 -0.8 -0.8 -0.8 -0.8 -0.8 0.0 0.0 0.8 0.8 0.8 0.8 0.8 1.6 1.6 1.6],
        [2.2 3.0 3.8 1.4 2.2 3.0 3.8 4.6 1.4 2.2 1.4 2.2 3.0 3.8 4.6 2.2 3.0 3.8
         2.4 2.4 2.4 3.2 3.2 3.2 3.2 3.2 4.0 4.0 4.8 4.8 4.8 4.8 4.8 5.6 5.6 5.6],
        [-3.8 -3.0 -2.2 -4.6 -3.8 -3.0 -2.2 -1.4 -4.6 -3.8 -4.6 -3.8 -3.0 -2.2 -1.4 -3.8 -3.0 -2.2
         -5.6 -5.6 -5.6 -4.8 -4.8 -4.8 -4.8 -4.8 -4.0 -4.0 -3.2 -3.2 -3.2 -3.2 -3.2 -2.4 -2.4 -2.4],
        [-3.8 -3.0 -2.2 -4.6 -3.8 -3.0 -2.2 -1.4 -4.6 -3.8 -4.6 -3.8 -3.0 -2.2 -1.4 -3.8 -3.0 -2.2
         2.4 2.4 2.4 3.2 3.2 3.2 3.2 3.2 4.0 4.0 4.8 4.8 4.8 4.8 4.8 5.6 5.6 5.6],
        [2.2 3.0 3.8 1.4 2.2 3.0 3.8 4.6 1.4 2.2 1.4 2.2 3.0 3.8 4.6 2.2 3.0 3.8
         -5.6 -5.6 -5.6 -4.8 -4.8 -4.8 -4.8 -4.8 -4.0 -4.0 -3.2 -3.2 -3.2 -3.2 -3.2 -2.4 -2.4 -2.4],
    ]

    expected_coords3 = [
        [0.8 0.4 -0.4 -0.8 -0.4 0.4 1.6 1.41673 0.908904 0.192859 -0.567368 -1.19762 -1.55351 -1.55351 -1.19762 -0.567368 0.192859 0.908904 1.41673
         0.0 0.69282 0.69282 9.79717e-17 -0.69282 -0.69282 0.0 0.743557 1.31677 1.58833 1.49603 1.061 0.382905 -0.382905 -1.061 -1.49603 -1.58833 -1.31677 -0.743557],
        [3.8 3.4 2.6 2.2 2.6 3.4 4.6 4.41673 3.9089 3.19286 2.43263 1.80238 1.44649 1.44649 1.80238 2.43263 3.19286 3.9089 4.41673
         4.0 4.69282 4.69282 4.0 3.30718 3.30718 4.0 4.74356 5.31677 5.58833 5.49603 5.061 4.38291 3.61709 2.939 2.50397 2.41167 2.68323 3.25644],
        [-2.2 -2.6 -3.4 -3.8 -3.4 -2.6 -1.4 -1.58327 -2.0911 -2.80714 -3.56737 -4.19762 -4.55351 -4.55351 -4.19762 -3.56737 -2.80714 -2.0911 -1.58327
         -4.0 -3.30718 -3.30718 -4.0 -4.69282 -4.69282 -4.0 -3.25644 -2.68323 -2.41167 -2.50397 -2.939 -3.61709 -4.38291 -5.061 -5.49603 -5.58833 -5.31677 -4.74356],
        [-2.2 -2.6 -3.4 -3.8 -3.4 -2.6 -1.4 -1.58327 -2.0911 -2.80714 -3.56737 -4.19762 -4.55351 -4.55351 -4.19762 -3.56737 -2.80714 -2.0911 -1.58327
         4.0 4.69282 4.69282 4.0 3.30718 3.30718 4.0 4.74356 5.31677 5.58833 5.49603 5.061 4.38291 3.61709 2.939 2.50397 2.41167 2.68323 3.25644],
        [3.8 3.4 2.6 2.2 2.6 3.4 4.6 4.41673 3.9089 3.19286 2.43263 1.80238 1.44649 1.44649 1.80238 2.43263 3.19286 3.9089 4.41673
         -4.0 -3.30718 -3.30718 -4.0 -4.69282 -4.69282 -4.0 -3.25644 -2.68323 -2.41167 -2.50397 -2.939 -3.61709 -4.38291 -5.061 -5.49603 -5.58833 -5.31677 -4.74356],
    ]

    expected_coords4 = [
        [0.4 -0.2 -0.2 1.2 0.919253 0.208378 -0.6 -1.12763 -1.12763 -0.6 0.208378 0.919253 2.0 1.84776 1.41421 0.765367 5.66554e-16 -0.765367 -1.41421 -1.84776 -2.0 -1.84776 -1.41421 -0.765367 -3.67394e-16 0.765367 1.41421 1.84776
         0.0 0.34641 -0.34641 0.0 0.771345 1.18177 1.03923 0.410424 -0.410424 -1.03923 -1.18177 -0.771345 0.0 0.765367 1.41421 1.84776 2.0 1.84776 1.41421 0.765367 1.13311e-15 -0.765367 -1.41421 -1.84776 -2.0 -1.84776 -1.41421 -0.765367],
        [3.4 2.8 2.8 4.2 3.91925 3.20838 2.4 1.87237 1.87237 2.4 3.20838 3.91925 5.0 4.84776 4.41421 3.76537 3.0 2.23463 1.58579 1.15224 1.0 1.15224 1.58579 2.23463 3.0 3.76537 4.41421 4.84776
         4.0 4.34641 3.65359 4.0 4.77135 5.18177 5.03923 4.41042 3.58958 2.96077 2.81823 3.22865 4.0 4.76537 5.41421 5.84776 6.0 5.84776 5.41421 4.76537 4.0 3.23463 2.58579 2.15224 2.0 2.15224 2.58579 3.23463],
        [-2.6 -3.2 -3.2 -1.8 -2.08075 -2.79162 -3.6 -4.12763 -4.12763 -3.6 -2.79162 -2.08075 -1.0 -1.15224 -1.58579 -2.23463 -3.0 -3.76537 -4.41421 -4.84776 -5.0 -4.84776 -4.41421 -3.76537 -3.0 -2.23463 -1.58579 -1.15224
         -4.0 -3.65359 -4.34641 -4.0 -3.22865 -2.81823 -2.96077 -3.58958 -4.41042 -5.03923 -5.18177 -4.77135 -4.0 -3.23463 -2.58579 -2.15224 -2.0 -2.15224 -2.58579 -3.23463 -4.0 -4.76537 -5.41421 -5.84776 -6.0 -5.84776 -5.41421 -4.76537],
        [-2.6 -3.2 -3.2 -1.8 -2.08075 -2.79162 -3.6 -4.12763 -4.12763 -3.6 -2.79162 -2.08075 -1.0 -1.15224 -1.58579 -2.23463 -3.0 -3.76537 -4.41421 -4.84776 -5.0 -4.84776 -4.41421 -3.76537 -3.0 -2.23463 -1.58579 -1.15224
         4.0 4.34641 3.65359 4.0 4.77135 5.18177 5.03923 4.41042 3.58958 2.96077 2.81823 3.22865 4.0 4.76537 5.41421 5.84776 6.0 5.84776 5.41421 4.76537 4.0 3.23463 2.58579 2.15224 2.0 2.15224 2.58579 3.23463],
        [3.4 2.8 2.8 4.2 3.91925 3.20838 2.4 1.87237 1.87237 2.4 3.20838 3.91925 5.0 4.84776 4.41421 3.76537 3.0 2.23463 1.58579 1.15224 1.0 1.15224 1.58579 2.23463 3.0 3.76537 4.41421 4.84776
         -4.0 -3.65359 -4.34641 -4.0 -3.22865 -2.81823 -2.96077 -3.58958 -4.41042 -5.03923 -5.18177 -4.77135 -4.0 -3.23463 -2.58579 -2.15224 -2.0 -2.15224 -2.58579 -3.23463 -4.0 -4.76537 -5.41421 -5.84776 -6.0 -5.84776 -5.41421 -4.76537],
    ]

    expected_coords5 = [
        [2.0 1.84776 1.41421 0.765367 5.66554e-16 -0.765367 -1.41421 -1.84776 -2.0 -1.84776 -1.41421 -0.765367 -3.67394e-16 0.765367 1.41421 1.84776
         0.0 0.765367 1.41421 1.84776 2.0 1.84776 1.41421 0.765367 1.13311e-15 -0.765367 -1.41421 -1.84776 -2.0 -1.84776 -1.41421 -0.765367],
        [5.0 4.84776 4.41421 3.76537 3.0 2.23463 1.58579 1.15224 1.0 1.15224 1.58579 2.23463 3.0 3.76537 4.41421 4.84776
         4.0 4.76537 5.41421 5.84776 6.0 5.84776 5.41421 4.76537 4.0 3.23463 2.58579 2.15224 2.0 2.15224 2.58579 3.23463],
        [-1.0 -1.15224 -1.58579 -2.23463 -3.0 -3.76537 -4.41421 -4.84776 -5.0 -4.84776 -4.41421 -3.76537 -3.0 -2.23463 -1.58579 -1.15224
         -4.0 -3.23463 -2.58579 -2.15224 -2.0 -2.15224 -2.58579 -3.23463 -4.0 -4.76537 -5.41421 -5.84776 -6.0 -5.84776 -5.41421 -4.76537],
        [-1.0 -1.15224 -1.58579 -2.23463 -3.0 -3.76537 -4.41421 -4.84776 -5.0 -4.84776 -4.41421 -3.76537 -3.0 -2.23463 -1.58579 -1.15224
         4.0 4.76537 5.41421 5.84776 6.0 5.84776 5.41421 4.76537 4.0 3.23463 2.58579 2.15224 2.0 2.15224 2.58579 3.23463],
        [5.0 4.84776 4.41421 3.76537 3.0 2.23463 1.58579 1.15224 1.0 1.15224 1.58579 2.23463 3.0 3.76537 4.41421 4.84776
         -4.0 -3.23463 -2.58579 -2.15224 -2.0 -2.15224 -2.58579 -3.23463 -4.0 -4.76537 -5.41421 -5.84776 -6.0 -5.84776 -5.41421 -4.76537],
    ]
    @testset "Position $i" for i in eachindex(position)

        # without recess
        shape1 = CircularShape(particle_spacing, radius, (position[i][1], position[i][2]))

        # with recess
        shape2 = CircularShape(particle_spacing, radius, (position[i][1], position[i][2]),
                               shape_type=FillCircle(x_recess=(position[i][1],
                                                               position[i][1] + radius),
                                                     y_recess=(position[i][2],
                                                               position[i][2] +
                                                               particle_spacing / 2)))

        # Circumference with multiple layers
        shape3 = CircularShape(particle_spacing, 0.8, (position[i][1], position[i][2]),
                               shape_type=DrawCircle(n_layers=2))

        # Circumference with multiple layers inwards
        shape4 = CircularShape(particle_spacing, radius, (position[i][1], position[i][2]),
                               shape_type=DrawCircle(n_layers=3, layer_inwards=true))

        # Circumference one layer
        shape5 = CircularShape(particle_spacing, radius, (position[i][1], position[i][2]),
                               shape_type=DrawCircle())

        @test shape1.coordinates == expected_coords1[i]
        @test shape2.coordinates == expected_coords2[i]
        @test isapprox(shape3.coordinates, expected_coords3[i]; atol=1e-4)
        @test isapprox(shape4.coordinates, expected_coords4[i]; atol=1e-4)
        @test isapprox(shape5.coordinates, expected_coords5[i]; atol=1e-4)
    end
end
