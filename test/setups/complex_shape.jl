@testset verbose=true "Complex Shapes 2D" begin
    @testset verbose=true "Rectangular Shifted" begin
        point_in_shape_algorithms = [WindingNumberHorman(), WindingNumberJacobson()]
        shifts = [-0.5, 0.0, 0.5]
        particle_spacings = [0.03, 0.05]

        test_name(algorithm, shift, particle_spacing) = "Algorithm: $(TrixiParticles.type2string(algorithm))" *
                                                        ", Shift: $shift" *
                                                        ", Particle Spacing: $particle_spacing"
        @testset verbose=true "$(test_name(point_in_shape_algorithm, shift,
        particle_spacing))" for point_in_shape_algorithm in point_in_shape_algorithms,
                                shift in shifts,
                                particle_spacing in particle_spacings
            points_rectangular = [[0.0, 0.0] [1.0, 0.0] [1.0, 0.5] [0.0, 0.5]] .+ shift

            shape = TrixiParticles.Polygon(points_rectangular)

            seed = points_rectangular[:, 1] .- 0.5particle_spacing
            shape_sampled = ComplexShape(shape; particle_spacing, density=1.0,
                                         point_in_shape_algorithm, seed)

            min_corner = points_rectangular[:, 1] .+ 0.5particle_spacing
            max_corner = points_rectangular[:, 3]

            ranges_x = min_corner[1]:particle_spacing:max_corner[1]
            ranges_y = min_corner[2]:particle_spacing:max_corner[2]

            coords = hcat(collect.(Iterators.product(ranges_x, ranges_y))...)

            @test shape_sampled.coordinates == coords
        end
    end
end
