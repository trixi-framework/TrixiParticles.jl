@testset verbose=true "SPH Interpolation" begin
    function compare_interpolation_result(actual, expected; tolerance=1e-8)
        @test length(actual.density) == length(expected.density)
        for i in 1:length(expected.density)
            @test actual.neighbor_count[i] == expected.neighbor_count[i]
            @test actual.coord[i] == expected.coord[i]
            @test isapprox(actual.density[i], expected.density[i], atol=tolerance)
            @test isapprox(actual.velocity[i], expected.velocity[i], atol=tolerance)
            @test isapprox(actual.pressure[i], expected.pressure[i], atol=tolerance)
        end
    end

    function binary_search_outside(start, end_, func=nothing; tolerance=1e-5)
        original_start = start
        direction = start <= end_ ? 1 : -1

        while abs(end_ - start) > tolerance
            mid = (start + end_) / 2
            result = func(mid)

            if result.neighbor_count == 0
                end_ = mid
            else
                start = mid
            end
        end

        mid_point = (start + end_) / 2

        if func(mid_point).neighbor_count > 0
            if func(start).neighbor_count == 0
                mid_point = start
            else
                mid_point = end_
            end
        end

        return direction > 0 ? mid_point - original_start : original_start - mid_point
    end

    @testset verbose=true "2D" begin
        function set_values(coord)
            mass = 1.0
            density = 666
            pressure = 2 * coord[2]
            velocity = [5 + 3 * coord[1], 0.1 * coord[2]^2 + 0.1]

            return (mass, density, pressure, velocity)
        end

        function set_values_bnd(coord)
            coord[2] = 0.0
            return set_values(coord)
        end

        function expected_result(wall_distance, max_wallheight, neighbor_count)
            if wall_distance > max_wallheight
                wall_distance = max_wallheight
            end

            const_density = 666.0
            const_pressure = 2 * wall_distance
            const_velocity = [5, 0.1 * wall_distance^2 + 0.1]

            return (density=const_density,
                    neighbor_count=neighbor_count,
                    coord=[0.0, wall_distance],
                    velocity=const_velocity,
                    pressure=const_pressure)
        end

        nx = 10
        ny = 10
        bnd_nx = nx
        bnd_ny = 3
        particle_spacing = 0.2
        wall_height = ny * particle_spacing - 0.5 * particle_spacing
        smoothing_length = 1.2 * 0.5 * particle_spacing
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        sound_speed = 10 * sqrt(9.81 * 0.9)

        state_equation = StateEquationCole(; sound_speed, reference_density=1000.0,
                                           exponent=7, clip_negative_pressure=false)

        fluid = rectangular_patch(particle_spacing, (nx, ny), seed=1,
                                  perturbation_factor=0.0, perturbation_factor_position=0.0,
                                  set_function=set_values,
                                  offset=[0.0, ny * 0.5 * particle_spacing])
        bnd = rectangular_patch(particle_spacing, (bnd_nx, bnd_ny), seed=1,
                                perturbation_factor=0.0, perturbation_factor_position=0.0,
                                set_function=set_values_bnd,
                                offset=[0.0, -bnd_ny * 0.5 * particle_spacing])

        viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

        fluid_system = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length, viscosity=viscosity,
                                                   acceleration=(0.0, -9.81))

        boundary_model = BoundaryModelDummyParticles(bnd.density, bnd.mass,
                                                     state_equation=state_equation,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length)

        boundary_system = BoundarySPHSystem(bnd, boundary_model)

        # Overwrite `system.pressure` because we skip the update step
        fluid_system.pressure .= fluid.pressure

        u_no_bnd = fluid.coordinates
        # Density is integrated with `ContinuityDensity`
        v_no_bnd = vcat(fluid.velocity, fluid.density')

        sol_no_boundary = (; u=[(; x=(v_no_bnd, u_no_bnd))])

        u_bnd = hcat(fluid.coordinates, bnd.coordinates)
        v_bnd_velocity = hcat(fluid.velocity, bnd.velocity)
        v_bnd_density = vcat(fluid.density, bnd.density)

        v_bnd = vcat(v_bnd_velocity, v_bnd_density')

        sol_boundary = (; u=[(; x=(v_bnd, u_bnd))])

        semi_no_boundary = Semidiscretization(fluid_system,
                                              neighborhood_search=GridNeighborhoodSearch)
        semi_boundary = Semidiscretization(fluid_system, boundary_system,
                                           neighborhood_search=GridNeighborhoodSearch)

        # some simple results
        expected_zero(y) = (density=0.0, neighbor_count=0, coord=[0.0, y],
                            velocity=[0.0, 0.0],
                            pressure=0.0)

        for cut_off_bnd in [true, false]
            @testset verbose=true "Interpolation Point no boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_walldistance(y) = TrixiParticles.interpolate_point([0.0, y],
                                                                                 semi_no_boundary,
                                                                                 fluid_system,
                                                                                 sol_no_boundary,
                                                                                 cut_off_bnd=cut_off_bnd)

                # top outside
                distance_top_outside = binary_search_outside(ny * particle_spacing,
                                                             (ny + 2) * particle_spacing,
                                                             interpolation_walldistance)
                @test isapprox(distance_top_outside, 0.11817321777343714, atol=1e-14)

                result_zero = interpolation_walldistance(ny * particle_spacing +
                                                         distance_top_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(ny * particle_spacing +
                                                           distance_top_outside))

                result_top_outside = interpolation_walldistance(ny * particle_spacing +
                                                                0.5 * distance_top_outside)
                compare_interpolation_result(result_top_outside,
                                             expected_result(ny * particle_spacing +
                                                             0.5 * distance_top_outside,
                                                             wall_height, 2))

                # top at free surface
                result_top = interpolation_walldistance(ny * particle_spacing)
                compare_interpolation_result(result_top,
                                             expected_result(ny * particle_spacing,
                                                             wall_height,
                                                             2))

                # center
                result_center = interpolation_walldistance(ny * 0.5 * particle_spacing)
                compare_interpolation_result(result_center,
                                             expected_result(ny * 0.5 * particle_spacing,
                                                             wall_height, 5))

                # at wall
                exp_res = (density=665.9999999999999,
                           neighbor_count=2,
                           coord=[0.0, 0.0],
                           velocity=[5.0, 0.16363759578999118],
                           pressure=0.2)
                result_bottom = interpolation_walldistance(0.0)
                compare_interpolation_result(result_bottom, exp_res)

                distance_bottom_outside = binary_search_outside(0.0, -2 * particle_spacing,
                                                                interpolation_walldistance)
                @test isapprox(distance_bottom_outside, 0.1181732177734375, atol=1e-14)

                exp_res = (density=666.0,
                           neighbor_count=2,
                           coord=[0.0, -0.5 * distance_bottom_outside],
                           velocity=[5.0, 0.18100000000000002],
                           pressure=0.2)
                result_bottom_outside = interpolation_walldistance(-0.5 *
                                                                   distance_bottom_outside)
                compare_interpolation_result(result_bottom_outside, exp_res)

                result_zero = interpolation_walldistance(-distance_bottom_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(-distance_bottom_outside))

                multi_point_coords = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]

                result_multipoint = TrixiParticles.interpolate_point(multi_point_coords,
                                                                     semi_no_boundary,
                                                                     fluid_system,
                                                                     sol_no_boundary,
                                                                     cut_off_bnd=cut_off_bnd)

                expected_multi = (density=[666.0, 666.0000000000001, 666.0],
                                  neighbor_count=[2, 6, 5],
                                  coord=[[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
                                  velocity=[
                                      [5.0, 0.10100000000000002],
                                      [5.000000000000001, 0.12501295337729817],
                                      [5.0, 0.20035665520692278],
                                  ],
                                  pressure=[0.19999999999999996, 1.0000000000000002, 2.0])

                compare_interpolation_result(result_multipoint, expected_multi)
            end
            @testset verbose=true "Interpolation Line no boundary - cut_off_bnd = $(cut_off_bnd)" begin
                result_endpoint = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0],
                                                                  5, semi_no_boundary,
                                                                  fluid_system,
                                                                  sol_no_boundary,
                                                                  endpoint=true,
                                                                  cut_off_bnd=cut_off_bnd)

                result = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0],
                                                         5, semi_no_boundary,
                                                         fluid_system, sol_no_boundary,
                                                         endpoint=false,
                                                         cut_off_bnd=cut_off_bnd)

                expected_res = (density=[666.0, 666.0, 666.0], neighbor_count=[2, 2, 1],
                                coord=SVector{2, Float64}[[1.0, 0.2125],
                                                          [1.0, 0.47500000000000003],
                                                          [1.0, 0.7375]],
                                velocity=SVector{2, Float64}[[
                                                                 7.699999999999999,
                                                                 0.10605429538320173,
                                                             ], [7.7, 0.12465095587703466],
                                                             [7.7, 0.14900000000000002]],
                                pressure=[
                                    0.4527147691600855,
                                    0.9912738969258665,
                                    1.4000000000000001,
                                ])
                expected_res_end = (density=[666.0, 666.0, 666.0, 666.0, 666.0],
                                    neighbor_count=[1, 2, 2, 1, 1],
                                    coord=SVector{2, Float64}[[1.0, -0.05], [1.0, 0.2125],
                                                              [1.0, 0.475], [1.0, 0.7375],
                                                              [1.0, 1.0]],
                                    velocity=SVector{2, Float64}[[7.7, 0.10099999999999999],
                                                                 [
                                                                     7.699999999999999,
                                                                     0.10605429538320173,
                                                                 ],
                                                                 [
                                                                     7.699999999999999,
                                                                     0.12465095587703465,
                                                                 ],
                                                                 [7.7, 0.14900000000000002],
                                                                 [7.7, 0.22100000000000006]],
                                    pressure=[
                                        0.19999999999999998,
                                        0.4527147691600855,
                                        0.9912738969258663,
                                        1.4000000000000001,
                                        2.2,
                                    ])

                compare_interpolation_result(result, expected_res)
                compare_interpolation_result(result_endpoint, expected_res_end)
            end
            @testset verbose=true "Interpolation Plane no boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_start = [0.0, 0.0]
                interpolation_end = [1.0, 1.0]
                resolution = 0.25

                result = interpolate_plane_2d(interpolation_start, interpolation_end,
                                              resolution, semi_no_boundary,
                                              fluid_system, sol_no_boundary)

                expected_res = (density=[
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0,
                                ],
                                neighbor_count=[
                                    2, 2, 3, 2, 1, 4, 4, 4, 4, 2, 6, 4, 5, 4, 3, 4, 4, 4,
                                    3, 1, 5, 4, 6, 3, 1,
                                ],
                                coord=SVector{2, Float64}[[0.0, 0.0], [0.25, 0.0],
                                                          [0.5, 0.0], [0.75, 0.0],
                                                          [1.0, 0.0], [0.0, 0.25],
                                                          [0.25, 0.25], [0.5, 0.25],
                                                          [0.75, 0.25], [1.0, 0.25],
                                                          [0.0, 0.5], [0.25, 0.5],
                                                          [0.5, 0.5], [0.75, 0.5],
                                                          [1.0, 0.5], [0.0, 0.75],
                                                          [0.25, 0.75], [0.5, 0.75],
                                                          [0.75, 0.75], [1.0, 0.75],
                                                          [0.0, 1.0], [0.25, 1.0],
                                                          [0.5, 1.0], [0.75, 1.0],
                                                          [1.0, 1.0]],
                                velocity=SVector{2, Float64}[[5.0, 0.10100000000000002],
                                                             [
                                                                 5.844853603211259,
                                                                 0.10099999999999999,
                                                             ], [6.5, 0.101],
                                                             [7.155146396788742, 0.101],
                                                             [7.7, 0.101],
                                                             [
                                                                 5.000000000000001,
                                                                 0.10826471470948347,
                                                             ],
                                                             [
                                                                 5.8376544066143845,
                                                                 0.10816872542152514,
                                                             ],
                                                             [
                                                                 6.499999999999999,
                                                                 0.10807333490890002,
                                                             ],
                                                             [
                                                                 7.162345593385616,
                                                                 0.10816872542152513,
                                                             ],
                                                             [7.7, 0.10826471470948347],
                                                             [
                                                                 5.000000000000001,
                                                                 0.12501295337729817,
                                                             ],
                                                             [
                                                                 5.8305001181675005,
                                                                 0.12504927969391108,
                                                             ],
                                                             [
                                                                 6.499999999999998,
                                                                 0.12507142857142856,
                                                             ],
                                                             [
                                                                 7.1694998818325,
                                                                 0.12504927969391108,
                                                             ],
                                                             [
                                                                 7.700000000000002,
                                                                 0.12501295337729815,
                                                             ],
                                                             [
                                                                 4.999999999999999,
                                                                 0.15194114116206617,
                                                             ],
                                                             [
                                                                 5.837654406614385,
                                                                 0.15232509831389957,
                                                             ],
                                                             [
                                                                 6.500000000000001,
                                                                 0.15270666036440003,
                                                             ],
                                                             [
                                                                 7.160218593182242,
                                                                 0.1522116583030529,
                                                             ],
                                                             [7.7, 0.14900000000000002],
                                                             [5.0, 0.20035665520692278],
                                                             [5.84485360321126, 0.201],
                                                             [6.5, 0.20100000000000004],
                                                             [
                                                                 7.128901370428878,
                                                                 0.2019633790142959,
                                                             ],
                                                             [7.7, 0.22100000000000006]],
                                pressure=[
                                    0.19999999999999996,
                                    0.19999999999999996,
                                    0.19999999999999996,
                                    0.19999999999999996,
                                    0.19999999999999996,
                                    0.5632357354741733,
                                    0.5584362710762559,
                                    0.5536667454449997,
                                    0.5584362710762559,
                                    0.5632357354741733,
                                    1.0000000000000002,
                                    1.0, 0.9999999999999999,
                                    0.9999999999999999,
                                    1.0, 1.436764264525827,
                                    1.4415637289237442,
                                    1.4463332545550007,
                                    1.4401457287881612,
                                    1.4000000000000001,
                                    2.0, 2.0,
                                    2.0, 2.009633790142959, 2.2,
                                ])

                compare_interpolation_result(result, expected_res)

                result = interpolate_plane_2d(interpolation_start, interpolation_end,
                                              resolution, semi_no_boundary, fluid_system,
                                              sol_no_boundary,
                                              smoothing_length=0.5 * smoothing_length)
                expected_res = (density=[
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                ],
                                neighbor_count=[
                                    1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                                    2, 2, 2,
                                ],
                                coord=SVector{2, Float64}[[0.25, 0.0], [0.5, 0.0],
                                                          [0.75, 0.0], [0.0, 0.25],
                                                          [0.25, 0.25], [0.5, 0.25],
                                                          [0.75, 0.25], [1.0, 0.25],
                                                          [0.0, 0.5], [0.25, 0.5],
                                                          [0.5, 0.5], [0.75, 0.5],
                                                          [1.0, 0.5], [0.0, 0.75],
                                                          [0.25, 0.75], [0.5, 0.75],
                                                          [0.75, 0.75], [1.0, 0.75],
                                                          [0.0, 1.0], [0.25, 1.0],
                                                          [0.5, 1.0], [0.75, 1.0]],
                                velocity=SVector{2, Float64}[[5.9, 0.101],
                                                             [6.499999999999999, 0.101],
                                                             [7.1000000000000005, 0.101],
                                                             [
                                                                 4.999999999999994,
                                                                 0.10900000000000003,
                                                             ], [5.9, 0.10900000000000001],
                                                             [6.5, 0.109],
                                                             [
                                                                 7.1000000000000005,
                                                                 0.10900000000000003,
                                                             ], [7.7, 0.10900000000000001],
                                                             [4.999999999999998, 0.125],
                                                             [5.9, 0.125], [6.5, 0.125],
                                                             [7.1, 0.125], [7.7, 0.125],
                                                             [
                                                                 4.999999999999995,
                                                                 0.14900000000000002,
                                                             ],
                                                             [
                                                                 5.900000000000001,
                                                                 0.14900000000000002,
                                                             ], [6.5, 0.14900000000000002],
                                                             [
                                                                 7.1000000000000005,
                                                                 0.14900000000000002,
                                                             ], [7.7, 0.14900000000000002],
                                                             [5.0, 0.2],
                                                             [
                                                                 5.8999999999999995,
                                                                 0.20099999999999962,
                                                             ], [6.5, 0.20099999999999985],
                                                             [
                                                                 7.100000000000001,
                                                                 0.20099999999999968,
                                                             ]],
                                pressure=[
                                    0.19999999999999996,
                                    0.19999999999999996,
                                    0.19999999999999996,
                                    0.6000000000000001,
                                    0.6000000000000001,
                                    0.6000000000000001,
                                    0.6000000000000001,
                                    0.6000000000000001,
                                    1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.4000000000000004,
                                    1.4000000000000001,
                                    1.4000000000000001,
                                    1.4000000000000001,
                                    1.4000000000000001,
                                    2.0, 1.9999999999999962,
                                    1.9999999999999984,
                                    1.9999999999999967,
                                ])

                compare_interpolation_result(result, expected_res)
            end
        end

        for cut_off_bnd in [true, false]
            @testset verbose=true "Interpolation Point boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_walldistance(y) = TrixiParticles.interpolate_point([0.0, y],
                                                                                 semi_boundary,
                                                                                 fluid_system,
                                                                                 sol_boundary,
                                                                                 cut_off_bnd=cut_off_bnd)

                # top outside
                distance_top_outside = binary_search_outside(ny * particle_spacing,
                                                             (ny + 2) * particle_spacing,
                                                             interpolation_walldistance)
                @test isapprox(distance_top_outside, 0.11817321777343714, atol=1e-14)

                result_zero = interpolation_walldistance(ny * particle_spacing +
                                                         distance_top_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(ny * particle_spacing +
                                                           distance_top_outside))

                result_top_outside = interpolation_walldistance(ny * particle_spacing +
                                                                0.5 * distance_top_outside)
                compare_interpolation_result(result_top_outside,
                                             expected_result(ny * particle_spacing +
                                                             0.5 * distance_top_outside,
                                                             wall_height, 2))

                # top at free surface
                result_top = interpolation_walldistance(ny * particle_spacing)
                compare_interpolation_result(result_top,
                                             expected_result(ny * particle_spacing,
                                                             wall_height,
                                                             2))

                # center
                result_center = interpolation_walldistance(ny * 0.5 * particle_spacing)
                compare_interpolation_result(result_center,
                                             expected_result(ny * 0.5 * particle_spacing,
                                                             wall_height, 5))

                # at wall
                result_bottom = interpolation_walldistance(0.0)
                if cut_off_bnd
                    exp_res = (density=666,
                               neighbor_count=4,
                               coord=[0.0, 0.0],
                               velocity=[5.0, 0.16363759578999118],
                               pressure=0.2)
                    compare_interpolation_result(result_bottom, exp_res)
                else
                    exp_res = (density=666,
                               neighbor_count=2,
                               coord=[0.0, 0.0],
                               velocity=[5.0, 0.16363759578999118],
                               pressure=0.2)
                    compare_interpolation_result(result_bottom, exp_res)
                end

                distance_bottom_outside = binary_search_outside(0.0, -2 * particle_spacing,
                                                                interpolation_walldistance,
                                                                tolerance=1e-12)
                if cut_off_bnd
                    @test isapprox(distance_bottom_outside, 3.637978807091713e-13,
                                   atol=1e-14)
                else
                    @test isapprox(distance_bottom_outside, 0.11817145975473978, atol=1e-14)
                end

                result_bottom_outside = interpolation_walldistance(-0.5 *
                                                                   distance_bottom_outside)

                if cut_off_bnd
                    compare_interpolation_result(result_bottom_outside,
                                                 expected_zero(-0.5 *
                                                               distance_bottom_outside))
                else
                    exp_res = (density=666,
                               neighbor_count=2,
                               coord=[0.0, -0.5 * distance_bottom_outside],
                               velocity=[5.0, 0.16363759578999118],
                               pressure=0.2)
                    compare_interpolation_result(result_bottom_outside, exp_res)
                end

                result_zero = interpolation_walldistance(-distance_bottom_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(-distance_bottom_outside))

                multi_point_coords = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]

                result_multipoint = TrixiParticles.interpolate_point(multi_point_coords,
                                                                     semi_boundary,
                                                                     fluid_system,
                                                                     sol_boundary,
                                                                     cut_off_bnd=cut_off_bnd)
                if cut_off_bnd
                    expected_multi = (density=[666.0, 666.0000000000001, 666.0],
                                      neighbor_count=[4, 6, 5],
                                      coord=[[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
                                      velocity=[
                                          [5.0, 0.10100000000000002],
                                          [5.000000000000001, 0.12501295337729817],
                                          [5.0, 0.20035665520692278],
                                      ],
                                      pressure=[
                                          0.19999999999999996,
                                          1.0000000000000002,
                                          2.0,
                                      ])

                    compare_interpolation_result(result_multipoint, expected_multi)
                else
                    expected_multi = (density=[666.0, 666.0000000000001, 666.0],
                                      neighbor_count=[2, 6, 5],
                                      coord=[[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
                                      velocity=[
                                          [5.0, 0.10100000000000002],
                                          [5.000000000000001, 0.12501295337729817],
                                          [5.0, 0.20035665520692278],
                                      ],
                                      pressure=[
                                          0.19999999999999996,
                                          1.0000000000000002,
                                          2.0,
                                      ])

                    compare_interpolation_result(result_multipoint, expected_multi)
                end
            end
            @testset verbose=true "Interpolation Line boundary - cut_off_bnd = $(cut_off_bnd)" begin
                result_endpoint = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0],
                                                                  5, semi_boundary,
                                                                  fluid_system,
                                                                  sol_boundary,
                                                                  endpoint=true,
                                                                  cut_off_bnd=cut_off_bnd)

                result = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0],
                                                         5, semi_no_boundary,
                                                         fluid_system, sol_no_boundary,
                                                         endpoint=false,
                                                         cut_off_bnd=cut_off_bnd)
                if cut_off_bnd
                    expected_res = (density=[666.0, 666.0, 666.0], neighbor_count=[2, 2, 1],
                                    coord=SVector{2, Float64}[[1.0, 0.2125],
                                                              [1.0, 0.47500000000000003],
                                                              [1.0, 0.7375]],
                                    velocity=SVector{2, Float64}[[
                                                                     7.699999999999999,
                                                                     0.10605429538320173,
                                                                 ],
                                                                 [7.7, 0.12465095587703466],
                                                                 [7.7, 0.14900000000000002]],
                                    pressure=[
                                        0.4527147691600855,
                                        0.9912738969258665,
                                        1.4000000000000001,
                                    ])
                    expected_res_end = (density=[
                                            0.0,
                                            666.0,
                                            665.9999999999999,
                                            666.0,
                                            666.0,
                                        ],
                                        neighbor_count=[0, 2, 2, 1, 1],
                                        coord=SVector{2, Float64}[[1.0, -0.05],
                                                                  [1.0, 0.2125],
                                                                  [1.0, 0.475],
                                                                  [1.0, 0.7375],
                                                                  [1.0, 1.0]],
                                        velocity=SVector{2, Float64}[[0.0, 0.0],
                                                                     [
                                                                         7.699999999999999,
                                                                         0.10605429538320173,
                                                                     ],
                                                                     [
                                                                         7.699999999999999,
                                                                         0.12465095587703465,
                                                                     ],
                                                                     [
                                                                         7.7,
                                                                         0.14900000000000002,
                                                                     ],
                                                                     [
                                                                         7.7,
                                                                         0.22100000000000006,
                                                                     ]],
                                        pressure=[
                                            0.0,
                                            0.4527147691600855,
                                            0.9912738969258663,
                                            1.4000000000000001,
                                            2.2,
                                        ])

                    compare_interpolation_result(result, expected_res)
                    compare_interpolation_result(result_endpoint, expected_res_end)

                else
                    expected_res = (density=[666.0, 666.0, 666.0], neighbor_count=[2, 2, 1],
                                    coord=SVector{2, Float64}[[1.0, 0.2125],
                                                              [1.0, 0.47500000000000003],
                                                              [1.0, 0.7375]],
                                    velocity=SVector{2, Float64}[[
                                                                     7.699999999999999,
                                                                     0.10605429538320173,
                                                                 ],
                                                                 [7.7, 0.12465095587703466],
                                                                 [7.7, 0.14900000000000002]],
                                    pressure=[
                                        0.4527147691600855,
                                        0.9912738969258665,
                                        1.4000000000000001,
                                    ])
                    expected_res_end = (density=[
                                            666.0,
                                            666.0,
                                            665.9999999999999,
                                            666.0,
                                            666.0,
                                        ], neighbor_count=[1, 2, 2, 1, 1],
                                        coord=SVector{2, Float64}[[1.0, -0.05],
                                                                  [1.0, 0.2125],
                                                                  [1.0, 0.475],
                                                                  [1.0, 0.7375],
                                                                  [1.0, 1.0]],
                                        velocity=SVector{2, Float64}[[
                                                                         7.7,
                                                                         0.10099999999999999,
                                                                     ],
                                                                     [
                                                                         7.699999999999999,
                                                                         0.10605429538320173,
                                                                     ],
                                                                     [
                                                                         7.699999999999999,
                                                                         0.12465095587703465,
                                                                     ],
                                                                     [
                                                                         7.7,
                                                                         0.14900000000000002,
                                                                     ],
                                                                     [
                                                                         7.7,
                                                                         0.22100000000000006,
                                                                     ]],
                                        pressure=[
                                            0.19999999999999998,
                                            0.4527147691600855,
                                            0.9912738969258663,
                                            1.4000000000000001,
                                            2.2,
                                        ])

                    compare_interpolation_result(result, expected_res)
                    compare_interpolation_result(result_endpoint, expected_res_end)
                end
            end
        end
    end

    @testset verbose=true "3D" begin
        function set_values(coord)
            mass = 1.0
            density = 666
            pressure = 2 * coord[2]
            velocity = [5 + 3 * coord[1], 0.1 * coord[2]^2 + 0.1, coord[3]]

            return (mass, density, pressure, velocity)
        end

        function set_values_bnd(coord)
            coord[2] = 0.0
            return set_values(coord)
        end

        function expected_result(wall_distance, max_wallheight, neighbor_count)
            if wall_distance > max_wallheight
                wall_distance = max_wallheight
            end

            const_density = 666.0
            const_pressure = 2 * wall_distance
            const_velocity = [5, 0.1 * wall_distance^2 + 0.1, 0.0]

            return (density=const_density,
                    neighbor_count=neighbor_count,
                    coord=[0.0, wall_distance, 0.0],
                    velocity=const_velocity,
                    pressure=const_pressure)
        end

        nx = 10
        ny = 10
        nz = 10
        bnd_nx = nx
        bnd_ny = 3
        bnd_nz = nz
        particle_spacing = 0.2
        wall_height = ny * particle_spacing - 0.5 * particle_spacing
        smoothing_length = 1.2 * 0.5 * particle_spacing
        smoothing_kernel = SchoenbergCubicSplineKernel{3}()
        sound_speed = 10 * sqrt(9.81 * 0.9)

        state_equation = StateEquationCole(; sound_speed, reference_density=1000.0,
                                           exponent=7, clip_negative_pressure=false)

        fluid = rectangular_patch(particle_spacing, (nx, ny, nz), seed=1,
                                  perturbation_factor=0.0, perturbation_factor_position=0.0,
                                  set_function=set_values,
                                  offset=[0.0, ny * 0.5 * particle_spacing, 0.0])
        bnd = rectangular_patch(particle_spacing, (bnd_nx, bnd_ny, bnd_nz), seed=1,
                                perturbation_factor=0.0, perturbation_factor_position=0.0,
                                set_function=set_values_bnd,
                                offset=[0.0, -bnd_ny * 0.5 * particle_spacing, 0.0])

        viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

        fluid_system = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length, viscosity=viscosity,
                                                   acceleration=(0.0, -9.81, 0.0))

        boundary_model = BoundaryModelDummyParticles(bnd.density, bnd.mass,
                                                     state_equation=state_equation,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length)

        boundary_system = BoundarySPHSystem(bnd, boundary_model)

        # Overwrite `system.pressure` because we skip the update step
        fluid_system.pressure .= fluid.pressure

        u_no_bnd = fluid.coordinates
        # Density is integrated with `ContinuityDensity`
        v_no_bnd = vcat(fluid.velocity, fluid.density')

        sol_no_boundary = (; u=[(; x=(v_no_bnd, u_no_bnd))])

        u_bnd = hcat(fluid.coordinates, bnd.coordinates)
        v_bnd_velocity = hcat(fluid.velocity, bnd.velocity)
        v_bnd_density = vcat(fluid.density, bnd.density)

        v_bnd = vcat(v_bnd_velocity, v_bnd_density')

        sol_boundary = (; u=[(; x=(v_bnd, u_bnd))])

        semi_no_boundary = Semidiscretization(fluid_system,
                                              neighborhood_search=GridNeighborhoodSearch)
        semi_boundary = Semidiscretization(fluid_system, boundary_system,
                                           neighborhood_search=GridNeighborhoodSearch)

        # some simple results
        expected_zero(y) = (density=0.0, neighbor_count=0, coord=[0.0, y, 0.0],
                            velocity=[0.0, 0.0, 0.0], pressure=0.0)

        for cut_off_bnd in [true, false]
            @testset verbose=true "Interpolation Point no boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_walldistance(y) = TrixiParticles.interpolate_point([
                                                                                     0.0,
                                                                                     y,
                                                                                     0.0,
                                                                                 ],
                                                                                 semi_no_boundary,
                                                                                 fluid_system,
                                                                                 sol_no_boundary,
                                                                                 cut_off_bnd=cut_off_bnd)

                # top outside
                distance_top_outside = binary_search_outside(ny * particle_spacing,
                                                             (ny + 2) * particle_spacing,
                                                             interpolation_walldistance,
                                                             tolerance=1e-8)
                @test isapprox(distance_top_outside, 0.09390581548213905, atol=1e-14)

                result_zero = interpolation_walldistance(ny * particle_spacing +
                                                         distance_top_outside)

                compare_interpolation_result(result_zero,
                                             expected_zero(ny * particle_spacing +
                                                           distance_top_outside))

                result_top_outside = interpolation_walldistance(ny * particle_spacing +
                                                                0.5 * distance_top_outside)
                compare_interpolation_result(result_top_outside,
                                             expected_result(ny * particle_spacing +
                                                             0.5 * distance_top_outside,
                                                             wall_height, 4))

                # top at free surface
                result_top = interpolation_walldistance(ny * particle_spacing)
                compare_interpolation_result(result_top,
                                             expected_result(ny * particle_spacing,
                                                             wall_height,
                                                             4))

                # center
                result_center = interpolation_walldistance(ny * 0.5 * particle_spacing)
                compare_interpolation_result(result_center,
                                             expected_result(ny * 0.5 * particle_spacing,
                                                             wall_height, 9))

                # at wall
                exp_res = (density=665.9999999999999,
                           neighbor_count=4,
                           coord=[0.0, 0.0],
                           velocity=[5.0, 0.16363759578999118],
                           pressure=0.2)
                result_bottom = interpolation_walldistance(0.0)
                compare_interpolation_result(result_bottom, exp_res)

                distance_bottom_outside = binary_search_outside(0.0, -2 * particle_spacing,
                                                                interpolation_walldistance,
                                                                tolerance=1e-12)
                @test isapprox(distance_bottom_outside, 0.09390581390689477, atol=1e-14)

                exp_res = (density=666.0,
                           neighbor_count=4,
                           coord=[0.0, -0.5 * distance_bottom_outside],
                           velocity=[5.0, 0.18100000000000002],
                           pressure=0.2)
                result_bottom_outside = interpolation_walldistance(-0.5 *
                                                                   distance_bottom_outside)
                compare_interpolation_result(result_bottom_outside, exp_res)

                result_zero = interpolation_walldistance(-distance_bottom_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(-distance_bottom_outside))

                multi_point_coords = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]]

                result_multipoint = TrixiParticles.interpolate_point(multi_point_coords,
                                                                     semi_no_boundary,
                                                                     fluid_system,
                                                                     sol_no_boundary,
                                                                     cut_off_bnd=cut_off_bnd)

                expected_multi = (density=[666.0, 666.0, 666.0], neighbor_count=[4, 4, 9],
                                  coord=[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
                                  velocity=[
                                      [5.0, 0.101, 0.0],
                                      [5.0, 0.125, 0.0],
                                      [5.0, 0.20025646054622268, 0.0],
                                  ], pressure=[0.19999999999999996, 1.0, 2.0])

                compare_interpolation_result(result_multipoint, expected_multi)
            end
        end

        for cut_off_bnd in [true, false]
            @testset verbose=true "Interpolation Point boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_walldistance(y) = TrixiParticles.interpolate_point([
                                                                                     0.0,
                                                                                     y,
                                                                                     0.0,
                                                                                 ],
                                                                                 semi_boundary,
                                                                                 fluid_system,
                                                                                 sol_boundary,
                                                                                 cut_off_bnd=cut_off_bnd)

                # top outside
                distance_top_outside = binary_search_outside(ny * particle_spacing,
                                                             (ny + 2) * particle_spacing,
                                                             interpolation_walldistance)
                @test isapprox(distance_top_outside, 0.09390869140624947, atol=1e-14)

                result_zero = interpolation_walldistance(ny * particle_spacing +
                                                         distance_top_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(ny * particle_spacing +
                                                           distance_top_outside))

                result_top_outside = interpolation_walldistance(ny * particle_spacing +
                                                                0.5 * distance_top_outside)
                compare_interpolation_result(result_top_outside,
                                             expected_result(ny * particle_spacing +
                                                             0.5 * distance_top_outside,
                                                             wall_height, 4))

                # top at free surface
                result_top = interpolation_walldistance(ny * particle_spacing)
                compare_interpolation_result(result_top,
                                             expected_result(ny * particle_spacing,
                                                             wall_height, 4))

                # center
                result_center = interpolation_walldistance(ny * 0.5 * particle_spacing)
                compare_interpolation_result(result_center,
                                             expected_result(ny * 0.5 * particle_spacing,
                                                             wall_height, 9))

                # at wall
                result_bottom = interpolation_walldistance(0.0)
                if cut_off_bnd
                    exp_res = (density=666,
                               neighbor_count=8,
                               coord=[0.0, 0.0],
                               velocity=[5.0, 0.16363759578999118],
                               pressure=0.2)
                    compare_interpolation_result(result_bottom, exp_res)
                else
                    exp_res = (density=666,
                               neighbor_count=4,
                               coord=[0.0, 0.0],
                               velocity=[5.0, 0.16363759578999118],
                               pressure=0.2)
                    compare_interpolation_result(result_bottom, exp_res)
                end

                distance_bottom_outside = binary_search_outside(0.0, -2 * particle_spacing,
                                                                interpolation_walldistance,
                                                                tolerance=1e-12)
                if cut_off_bnd
                    @test isapprox(distance_bottom_outside, 3.637978807091713e-13,
                                   atol=1e-14)
                else
                    @test isapprox(distance_bottom_outside, 0.09390581390689477, atol=1e-14)
                end

                result_bottom_outside = interpolation_walldistance(-0.5 *
                                                                   distance_bottom_outside)

                if cut_off_bnd
                    compare_interpolation_result(result_bottom_outside,
                                                 expected_zero(-0.5 *
                                                               distance_bottom_outside))
                else
                    exp_res = (density=666,
                               neighbor_count=4,
                               coord=[0.0, -0.5 * distance_bottom_outside],
                               velocity=[5.0, 0.16363759578999118],
                               pressure=0.2)
                    compare_interpolation_result(result_bottom_outside, exp_res)
                end

                result_zero = interpolation_walldistance(-distance_bottom_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(-distance_bottom_outside))

                multi_point_coords = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]]

                result_multipoint = TrixiParticles.interpolate_point(multi_point_coords,
                                                                     semi_no_boundary,
                                                                     fluid_system,
                                                                     sol_no_boundary,
                                                                     cut_off_bnd=cut_off_bnd)

                expected_multi = (density=[666.0, 666.0, 666.0], neighbor_count=[4, 4, 9],
                                  coord=[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
                                  velocity=[
                                      [5.0, 0.101, 0.0],
                                      [5.0, 0.125, 0.0],
                                      [5.0, 0.20025646054622268, 0.0],
                                  ], pressure=[0.19999999999999996, 1.0, 2.0])

                compare_interpolation_result(result_multipoint, expected_multi)
            end

            @testset verbose=true "Interpolation Plane no boundary - cut_off_bnd = $(cut_off_bnd)" begin
                p1 = [0.0, 0.0, 0.0]
                p2 = [1.0, 1.0, 0.0]
                p3 = [0.0, 1.0, 0.0]
                resolution = 0.5

                result = interpolate_plane_3d(p1, p2, p3,
                                              resolution, semi_no_boundary,
                                              fluid_system, sol_no_boundary)

                expected_res = (density=[
                                    666.0,
                                    666.0,
                                    666.0,
                                    666.0,
                                    666.0,
                                    666.0,
                                    666.0,
                                    666.0,
                                    666.0,
                                    666.0,
                                    666.0,
                                    666.0,
                                ], neighbor_count=[4, 4, 9, 6, 8, 6, 6, 8, 6, 4, 2, 1],
                                coord=SVector{3, Float64}[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0],
                                                          [0.0, 1.0, 0.0],
                                                          [
                                                              0.3333333333333333,
                                                              0.3333333333333333,
                                                              0.0,
                                                          ],
                                                          [
                                                              0.3333333333333333,
                                                              0.8333333333333333,
                                                              0.0,
                                                          ],
                                                          [
                                                              0.3333333333333333,
                                                              1.3333333333333333,
                                                              0.0,
                                                          ],
                                                          [
                                                              0.6666666666666666,
                                                              0.6666666666666666,
                                                              0.0,
                                                          ],
                                                          [
                                                              0.6666666666666666,
                                                              1.1666666666666665,
                                                              0.0,
                                                          ],
                                                          [
                                                              0.6666666666666666,
                                                              1.6666666666666665,
                                                              0.0,
                                                          ], [1.0, 1.0, 0.0],
                                                          [1.0, 1.5, 0.0], [1.0, 2.0, 0.0]],
                                velocity=SVector{3, Float64}[[
                                                                 5.0,
                                                                 0.101,
                                                                 -8.637243445401583e-17,
                                                             ],
                                                             [
                                                                 5.0,
                                                                 0.125,
                                                                 -7.070877847937661e-17,
                                                             ],
                                                             [
                                                                 5.0,
                                                                 0.20025646054622268,
                                                                 -2.2734045974413372e-17,
                                                             ],
                                                             [
                                                                 5.920251592989799,
                                                                 0.10954004247972798,
                                                                 -7.466540216672048e-17,
                                                             ],
                                                             [
                                                                 5.912901655029027,
                                                                 0.1752639260703962,
                                                                 -8.888366469079487e-17,
                                                             ],
                                                             [
                                                                 5.920251592989798,
                                                                 0.2708901486790478,
                                                                 -5.308243435290283e-17,
                                                             ],
                                                             [
                                                                 7.079748407010201,
                                                                 0.14818993628040808,
                                                                 -7.466540216672064e-17,
                                                             ],
                                                             [
                                                                 7.087098344970975,
                                                                 0.22960411089440566,
                                                                 -8.19920883032382e-17,
                                                             ],
                                                             [
                                                                 7.079748407010202,
                                                                 0.38683983008108835,
                                                                 -1.119981032500812e-16,
                                                             ],
                                                             [
                                                                 7.699999999999998,
                                                                 0.20099999999999996,
                                                                 -1.2273977527675892e-16,
                                                             ],
                                                             [
                                                                 7.700000000000001,
                                                                 0.32500000000000007,
                                                                 -6.222372506185122e-17,
                                                             ],
                                                             [
                                                                 7.699999999999999,
                                                                 0.4610000000000001,
                                                                 0.10000000000000009,
                                                             ]],
                                pressure=[
                                    0.19999999999999996,
                                    1.0,
                                    2.0,
                                    0.6135010619931992,
                                    1.728299075879952,
                                    2.6135010619931984,
                                    1.386498938006801,
                                    2.2717009241200476,
                                    3.386498938006801,
                                    1.9999999999999998,
                                    3.0000000000000004,
                                    3.8000000000000003,
                                ])

                compare_interpolation_result(result, expected_res)
            end
        end
    end
end
