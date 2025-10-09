@testset verbose=true "SPH Interpolation" begin
    function compare_interpolation_result(actual, expected; tolerance=5e-4)
        @test length(actual.density) == length(expected.density)
        for i in 1:length(expected.density)
            @test actual.neighbor_count[i] == expected.neighbor_count[i]
            @test actual.point_coords[:, i] == expected.point_coords[:, i]
            if !(isnan(actual.density[i]) && isnan(expected.density[i]))
                @test isapprox(actual.density[i], expected.density[i], atol=tolerance)
            end
            if !(isnan(actual.pressure[i]) && isnan(expected.pressure[i]))
                @test isapprox(actual.pressure[i], expected.pressure[i], atol=tolerance)
            end
            if !(all(isnan.(actual.velocity[:, i])) && all(isnan.(expected.velocity[:, i])))
                @test isapprox(actual.velocity[:, i], expected.velocity[:, i],
                               atol=tolerance)
            end
        end
    end

    # Binary search to find the distance from `inside` to the point where
    # the `neighbor_count` becomes zero.
    # `inside` must be inside the fluid, `outside` must be outside the fluid.
    function binary_search_outside(inside, outside, func=nothing; tolerance=1e-5)
        original_inside = inside

        # Bisect to find the point where the interpolated fluid ends (`neighbor_count` == 0)
        while abs(outside - inside) > tolerance
            mid = (inside + outside) / 2
            result = func(mid)

            if result.neighbor_count[1] == 0
                # `mid` is outside the fluid, so it becomes the new `outside`
                outside = mid
            else
                # `mid` is inside the fluid, so it becomes the new `inside`
                inside = mid
            end
        end

        return abs(outside - original_inside)
    end

    @testset verbose=true "2D" begin
        function set_values(coord)
            mass = 1.0
            density = 666
            pressure = 2 * coord[2]
            velocity = [5+3 * coord[1]; 0.1 * coord[2]^2+0.1;;]

            return (mass, density, pressure, velocity)
        end

        function set_values_bnd(coord)
            coord[2] = 0.0
            return set_values(coord)
        end

        function expected_result(wall_distance, max_wallheight, neighbor_count)
            # Interpolated values above the last particle but within the interpolation
            # cutoff return the values of the last particle.
            new_wall_distance = wall_distance
            if wall_distance > max_wallheight
                new_wall_distance = max_wallheight
            end

            const_density = [666.0]
            const_pressure = [2 * new_wall_distance]
            const_velocity = [5; 0.1 * new_wall_distance^2+0.1;;]

            return (density=const_density,
                    neighbor_count=[neighbor_count],
                    point_coords=[0.0; wall_distance;;],
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

        # Create fluid starting at y = 0
        fluid = rectangular_patch(particle_spacing, (nx, ny), seed=1,
                                  perturbation_factor=0.0, perturbation_factor_position=0.0,
                                  set_function=set_values,
                                  offset=[0.0, ny * 0.5 * particle_spacing])
        # Create boundary below the fluid ending at y = 0
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
                                                     viscosity=viscosity,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length)

        boundary_system = WallBoundarySystem(bnd, boundary_model)

        # Overwrite `system.pressure` because we skip the update step
        fluid_system.pressure .= fluid.pressure

        u_no_bnd = fluid.coordinates
        # Density is integrated with `ContinuityDensity`
        v_no_bnd = vcat(fluid.velocity, fluid.density')

        u_bnd = hcat(fluid.coordinates, bnd.coordinates)
        v_bnd_velocity = hcat(fluid.velocity, bnd.velocity)
        v_bnd_density = vcat(fluid.density, bnd.density)

        v_bnd = vcat(v_bnd_velocity, v_bnd_density')

        semi_no_boundary = Semidiscretization(fluid_system)
        TrixiParticles.initialize_neighborhood_searches!(semi_no_boundary)
        semi_boundary = Semidiscretization(fluid_system, boundary_system)
        TrixiParticles.initialize_neighborhood_searches!(semi_boundary)

        # Some simple results
        expected_zero(y) = (density=[NaN], neighbor_count=[0], point_coords=[0.0; y;;],
                            velocity=[NaN; NaN;;], pressure=[NaN])

        for cut_off_bnd in [true, false]
            @testset verbose=true "Interpolation Point no boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_walldistance(y) = TrixiParticles.interpolate_points([0.0; y;;],
                                                                                  semi_no_boundary,
                                                                                  fluid_system,
                                                                                  v_no_bnd,
                                                                                  u_no_bnd,
                                                                                  cut_off_bnd=cut_off_bnd)

                # Top outside
                distance_top_outside = binary_search_outside(ny * particle_spacing,
                                                             (ny + 2) * particle_spacing,
                                                             interpolation_walldistance)
                @test isapprox(distance_top_outside, 0.11817626953124982, atol=1e-14)

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

                # Top at free surface
                result_top = interpolation_walldistance(ny * particle_spacing)
                compare_interpolation_result(result_top,
                                             expected_result(ny * particle_spacing,
                                                             wall_height, 2))

                # Center
                result_center = interpolation_walldistance(ny * 0.5 * particle_spacing)
                compare_interpolation_result(result_center,
                                             expected_result(ny * 0.5 * particle_spacing,
                                                             wall_height, 5))

                # At wall
                exp_res = (density=[665.9999999999999],
                           neighbor_count=[2],
                           point_coords=[0.0; 0.0;;],
                           velocity=[5.0; 0.101;;],
                           pressure=[0.2])
                result_bottom = interpolation_walldistance(0.0)
                compare_interpolation_result(result_bottom, exp_res)

                distance_bottom_outside = binary_search_outside(0.0, -2 * particle_spacing,
                                                                interpolation_walldistance)
                @test isapprox(distance_bottom_outside, 0.11817626953125, atol=1e-14)

                exp_res = (density=[666.0],
                           neighbor_count=[2],
                           point_coords=[0.0; -0.5*distance_bottom_outside;;],
                           velocity=[5.0; 0.101;;],
                           pressure=[0.2])
                result_bottom_outside = interpolation_walldistance(-0.5 *
                                                                   distance_bottom_outside)
                compare_interpolation_result(result_bottom_outside, exp_res)

                result_zero = interpolation_walldistance(-distance_bottom_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(-distance_bottom_outside))

                multi_point_coords = [0.0 0.0 0.0; 0.0 0.5 1.0]

                result_multipoint = TrixiParticles.interpolate_points(multi_point_coords,
                                                                      semi_no_boundary,
                                                                      fluid_system,
                                                                      v_no_bnd, u_no_bnd,
                                                                      cut_off_bnd=cut_off_bnd)

                expected_multi = (density=[666.0, 666.0000000000001, 666.0],
                                  neighbor_count=[2, 6, 5],
                                  point_coords=multi_point_coords,
                                  velocity=[5.0 5.0 5.0;
                                            0.101 0.125 0.20035665520692278],
                                  pressure=[0.19999999999999996, 1.0000000000000002, 2.0])

                compare_interpolation_result(result_multipoint, expected_multi)
            end
            @testset verbose=true "Interpolation Line no boundary - cut_off_bnd = $(cut_off_bnd)" begin
                result_endpoint = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0],
                                                                  5, semi_no_boundary,
                                                                  fluid_system,
                                                                  v_no_bnd, u_no_bnd,
                                                                  endpoint=true,
                                                                  cut_off_bnd=cut_off_bnd)

                result = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0],
                                                         5, semi_no_boundary,
                                                         fluid_system, v_no_bnd, u_no_bnd,
                                                         endpoint=false,
                                                         cut_off_bnd=cut_off_bnd)

                expected_res = (density=[666.0, 666.0, 666.0], neighbor_count=[2, 2, 1],
                                point_coords=[1.0 1.0 1.0;
                                              0.2125 0.47500000000000003 0.7375],
                                velocity=[7.699999999999999 7.7 7.7;
                                          0.10605429538320173 0.12465095587703466 0.14900000000000002],
                                pressure=[
                                    0.4527147691600855,
                                    0.9912738969258665,
                                    1.4000000000000001
                                ])
                expected_res_end = (density=[666.0, 666.0, 666.0, 666.0, 666.0],
                                    neighbor_count=[1, 2, 2, 1, 1],
                                    point_coords=[1.0 1.0 1.0 1.0 1.0;
                                                  -0.05 0.2125 0.475 0.7375 1.0],
                                    velocity=[7.7 7.699999999999999 7.699999999999999 7.7 7.7;
                                              0.10100000000000002 0.10605429538320173 0.12465095587703466 0.14900000000000002 0.22100000000000006],
                                    pressure=[
                                        0.19999999999999998,
                                        0.4527147691600855,
                                        0.9912738969258663,
                                        1.4000000000000001,
                                        2.2
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
                                              fluid_system, v_no_bnd, u_no_bnd)

                expected_res = (density=[
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0
                                ],
                                neighbor_count=[
                                    2, 2, 3, 2, 1, 4, 4, 4, 4, 2, 6, 4, 5, 4, 3, 4, 4, 4,
                                    3, 1, 5, 4, 6, 3, 1
                                ],
                                point_coords=[0.0 0.25 0.5 0.75 1.0 0.0 0.25 0.5 0.75 1.0 0.0 0.25 0.5 0.75 1.0 0.0 0.25 0.5 0.75 1.0 0.0 0.25 0.5 0.75 1.0;
                                              0.0 0.0 0.0 0.0 0.0 0.25 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.5 0.5 0.75 0.75 0.75 0.75 0.75 1.0 1.0 1.0 1.0 1.0],
                                velocity=[5.0 5.844853603211259 6.5 7.155146396788742 7.7 5.000000000000001 5.8376544066143845 6.499999999999999 7.162345593385616 7.7 5.000000000000001 5.8305001181675005 6.499999999999998 7.1694998818325 7.700000000000002 4.999999999999999 5.837654406614385 6.500000000000001 7.160218593182242 7.7 5.0 5.84485360321126 6.5 7.128901370428878 7.7;
                                          0.10100000000000002 0.10099999999999999 0.101 0.101 0.101 0.10826471470948347 0.10816872542152514 0.10807333490890002 0.10816872542152513 0.10826471470948347 0.12501295337729817 0.12504927969391108 0.12507142857142856 0.12504927969391108 0.12501295337729815 0.15194114116206617 0.15232509831389957 0.15270666036440003 0.1522116583030529 0.14900000000000002 0.20035665520692278 0.201 0.20100000000000004 0.2019633790142959 0.22100000000000006],
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
                                    2.0, 2.009633790142959, 2.2
                                ])

                compare_interpolation_result(result, expected_res)

                result = interpolate_plane_2d(interpolation_start, interpolation_end,
                                              resolution, semi_no_boundary, fluid_system,
                                              v_no_bnd, u_no_bnd,
                                              smoothing_length=0.5 * smoothing_length)
                expected_res = (density=[
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0, 666.0, 666.0
                                ],
                                neighbor_count=[
                                    1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                                    2, 2, 2
                                ],
                                point_coords=[0.25 0.5 0.75 0.0 0.25 0.5 0.75 1.0 0.0 0.25 0.5 0.75 1.0 0.0 0.25 0.5 0.75 1.0 0.0 0.25 0.5 0.75;
                                              0.0 0.0 0.0 0.25 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.5 0.5 0.75 0.75 0.75 0.75 0.75 1.0 1.0 1.0 1.0],
                                velocity=[5.9 6.499999999999999 7.1000000000000005 4.999999999999994 5.9 6.5 7.1000000000000005 7.7 4.999999999999998 5.9 6.5 7.1 7.7 4.999999999999995 5.900000000000001 6.5 7.1000000000000005 7.7 5.0 5.8999999999999995 6.5 7.100000000000001;
                                          0.101 0.101 0.101 0.10900000000000003 0.10900000000000001 0.109 0.10900000000000003 0.10900000000000001 0.125 0.125 0.125 0.125 0.125 0.14900000000000002 0.14900000000000002 0.14900000000000002 0.14900000000000002 0.14900000000000002 0.2 0.20099999999999962 0.20099999999999985 0.20099999999999968],
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
                                    1.9999999999999967
                                ])

                compare_interpolation_result(result, expected_res)
            end
        end

        for cut_off_bnd in [true, false]
            @testset verbose=true "Interpolation Point boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_walldistance(y) = TrixiParticles.interpolate_points([0.0; y;;],
                                                                                  semi_boundary,
                                                                                  fluid_system,
                                                                                  v_bnd,
                                                                                  u_bnd,
                                                                                  cut_off_bnd=cut_off_bnd)

                # Top outside
                distance_top_outside = binary_search_outside(ny * particle_spacing,
                                                             (ny + 2) * particle_spacing,
                                                             interpolation_walldistance)
                @test isapprox(distance_top_outside, 0.11817626953124982, atol=1e-14)

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

                # Top at free surface
                result_top = interpolation_walldistance(ny * particle_spacing)
                compare_interpolation_result(result_top,
                                             expected_result(ny * particle_spacing,
                                                             wall_height,
                                                             2))

                # Center
                result_center = interpolation_walldistance(ny * 0.5 * particle_spacing)
                compare_interpolation_result(result_center,
                                             expected_result(ny * 0.5 * particle_spacing,
                                                             wall_height, 5))

                # At wall
                result_bottom = interpolation_walldistance(0.0)
                if cut_off_bnd
                    exp_res = (density=[666],
                               neighbor_count=[4],
                               point_coords=[0.0; 0.0;;],
                               velocity=[5.0; 0.101;;],
                               pressure=[0.2])
                    compare_interpolation_result(result_bottom, exp_res)
                else
                    exp_res = (density=[666],
                               neighbor_count=[2],
                               point_coords=[0.0; 0.0;;],
                               velocity=[5.0; 0.101;;],
                               pressure=[0.2])
                    compare_interpolation_result(result_bottom, exp_res)
                end

                distance_bottom_outside = binary_search_outside(0.0, -2 * particle_spacing,
                                                                interpolation_walldistance,
                                                                tolerance=1e-12)
                if cut_off_bnd
                    @test isapprox(distance_bottom_outside, 0, atol=1e-12)
                else
                    @test isapprox(distance_bottom_outside, 0.11817392366574496, atol=1e-14)
                end

                result_bottom_outside = interpolation_walldistance(-0.5 *
                                                                   distance_bottom_outside)

                if cut_off_bnd
                    compare_interpolation_result(result_bottom_outside,
                                                 expected_zero(-0.5 *
                                                               distance_bottom_outside))
                else
                    exp_res = (density=[666],
                               neighbor_count=[2],
                               point_coords=[0.0; -0.5*distance_bottom_outside;;],
                               velocity=[5.0; 0.101;;],
                               pressure=[0.2])
                    compare_interpolation_result(result_bottom_outside, exp_res)
                end

                result_zero = interpolation_walldistance(-distance_bottom_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(-distance_bottom_outside))

                multi_point_coords = [0.0 0.0 0.0; 0.0 0.5 1.0]

                result_multipoint = TrixiParticles.interpolate_points(multi_point_coords,
                                                                      semi_boundary,
                                                                      fluid_system,
                                                                      v_bnd, u_bnd,
                                                                      cut_off_bnd=cut_off_bnd)
                if cut_off_bnd
                    expected_multi = (density=[666.0, 666.0000000000001, 666.0],
                                      neighbor_count=[4, 6, 5],
                                      point_coords=multi_point_coords,
                                      velocity=[5.0 5.000000000000001 5.0;
                                                0.10100000000000002 0.12501295337729817 0.20035665520692278],
                                      pressure=[
                                          0.19999999999999996,
                                          1.0000000000000002,
                                          2.0
                                      ])

                    compare_interpolation_result(result_multipoint, expected_multi)
                else
                    expected_multi = (density=[666.0, 666.0000000000001, 666.0],
                                      neighbor_count=[2, 6, 5],
                                      point_coords=multi_point_coords,
                                      velocity=[5.0 5.000000000000001 5.0;
                                                0.10100000000000002 0.12501295337729817 0.20035665520692278],
                                      pressure=[
                                          0.19999999999999996,
                                          1.0000000000000002,
                                          2.0
                                      ])

                    compare_interpolation_result(result_multipoint, expected_multi)
                end
            end
            @testset verbose=true "Interpolation Line boundary - cut_off_bnd = $(cut_off_bnd)" begin
                result_endpoint = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0],
                                                                  5, semi_boundary,
                                                                  fluid_system,
                                                                  v_bnd, u_bnd,
                                                                  endpoint=true,
                                                                  cut_off_bnd=cut_off_bnd)

                result = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0],
                                                         5, semi_no_boundary,
                                                         fluid_system, v_no_bnd, u_no_bnd,
                                                         endpoint=false,
                                                         cut_off_bnd=cut_off_bnd)
                if cut_off_bnd
                    expected_res = (density=[666.0, 666.0, 666.0], neighbor_count=[2, 2, 1],
                                    point_coords=[1.0 1.0 1.0;
                                                  0.2125 0.47500000000000003 0.7375],
                                    velocity=[7.699999999999999 7.7 7.7;
                                              0.10605429538320173 0.12465095587703466 0.14900000000000002],
                                    pressure=[
                                        0.4527147691600855,
                                        0.9912738969258665,
                                        1.4000000000000001
                                    ])
                    expected_res_end = (density=[
                                            NaN,
                                            666.0,
                                            665.9999999999999,
                                            666.0,
                                            666.0
                                        ],
                                        neighbor_count=[0, 2, 2, 1, 1],
                                        point_coords=[1.0 1.0 1.0 1.0 1.0;
                                                      -0.05 0.2125 0.475 0.7375 1.0],
                                        velocity=[NaN 7.699999999999999 7.699999999999999 7.7 7.7;
                                                  NaN 0.10605429538320173 0.12465095587703465 0.14900000000000002 0.22100000000000006],
                                        pressure=[
                                            NaN,
                                            0.4527147691600855,
                                            0.9912738969258663,
                                            1.4000000000000001,
                                            2.2
                                        ])

                    compare_interpolation_result(result, expected_res)
                    compare_interpolation_result(result_endpoint, expected_res_end)

                else
                    expected_res = (density=[666.0, 666.0, 666.0], neighbor_count=[2, 2, 1],
                                    point_coords=[1.0 1.0 1.0;
                                                  0.2125 0.47500000000000003 0.7375],
                                    velocity=[7.699999999999999 7.7 7.7;
                                              0.10605429538320173 0.12465095587703466 0.14900000000000002],
                                    pressure=[
                                        0.4527147691600855,
                                        0.9912738969258665,
                                        1.4000000000000001
                                    ])
                    expected_res_end = (density=[
                                            666.0,
                                            666.0,
                                            665.9999999999999,
                                            666.0,
                                            666.0
                                        ], neighbor_count=[1, 2, 2, 1, 1],
                                        point_coords=[1.0 1.0 1.0 1.0 1.0;
                                                      -0.05 0.2125 0.475 0.7375 1.0],
                                        velocity=[7.7 7.699999999999999 7.699999999999999 7.7 7.7;
                                                  0.10099999999999999 0.10605429538320173 0.12465095587703465 0.14900000000000002 0.22100000000000006],
                                        pressure=[
                                            0.19999999999999998,
                                            0.4527147691600855,
                                            0.9912738969258663,
                                            1.4000000000000001,
                                            2.2
                                        ])

                    compare_interpolation_result(result, expected_res)
                    compare_interpolation_result(result_endpoint, expected_res_end)
                end
            end
        end

        @testset verbose=true "Include Wall Velocity" begin
            # Define vertical interpolation line
            start_svector = SVector(0.0, 0.0)
            end_svector = SVector(0.0, 1.0)
            points_coords_ = range(start_svector, end_svector, length=10)
            points_coords = collect(reinterpret(reshape, eltype(start_svector),
                                                points_coords_))

            # Linear velocity field for fluid and boundary
            v_fluid = InitialCondition(;
                                       coordinates=fluid_system.initial_condition.coordinates,
                                       density=1000.0, particle_spacing,
                                       velocity=(pos) -> SVector(0.0, pos[2])).velocity
            v_boundary = InitialCondition(; coordinates=boundary_system.coordinates,
                                          density=1000.0, particle_spacing,
                                          velocity=(pos) -> SVector(0.0, pos[2])).velocity

            boundary_system.boundary_model.cache.wall_velocity .= v_boundary

            v_ode = vcat(v_fluid, fluid.density')
            u_ode = fluid_system.initial_condition.coordinates

            v_wall_velocity = interpolate_points(points_coords, semi_boundary,
                                                 include_wall_velocity=true,
                                                 fluid_system, v_ode, u_ode).velocity

            v_no_wall_velocity = interpolate_points(points_coords, semi_boundary,
                                                    include_wall_velocity=false,
                                                    fluid_system, v_ode, u_ode).velocity

            @test isapprox(v_wall_velocity[2, 1], 0.0; atol=eps())
            @test isapprox(v_no_wall_velocity[2, 1], 0.1; atol=eps())
            @test any(isapprox.(v_wall_velocity[:, 3:end], v_no_wall_velocity[:, 3:end],
                                atol=eps()))
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
            # Interpolated values above the last particle but within the interpolation
            # cutoff return the values of the last particle.
            new_wall_distance = wall_distance
            if wall_distance > max_wallheight
                new_wall_distance = max_wallheight
            end

            const_density = [666.0]
            const_pressure = [2 * new_wall_distance]
            const_velocity = [5; 0.1 * new_wall_distance^2+0.1; 0.0;;]

            return (density=const_density,
                    neighbor_count=neighbor_count,
                    point_coords=[0.0; wall_distance; 0.0;;],
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
                                                     viscosity=viscosity,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length)

        boundary_system = WallBoundarySystem(bnd, boundary_model)

        # Overwrite `system.pressure` because we skip the update step
        fluid_system.pressure .= fluid.pressure

        u_no_bnd = fluid.coordinates
        # Density is integrated with `ContinuityDensity`
        v_no_bnd = vcat(fluid.velocity, fluid.density')

        u_bnd = hcat(fluid.coordinates, bnd.coordinates)
        v_bnd_velocity = hcat(fluid.velocity, bnd.velocity)
        v_bnd_density = vcat(fluid.density, bnd.density)

        v_bnd = vcat(v_bnd_velocity, v_bnd_density')

        semi_no_boundary = Semidiscretization(fluid_system)
        TrixiParticles.initialize_neighborhood_searches!(semi_no_boundary)
        semi_boundary = Semidiscretization(fluid_system, boundary_system)
        TrixiParticles.initialize_neighborhood_searches!(semi_boundary)

        # some simple results
        expected_zero(y) = (density=[NaN], neighbor_count=[0], point_coords=[0.0; y; 0.0;;],
                            velocity=[NaN; NaN; NaN;;], pressure=[NaN])

        for cut_off_bnd in [true, false]
            @testset verbose=true "Interpolation Point no boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_walldistance(y) = TrixiParticles.interpolate_points([0.0;
                                                                                   y;
                                                                                   0.0;;],
                                                                                  semi_no_boundary,
                                                                                  fluid_system,
                                                                                  v_no_bnd,
                                                                                  u_no_bnd,
                                                                                  cut_off_bnd=cut_off_bnd)

                # Top outside
                distance_top_outside = binary_search_outside(ny * particle_spacing,
                                                             (ny + 2) * particle_spacing,
                                                             interpolation_walldistance,
                                                             tolerance=1e-8)
                @test isapprox(distance_top_outside, 0.09390704035758901, atol=1e-14)

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

                # Top at free surface
                result_top = interpolation_walldistance(ny * particle_spacing)
                compare_interpolation_result(result_top,
                                             expected_result(ny * particle_spacing,
                                                             wall_height,
                                                             4))

                # Center
                result_center = interpolation_walldistance(ny * 0.5 * particle_spacing)
                compare_interpolation_result(result_center,
                                             expected_result(ny * 0.5 * particle_spacing,
                                                             wall_height, 9))

                # At wall
                exp_res = (density=[665.9999999999999],
                           neighbor_count=[4],
                           point_coords=[0.0; 0.0; 0.0;;],
                           velocity=[5.0; 0.101; 0.0;;],
                           pressure=[0.2])
                result_bottom = interpolation_walldistance(0.0)
                compare_interpolation_result(result_bottom, exp_res)

                distance_bottom_outside = binary_search_outside(0.0, -2 * particle_spacing,
                                                                interpolation_walldistance,
                                                                tolerance=1e-12)
                @test isapprox(distance_bottom_outside, 0.0939070362292114, atol=1e-14)

                exp_res = (density=[666.0],
                           neighbor_count=[4],
                           point_coords=[0.0; -0.5*distance_bottom_outside; 0.0;;],
                           velocity=[5.0; 0.101; 0.0;;],
                           pressure=[0.2])
                result_bottom_outside = interpolation_walldistance(-0.5 *
                                                                   distance_bottom_outside)
                compare_interpolation_result(result_bottom_outside, exp_res)

                result_zero = interpolation_walldistance(-distance_bottom_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(-distance_bottom_outside))

                multi_point_coords = [0.0 0.0 0.0; 0.0 0.5 1.0; 0.0 0.0 0.0]

                result_multipoint = TrixiParticles.interpolate_points(multi_point_coords,
                                                                      semi_no_boundary,
                                                                      fluid_system,
                                                                      v_no_bnd, u_no_bnd,
                                                                      cut_off_bnd=cut_off_bnd)

                expected_multi = (density=[666.0, 666.0, 666.0], neighbor_count=[4, 4, 9],
                                  point_coords=multi_point_coords,
                                  velocity=[5.0 5.0 5.0;
                                            0.101 0.125 0.20025646054622268;
                                            0.0 0.0 0.0],
                                  pressure=[0.19999999999999996, 1.0, 2.0])

                compare_interpolation_result(result_multipoint, expected_multi)
            end
        end

        for cut_off_bnd in [true, false]
            @testset verbose=true "Interpolation Point boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_walldistance(y) = TrixiParticles.interpolate_points([0.0;
                                                                                   y;
                                                                                   0.0;;],
                                                                                  semi_boundary,
                                                                                  fluid_system,
                                                                                  v_no_bnd,
                                                                                  u_no_bnd,
                                                                                  cut_off_bnd=cut_off_bnd)

                # Top outside
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

                # Top at free surface
                result_top = interpolation_walldistance(ny * particle_spacing)
                compare_interpolation_result(result_top,
                                             expected_result(ny * particle_spacing,
                                                             wall_height, 4))

                # Center
                result_center = interpolation_walldistance(ny * 0.5 * particle_spacing)
                compare_interpolation_result(result_center,
                                             expected_result(ny * 0.5 * particle_spacing,
                                                             wall_height, 9))

                # At wall
                result_bottom = interpolation_walldistance(0.0)
                if cut_off_bnd
                    exp_res = (density=[666],
                               neighbor_count=[8],
                               point_coords=[0.0; 0.0; 0.0;;],
                               velocity=[5.0; 0.101; 0.0;;],
                               pressure=[0.2])
                    compare_interpolation_result(result_bottom, exp_res)
                else
                    exp_res = (density=[666],
                               neighbor_count=[4],
                               point_coords=[0.0; 0.0; 0.0;;],
                               velocity=[5.0; 0.101; 0.0;;],
                               pressure=[0.2])
                    compare_interpolation_result(result_bottom, exp_res)
                end

                distance_bottom_outside = binary_search_outside(0.0, -2 * particle_spacing,
                                                                interpolation_walldistance,
                                                                tolerance=1e-12)
                if cut_off_bnd
                    @test isapprox(distance_bottom_outside, 0, atol=1e-12)
                else
                    @test isapprox(distance_bottom_outside, 0.0939070362292114, atol=1e-14)
                end

                result_bottom_outside = interpolation_walldistance(-0.5 *
                                                                   distance_bottom_outside)

                if cut_off_bnd
                    compare_interpolation_result(result_bottom_outside,
                                                 expected_zero(-0.5 *
                                                               distance_bottom_outside))
                else
                    exp_res = (density=[666],
                               neighbor_count=[4],
                               point_coords=[0.0; -0.5*distance_bottom_outside; 0.0;;],
                               velocity=[5.0; 0.101; 0.0;;],
                               pressure=[0.2])
                    compare_interpolation_result(result_bottom_outside, exp_res)
                end

                result_zero = interpolation_walldistance(-distance_bottom_outside)
                compare_interpolation_result(result_zero,
                                             expected_zero(-distance_bottom_outside))

                multi_point_coords = [0.0 0.0 0.0; 0.0 0.5 1.0; 0.0 0.0 0.0]

                result_multipoint = TrixiParticles.interpolate_points(multi_point_coords,
                                                                      semi_no_boundary,
                                                                      fluid_system,
                                                                      v_no_bnd, u_no_bnd,
                                                                      cut_off_bnd=cut_off_bnd)

                expected_multi = (density=[666.0, 666.0, 666.0], neighbor_count=[4, 4, 9],
                                  point_coords=multi_point_coords,
                                  velocity=[5.0 5.0 5.0;
                                            0.101 0.125 0.20025646054622268;
                                            0.0 0.0 0.0],
                                  pressure=[0.19999999999999996, 1.0, 2.0])

                compare_interpolation_result(result_multipoint, expected_multi)
            end

            @testset verbose=true "Interpolation Plane no boundary - cut_off_bnd = $(cut_off_bnd)" begin
                p1 = [0.0, 0.0, 0.0]
                p2 = [1.0, 1.0, 0.0]
                p3 = [0.0, 1.0, 0.0]
                resolution = 0.5

                result = interpolate_plane_3d(p1, p2, p3,
                                              resolution, semi_no_boundary,
                                              fluid_system, v_no_bnd, u_no_bnd)

                expected_res = (density=[666.0, 666.0, 666.0, 666.0, 666.0, 666.0, 666.0,
                                    666.0, 666.0, 666.0, 666.0, 666.0
                                ], neighbor_count=[4, 4, 9, 6, 8, 6, 6, 8, 6, 4, 2, 1],
                                point_coords=[0.0 0.0 0.0 0.3333333333333333 0.3333333333333333 0.3333333333333333 0.6666666666666666 0.6666666666666666 0.6666666666666666 1.0 1.0 1.0;
                                              0.0 0.5 1.0 0.3333333333333333 0.8333333333333333 1.3333333333333333 0.6666666666666666 1.1666666666666665 1.6666666666666665 1.0 1.5 2.0;
                                              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0],
                                velocity=[5.0 5.0 5.0 5.920251592989799 5.912901655029027 5.920251592989798 7.079748407010201 7.087098344970975 7.079748407010202 7.699999999999998 7.700000000000001 7.699999999999999;
                                          0.101 0.125 0.20025646054622268 0.10954004247972798 0.1752639260703962 0.2708901486790478 0.14818993628040808 0.22960411089440566 0.38683983008108835 0.20099999999999996 0.32500000000000007 0.4610000000000001;
                                          -8.637243445401583e-17 -7.070877847937661e-17 -2.2734045974413372e-17 -7.466540216672048e-17 -8.888366469079487e-17 -5.308243435290283e-17 -7.466540216672064e-17 -8.19920883032382e-17 -1.119981032500812e-16 -1.2273977527675892e-16 -6.222372506185122e-17 0.10000000000000009],
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
                                    3.8000000000000003
                                ])

                compare_interpolation_result(result, expected_res)
            end
        end

        @testset verbose=true "Include Wall Velocity" begin
            # Define vertical interpolation line
            start_svector = SVector(0.0, 0.0, 0.0)
            end_svector = SVector(0.0, 1.0, 0.0)
            points_coords_ = range(start_svector, end_svector, length=10)
            points_coords = collect(reinterpret(reshape, eltype(start_svector),
                                                points_coords_))

            # Linear velocity field for fluid and boundary
            v_fluid = InitialCondition(;
                                       coordinates=fluid_system.initial_condition.coordinates,
                                       density=1000.0, particle_spacing,
                                       velocity=(pos) -> SVector(0.0, pos[2], 0.0)).velocity
            v_boundary = InitialCondition(; coordinates=boundary_system.coordinates,
                                          density=1000.0, particle_spacing,
                                          velocity=(pos) -> SVector(0.0, pos[2], 0.0)).velocity

            boundary_system.boundary_model.cache.wall_velocity .= v_boundary

            v_ode = vcat(v_fluid, fluid.density')
            u_ode = fluid_system.initial_condition.coordinates

            v_wall_velocity = interpolate_points(points_coords, semi_boundary,
                                                 include_wall_velocity=true,
                                                 fluid_system, v_ode, u_ode).velocity

            v_no_wall_velocity = interpolate_points(points_coords, semi_boundary,
                                                    include_wall_velocity=false,
                                                    fluid_system, v_ode, u_ode).velocity

            @test isapprox(v_wall_velocity[2, 1], 0.0; atol=eps())
            @test isapprox(v_no_wall_velocity[2, 1], 0.1; atol=eps())
            @test any(isapprox.(v_wall_velocity[:, 3:end], v_no_wall_velocity[:, 3:end],
                                atol=eps()))
        end
    end
end;
