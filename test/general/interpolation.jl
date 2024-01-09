using OrdinaryDiffEq
include("../test_util.jl")

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

        state_equation = StateEquationCole(sound_speed, 7, 1000, 100000.0,
                                           background_pressure=100000.0,
                                           clip_negative_pressure=false)

        fluid = rectangular_patch(particle_spacing, (nx, ny), seed=1,
                                  perturbation_factor=0.0,
                                  set_function=set_values,
                                  offset=[0.0, ny * 0.5 * particle_spacing])
        bnd = rectangular_patch(particle_spacing, (bnd_nx, bnd_ny), seed=1,
                                perturbation_factor=0.0, set_function=set_values_bnd,
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

        state_equation = StateEquationCole(sound_speed, 7, 1000, 100000.0,
                                           background_pressure=100000.0,
                                           clip_negative_pressure=false)

        fluid = rectangular_patch(particle_spacing, (nx, ny, nz), seed=1,
                                  perturbation_factor=0.0,
                                  set_function=set_values,
                                  offset=[0.0, ny * 0.5 * particle_spacing, 0.0])
        bnd = rectangular_patch(particle_spacing, (bnd_nx, bnd_ny, bnd_nz), seed=1,
                                perturbation_factor=0.0, set_function=set_values_bnd,
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
                interpolation_walldistance(y) = TrixiParticles.interpolate_point([0.0, y, 0.0],
                                                                                 semi_no_boundary,
                                                                                 fluid_system,
                                                                                 sol_no_boundary,
                                                                                 cut_off_bnd=cut_off_bnd)

                # top outside
                distance_top_outside = binary_search_outside(ny * particle_spacing,
                                                             (ny + 2) * particle_spacing,
                                                             interpolation_walldistance, tolerance=1e-8)
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
                                                                interpolation_walldistance, tolerance=1e-12)
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
            end
        end

        for cut_off_bnd in [true, false]
            @testset verbose=true "Interpolation Point boundary - cut_off_bnd = $(cut_off_bnd)" begin
                interpolation_walldistance(y) = TrixiParticles.interpolate_point([0.0, y, 0.0],
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
            end
        end
    end
end

# @testset verbose=true "SPHInterpolation" begin
#     smoothing_length = 1.2 * 0.1
#     smoothing_kernel = SchoenbergCubicSplineKernel{2}()
#     sound_speed = 10 * sqrt(9.81 * 0.9)

#     state_equation = StateEquationCole(sound_speed, 7, 1000, 100000.0,
#                                        background_pressure=100000.0,
#                                        clip_negative_pressure=false)

#     tank = RectangularTank(0.1, (2.0, 0.9), (2.0, 1.0), 1000, n_layers=4,
#                            acceleration=(0.0, -9.81), state_equation=state_equation)

#     viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

#     fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, ContinuityDensity(),
#                                                state_equation, smoothing_kernel,
#                                                smoothing_length, viscosity=viscosity,
#                                                acceleration=(0.0, -9.81))

#     boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
#                                                  state_equation=state_equation,
#                                                  AdamiPressureExtrapolation(),
#                                                  smoothing_kernel, smoothing_length)

#     boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

#     semi = Semidiscretization(fluid_system, boundary_system,
#                               neighborhood_search=GridNeighborhoodSearch)
#     ode = semidiscretize(semi, (0.0, 0.1))

#     sol = solve(ode, RDPK3SpFSAL49(), abstol=1e-5, reltol=1e-3, dtmax=1e-2,
#                 save_everystep=false)

#     function compare_interpolation_result(actual, expected; tolerance=1e-8)
#         @test length(actual.density) == length(expected.density)
#         for i in 1:length(expected.density)
#             @test actual.neighbor_count[i] == expected.neighbor_count[i]
#             @test actual.coord[i] == expected.coord[i]
#             @test isapprox(actual.density[i], expected.density[i], atol=tolerance)
#             @test isapprox(actual.velocity[i], expected.velocity[i], atol=tolerance)
#             @test isapprox(actual.pressure[i], expected.pressure[i], atol=tolerance)
#         end
#     end

#     @testset verbose=true "Interpolation Point" begin
#         result = TrixiParticles.interpolate_point([1.0, 0.01], semi, fluid_system, sol,
#                                                   cut_off_bnd=true)

#         expected_result = (density=1009.4921501393262, neighbor_count=16, coord=[1.0, 0.01],
#                            velocity=[1.7405906444065288e-14, 0.005661512837091638],
#                            pressure=8623.500764832543)
#         compare_interpolation_result(result, expected_result)

#         # with deactivated cut_off_bnd
#         # results extend outside of the fluid domain
#         result = TrixiParticles.interpolate_point([1.0, 0.01], semi, fluid_system, sol,
#                                                   cut_off_bnd=false)

#         expected_result = (density=1009.4921501393262, neighbor_count=8, coord=[1.0, 0.01],
#                            velocity=[1.7405906444065288e-14, 0.005661512837091638],
#                            pressure=8623.500764832543)
#         compare_interpolation_result(result, expected_result)
#     end
#     @testset verbose=true "Interpolation Multi Point" begin
#         result = TrixiParticles.interpolate_point([
#                                                       [1.0, 0.01],
#                                                       [1.0, 0.1],
#                                                       [1.0, 0.0],
#                                                       [1.0, -0.01],
#                                                       [1.0, -0.05],
#                                                   ], semi, fluid_system, sol,
#                                                   cut_off_bnd=true)

#         expected_result = (density=[
#                                1009.4921501393262,
#                                1008.9764089268859,
#                                1009.5229791447398,
#                                0.0,
#                                0.0,
#                            ],
#                            neighbor_count=[16, 16, 16, 0, 0],
#                            coord=[
#                                [1.0, 0.01],
#                                [1.0, 0.1],
#                                [1.0, 0.0],
#                                [1.0, -0.01],
#                                [1.0, -0.05],
#                            ],
#                            velocity=[
#                                [1.7405906444065288e-14, 0.005661512837091638],
#                                [1.442422876475543e-14, 0.004993889596774216],
#                                [1.763368469598575e-14, 0.005720069349028347],
#                                [0.0, 0.0],
#                                [0.0, 0.0],
#                            ],
#                            pressure=[
#                                8623.500764832543,
#                                8143.159169507029,
#                                8652.234332097962,
#                                0.0,
#                                0.0,
#                            ])
#         compare_interpolation_result(result, expected_result)

#         # with deactivated cut_off_bnd
#         # results extend outside of the fluid domain
#         result = TrixiParticles.interpolate_point([
#                                                       [1.0, 0.01],
#                                                       [1.0, 0.1],
#                                                       [1.0, 0.0],
#                                                       [1.0, -0.01],
#                                                       [1.0, -0.05],
#                                                       [1.0, -0.1],
#                                                       [1.0, -0.2],
#                                                   ], semi, fluid_system, sol,
#                                                   cut_off_bnd=false)

#         expected_result = (density=[1009.4921501393262, 1008.9764089268859,
#                                1009.5229791447398, 1009.5501117234779,
#                                1009.6243395503883, 1009.6443775884171, 0.0],
#                            neighbor_count=[8, 12, 8, 8, 6, 4, 0],
#                            coord=[
#                                [1.0, 0.01],
#                                [1.0, 0.1],
#                                [1.0, 0.0],
#                                [1.0, -0.01],
#                                [1.0, -0.05],
#                                [1.0, -0.1],
#                                [1.0, -0.2],
#                            ],
#                            velocity=[[1.7405906444065288e-14, 0.005661512837091638],
#                                [1.442422876475543e-14, 0.004993889596774216],
#                                [1.763368469598575e-14, 0.005720069349028347],
#                                [1.7847274797051723e-14, 0.005771554677158952],
#                                [1.8600165065214594e-14, 0.005911333559787621],
#                                [1.9714720554750176e-14, 0.005939365031527732],
#                                [0.0, 0.0]],
#                            pressure=[8623.500764832543, 8143.159169507029,
#                                8652.234332097962, 8677.522799474411,
#                                8746.706496708202, 8765.385734383106, 0.0])

#         compare_interpolation_result(result, expected_result)
#     end
#     @testset verbose=true "Interpolation Line" begin
#         expected_result = (density=[
#                                0.0,
#                                1007.8388079925885,
#                                1004.907892717609,
#                                1001.899652412931,
#                                0.0,
#                            ],
#                            neighbor_count=[0, 16, 18, 14, 0],
#                            coord=[
#                                [1.0, -0.05],
#                                [1.0, 0.21250000000000002],
#                                [1.0, 0.47500000000000003],
#                                [1.0, 0.7375],
#                                [1.0, 1.0],
#                            ],
#                            velocity=[[0.0, 0.0],
#                                [1.0308096030666658e-14, 0.005907502958502652],
#                                [7.52868309149294e-15, 0.007855784662715764],
#                                [9.025768335787871e-15, 0.009933832252516267], [0.0, 0.0]],
#                            pressure=[
#                                0.0,
#                                7087.237427116723,
#                                4399.142055600923,
#                                1688.3585518071602,
#                                0.0,
#                            ])

#         result = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0], 5, semi,
#                                                  fluid_system, sol,
#                                                  cut_off_bnd=true)
#         compare_interpolation_result(result, expected_result)

#         expected_result_endpoint = (density=[
#                                         1008.7564078596042,
#                                         1006.8750741471611,
#                                         1004.907892717609,
#                                         1002.9010166322735,
#                                         1001.1010154235912,
#                                     ],
#                                     neighbor_count=[18, 16, 18, 16, 14],
#                                     coord=[
#                                         [1.0, 0.12499999999999999],
#                                         [1.0, 0.3],
#                                         [1.0, 0.47500000000000003],
#                                         [1.0, 0.6499999999999999],
#                                         [1.0, 0.8250000000000001],
#                                     ],
#                                     velocity=[
#                                         [1.337918590523148e-14, 0.004931323967907031],
#                                         [8.615538908774422e-15, 0.006930108317141729],
#                                         [7.52868309149294e-15, 0.007855784662715764],
#                                         [8.79464240569385e-15, 0.009688271686523272],
#                                         [6.684982725866203e-15, 0.010449429078930126],
#                                     ],
#                                     pressure=[
#                                         7938.501675547604,
#                                         6198.230935398052,
#                                         4399.142055600923,
#                                         2585.310932375168,
#                                         976.2032997317216,
#                                     ])

#         result_endpoint = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0], 5, semi,
#                                                           fluid_system, sol, endpoint=false,
#                                                           cut_off_bnd=true)
#         compare_interpolation_result(result_endpoint, expected_result_endpoint)
#     end
# end
