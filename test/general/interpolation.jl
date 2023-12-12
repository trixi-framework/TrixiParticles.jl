using OrdinaryDiffEq

@testset verbose=true "SPHInterpolation" begin
    smoothing_length = 1.2 * 0.1
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    sound_speed = 10 * sqrt(9.81 * 0.9)

    state_equation = StateEquationCole(sound_speed, 7, 1000, 100000.0,
                                       background_pressure=100000.0,
                                       clip_negative_pressure=false)

    tank = RectangularTank(0.1, (2.0, 0.9), (2.0, 1.0), 1000, n_layers=4,
                           acceleration=(0.0, -9.81), state_equation=state_equation)

    viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

    fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, ContinuityDensity(),
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity=viscosity,
                                               acceleration=(0.0, -9.81))

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                                 state_equation=state_equation,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length)

    boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

    semi = Semidiscretization(fluid_system, boundary_system,
                              neighborhood_search=GridNeighborhoodSearch)
    ode = semidiscretize(semi, (0.0, 0.1))

    sol = solve(ode, RDPK3SpFSAL49(), abstol=1e-5, reltol=1e-3, dtmax=1e-2,
                save_everystep=false)

    function compare_interpolation_result(actual, expected; tolerance=1e-8)
        @test actual.neighbor_count == expected.neighbor_count
        @test actual.coord == expected.coord
        @test isapprox(actual.density, expected.density, atol=tolerance)
        @test isapprox(actual.velocity, expected.velocity, atol=tolerance)
        @test isapprox(actual.pressure, expected.pressure, atol=tolerance)
    end

    function compare_interpolation_results(actuals, expecteds; tolerance=1e-8)
        @test length(actuals) == length(expecteds)
        for (actual, expected) in zip(actuals, expecteds)
            compare_interpolation_result(actual, expected, tolerance=tolerance)
        end
    end

    @testset verbose=true "Interpolation Point" begin
        result = TrixiParticles.interpolate_point([1.0, 0.01], semi, fluid_system, sol,
                                                  calculate_other_system_density=true)

        expected_result = (density=1009.4921501393262, neighbor_count=16, coord=[1.0, 0.01],
                           velocity=[1.7405906444065288e-14, 0.005661512837091638],
                           pressure=8623.500764832543)
        compare_interpolation_result(result, expected_result)

        # with deactivated calculate_other_system_density
        # results extend outside of the fluid domain
        result = TrixiParticles.interpolate_point([1.0, 0.01], semi, fluid_system, sol,
                                                  calculate_other_system_density=false)

        expected_result = (density=1009.4921501393262, neighbor_count=8, coord=[1.0, 0.01],
                           velocity=[1.7405906444065288e-14, 0.005661512837091638],
                           pressure=8623.500764832543)
        compare_interpolation_result(result, expected_result)
    end
    @testset verbose=true "Interpolation Multi Point" begin
        result = TrixiParticles.interpolate_point([
                                                      [1.0, 0.01],
                                                      [1.0, 0.1],
                                                      [1.0, 0.0],
                                                      [1.0, -0.01],
                                                      [1.0, -0.05],
                                                  ], semi, fluid_system, sol,
                                                  calculate_other_system_density=true)

        expected_result = [
            (density=1009.4921501393262, neighbor_count=16, coord=[1.0, 0.01],
             velocity=[1.7405906444065288e-14, 0.005661512837091638],
             pressure=8623.500764832543),
            (density=1008.9764089268859, neighbor_count=16, coord=[1.0, 0.1],
             velocity=[1.442422876475543e-14, 0.004993889596774216],
             pressure=8143.159169507029),
            (density=1009.5229791447398, neighbor_count=16, coord=[1.0, 0.0],
             velocity=[1.763368469598575e-14, 0.005720069349028347],
             pressure=8652.234332097962),
            (density=0.0, neighbor_count=0, coord=[1.0, -0.01], velocity=[0.0, 0.0],
             pressure=0.0),
            (density=0.0, neighbor_count=0, coord=[1.0, -0.05], velocity=[0.0, 0.0],
             pressure=0.0),
        ]
        compare_interpolation_results(result, expected_result)

        # with deactivated calculate_other_system_density
        # results extend outside of the fluid domain
        result = TrixiParticles.interpolate_point([
                                                      [1.0, 0.01],
                                                      [1.0, 0.1],
                                                      [1.0, 0.0],
                                                      [1.0, -0.01],
                                                      [1.0, -0.05],
                                                      [1.0, -0.1],
                                                      [1.0, -0.2],
                                                  ], semi, fluid_system, sol,
                                                  calculate_other_system_density=false)

        expected_result = [
            (density=1009.4921501393262, neighbor_count=8, coord=[1.0, 0.01],
             velocity=[1.7405906444065288e-14, 0.005661512837091638],
             pressure=8623.500764832543),
            (density=1008.9764089268859, neighbor_count=12, coord=[1.0, 0.1],
             velocity=[1.442422876475543e-14, 0.004993889596774216],
             pressure=8143.159169507029),
            (density=1009.5229791447398, neighbor_count=8, coord=[1.0, 0.0],
             velocity=[1.763368469598575e-14, 0.005720069349028347],
             pressure=8652.234332097962),
            (density=1009.5501117234779, neighbor_count=8, coord=[1.0, -0.01],
             velocity=[1.7847274797051723e-14, 0.005771554677158952],
             pressure=8677.522799474411),
            (density=1009.6243395503883, neighbor_count=6, coord=[1.0, -0.05],
             velocity=[1.8600165065214594e-14, 0.005911333559787621],
             pressure=8746.706496708202),
            (density=1009.6443775884171, neighbor_count=4, coord=[1.0, -0.1],
             velocity=[1.9714720554750176e-14, 0.005939365031527732],
             pressure=8765.385734383106),
            (density=0.0, neighbor_count=0, coord=[1.0, -0.2],
             velocity=[0.0, 0.0], pressure=0.0),
        ]
        compare_interpolation_results(result, expected_result)
    end
    @testset verbose=true "Interpolation Line" begin
        expected_result = [
            (density=0.0, neighbor_count=0, coord=[1.0, -0.05], velocity=[0.0, 0.0],
             pressure=0.0),
            (density=1007.8388079925885, neighbor_count=16,
             coord=[1.0, 0.21250000000000002],
             velocity=[1.0308096030666658e-14, 0.005907502958502652],
             pressure=7087.237427116723),
            (density=1004.907892717609, neighbor_count=18, coord=[1.0, 0.47500000000000003],
             velocity=[7.52868309149294e-15, 0.007855784662715764],
             pressure=4399.142055600923),
            (density=1001.899652412931, neighbor_count=14, coord=[1.0, 0.7375],
             velocity=[9.025768335787871e-15, 0.009933832252516267],
             pressure=1688.3585518071602),
            (density=0.0, neighbor_count=0, coord=[1.0, 1.0], velocity=[0.0, 0.0],
             pressure=0.0),
        ]

        result = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0], 5, semi,
                                                 fluid_system, sol,
                                                 calculate_other_system_density=true)
        compare_interpolation_results(result, expected_result)

        expected_result_endpoint = [
            (density=1008.7564078596042, neighbor_count=18,
             coord=[1.0, 0.12499999999999999],
             velocity=[1.337918590523148e-14, 0.004931323967907031],
             pressure=7938.501675547604),
            (density=1006.8750741471611, neighbor_count=16, coord=[1.0, 0.3],
             velocity=[8.615538908774422e-15, 0.006930108317141729],
             pressure=6198.230935398052),
            (density=1004.907892717609, neighbor_count=18, coord=[1.0, 0.47500000000000003],
             velocity=[7.52868309149294e-15, 0.007855784662715764],
             pressure=4399.142055600923),
            (density=1002.9010166322735, neighbor_count=16, coord=[1.0, 0.6499999999999999],
             velocity=[8.79464240569385e-15, 0.009688271686523272],
             pressure=2585.310932375168),
            (density=1001.1010154235912, neighbor_count=14, coord=[1.0, 0.8250000000000001],
             velocity=[6.684982725866203e-15, 0.010449429078930126],
             pressure=976.2032997317216),
        ]
        result_endpoint = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0], 5, semi,
                                                          fluid_system, sol, endpoint=false,
                                                          calculate_other_system_density=true)
        compare_interpolation_results(result_endpoint, expected_result_endpoint)
    end
end
