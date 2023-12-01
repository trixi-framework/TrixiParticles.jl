using OrdinaryDiffEq
include("../test_util.jl")

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

    function compare_interpolation_result(actual, expected; tolerance=1e-12)
        @test actual.neighbor_count == expected.neighbor_count
        @test actual.coord == expected.coord
        @test isapprox(actual.density, expected.density, atol=tolerance)
        @test isapprox(actual.velocity, expected.velocity, atol=tolerance)
        @test isapprox(actual.pressure, expected.pressure, atol=tolerance)
    end

    function compare_interpolation_results(actuals, expecteds; tolerance=1e-12)
        @test length(actuals) == length(expecteds)
        for (actual, expected) in zip(actuals, expecteds)
            compare_interpolation_result(actual, expected, tolerance=tolerance)
        end
    end

    @testset verbose=true "Interpolation Point" begin
        result = TrixiParticles.interpolate_point([1.0, 0.01], semi, fluid_system, sol)

        expected_result = (density=1010.0529725852786, neighbor_count=16, coord=[1.0, 0.01],
                           velocity=[9.815064195161356e-15, 0.0031924859596571607],
                           pressure=9149.453039221622)
        compare_interpolation_result(result, expected_result)
    end
    @testset verbose=true "Interpolation Multi Point" begin
        result = TrixiParticles.interpolate_point([
                                                      [1.0, 0.01],
                                                      [1.0, 0.1],
                                                      [1.0, 0.0],
                                                      [1.0, -0.01],
                                                      [1.0, -0.05],
                                                  ], semi, fluid_system, sol)

        expected_result = [
            (density=1010.0529725852786, neighbor_count=16, coord=[1.0, 0.01],
             velocity=[9.815064195161356e-15, 0.0031924859596571607],
             pressure=9149.453039221622),
            (density=1009.0647585308516, neighbor_count=16, coord=[1.0, 0.1],
             velocity=[1.3682298856802571e-14, 0.004737022064423743],
             pressure=8225.817113637197),
            (density=1010.1591108027769, neighbor_count=16, coord=[1.0, 0.0],
             velocity=[8.868604289943732e-15, 0.002876825373832205],
             pressure=9248.978120690132),
            (density=0.0, neighbor_count=0, coord=[1.0, -0.01], velocity=[0.0, 0.0],
             pressure=0.0),
            (density=0.0, neighbor_count=0, coord=[1.0, -0.05], velocity=[0.0, 0.0],
             pressure=0.0),
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
                                                 fluid_system, sol)
        compare_interpolation_results(result, expected_result)

        expected_result_endpoint = [
            (density=1008.7906994450206, neighbor_count=18,
             coord=[1.0, 0.12499999999999999],
             velocity=[1.314244588843225e-14, 0.0048440659144444315],
             pressure=7970.563443828845),
            (density=1006.8750741471611, neighbor_count=16, coord=[1.0, 0.3],
             velocity=[8.615538908774422e-15, 0.006930108317141729],
             pressure=6198.230935398052),
            (density=1004.907892717609, neighbor_count=18, coord=[1.0, 0.47500000000000003],
             velocity=[7.52868309149294e-15, 0.007855784662715764],
             pressure=4399.142055600923),
            (density=1002.9010166322735, neighbor_count=16, coord=[1.0, 0.6499999999999999],
             velocity=[8.79464240569385e-15, 0.009688271686523272],
             pressure=2585.310932375168),
            (density=1001.1009176645862, neighbor_count=14, coord=[1.0, 0.8250000000000001],
             velocity=[6.6843891671872114e-15, 0.010448501275587224],
             pressure=976.1166227776027),
        ]
        result_endpoint = TrixiParticles.interpolate_line([1.0, -0.05], [1.0, 1.0], 5, semi,
                                                          fluid_system, sol, endpoint=false)
        compare_interpolation_results(result_endpoint, expected_result_endpoint)
    end
end
