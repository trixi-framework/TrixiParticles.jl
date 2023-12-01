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
    @testset verbose=true "Interpolation Plane" begin
        lower_left = [0.0, 0.0]
        top_right = [1.0, 1.0]
        resolution = 0.2

        result = TrixiParticles.interpolate_plane(lower_left, top_right, resolution, semi,
                                                  fluid_system, sol)

        # # Define the expected results for comparison
        expected_result = [
            (density=1010.1686159336, neighbor_count=16, coord=[0.2, 0.0],
             velocity=[-0.0009327192172939572, 0.002909192963276262],
             pressure=9257.923696807055),
            (density=1010.1462386780697, neighbor_count=16, coord=[0.4, 0.0],
             velocity=[4.59840521584055e-5, 0.0028755077876233366],
             pressure=9236.910134121725),
            (density=1010.1560848938921, neighbor_count=16, coord=[0.6, 0.0],
             velocity=[0.00010838113109092406, 0.00292450541870722],
             pressure=9246.148204113459),
            (density=1010.151573725781, neighbor_count=16, coord=[0.8, 0.0],
             velocity=[0.00010801847176297891, 0.002948099358945184],
             pressure=9241.91602891712),
            (density=1010.1591108027769, neighbor_count=16, coord=[1.0, 0.0],
             velocity=[8.868604289943732e-15, 0.002876825373832205],
             pressure=9248.978120690132),
            (density=1007.9967989965918, neighbor_count=16, coord=[0.0, 0.2],
             velocity=[-0.0005028044536040427, 0.0035457731040680466],
             pressure=7233.466763518747),
            (density=1007.9602756587348, neighbor_count=16, coord=[0.2, 0.2],
             velocity=[-0.00035540353391662387, 0.004974385314877303],
             pressure=7199.642559864156),
            (density=1007.969025243849, neighbor_count=16, coord=[0.4, 0.2],
             velocity=[0.00013246243189212573, 0.005548696158795254],
             pressure=7207.753317752376),
            (density=1007.9664194380072, neighbor_count=16, coord=[0.6, 0.2],
             velocity=[0.0003287548384643027, 0.005688650253531116],
             pressure=7205.344062330894),
            (density=1007.968411919849, neighbor_count=16, coord=[0.8, 0.2],
             velocity=[0.00013992042252419147, 0.005737692463017649],
             pressure=7207.180275758381),
            (density=1007.9721356287379, neighbor_count=16, coord=[1.0, 0.2],
             velocity=[1.0639937898406559e-14, 0.0056907654660568584],
             pressure=7210.64575024627),
            (density=1005.7802387668306, neighbor_count=16, coord=[0.0, 0.4],
             velocity=[-7.061687266765948e-5, 0.0036995230536662703],
             pressure=5194.22514614874),
            (density=1005.7546755077452, neighbor_count=16, coord=[0.2, 0.4],
             velocity=[0.0007740113477530656, 0.005846335143809758],
             pressure=5170.894215223812),
            (density=1005.7613242032403, neighbor_count=16, coord=[0.4, 0.4],
             velocity=[0.0007475472656009219, 0.006916181635259845],
             pressure=5176.952699930797),
            (density=1005.7593841086034, neighbor_count=16, coord=[0.6, 0.4],
             velocity=[0.0003266584397572784, 0.007365260505270947],
             pressure=5175.181019775542),
            (density=1005.7567271676268, neighbor_count=16, coord=[0.8, 0.4],
             velocity=[0.0001431625118187707, 0.007436133797274004],
             pressure=5172.770842889837),
            (density=1005.757540996977, neighbor_count=16, coord=[1.0, 0.4],
             velocity=[8.100245816185086e-15, 0.007335450134367711],
             pressure=5173.521702742268),
            (density=1003.5265588359863, neighbor_count=16, coord=[0.0, 0.6],
             velocity=[0.0008324180396277633, 0.0026652021292882565],
             pressure=3148.2895879661364),
            (density=1003.5073740015356, neighbor_count=16, coord=[0.2, 0.6],
             velocity=[0.002826454583060609, 0.006566607831032938],
             pressure=3131.0338626666753),
            (density=1003.4922203229851, neighbor_count=16, coord=[0.4, 0.6],
             velocity=[0.0017647005482168605, 0.008626472276613315],
             pressure=3117.334321569359),
            (density=1003.4882458518805, neighbor_count=16, coord=[0.6, 0.6],
             velocity=[0.0005788316694771055, 0.009054638039309847],
             pressure=3113.780462695287),
            (density=1003.477072861037, neighbor_count=16, coord=[0.8, 0.6],
             velocity=[0.0001476293656383203, 0.009045494866674362],
             pressure=3103.709998725012),
            (density=1003.4725897077725, neighbor_count=16, coord=[1.0, 0.6],
             velocity=[7.63170602200834e-15, 0.009154849467188985],
             pressure=3099.6721353686776),
            (density=1001.3196346982327, neighbor_count=12, coord=[0.2, 0.8],
             velocity=[0.009102116389801993, 0.011820668684509835],
             pressure=1170.8694317362454),
            (density=1001.3036507399495, neighbor_count=12, coord=[0.4, 0.8],
             velocity=[0.0023667996884615575, 0.011172085099320939],
             pressure=1156.6534267237446),
            (density=1001.2911263974993, neighbor_count=12, coord=[0.6, 0.8],
             velocity=[0.0007521130361000565, 0.010410223006966842],
             pressure=1145.5035751765408),
            (density=1001.2858693148222, neighbor_count=12, coord=[0.8, 0.8],
             velocity=[9.136967189698641e-5, 0.010270354621482288],
             pressure=1140.8349235468484),
            (density=1001.2846096528532, neighbor_count=12, coord=[1.0, 0.8],
             velocity=[7.398790653432058e-15, 0.010258746201217956],
             pressure=1139.7084096362842),
        ]

        compare_interpolation_results(result, expected_result)
    end
end
