using OrdinaryDiffEq
include("../test_util.jl")

@testset verbose=true "SPH Interpolation" begin
    struct MockSolution
        u::Vector{MockState}
    end

    struct MockState
        x::Vector{Any}
    end

    particle_spacing = 0.1
    smoothing_length = 1.2 * 0.1
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    sound_speed = 10 * sqrt(9.81 * 0.9)

    state_equation = StateEquationCole(sound_speed, 7, 1000, 100000.0,
                                       background_pressure=100000.0,
                                       clip_negative_pressure=false)

    # tank = RectangularTank(0.1, (2.0, 0.9), (2.0, 1.0), 1000, n_layers=4,
    #                        acceleration=(0.0, -9.81), state_equation=state_equation)
    fluid = rectangular_patch(particle_spacing, (10, 10), seed=1, perturbation_factor = 0.0)

    viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity=viscosity,
                                               acceleration=(0.0, -9.81))

    # boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
    #                                              state_equation=state_equation,
    #                                              AdamiPressureExtrapolation(),
    #                                              smoothing_kernel, smoothing_length)

    # boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

    semi = Semidiscretization(fluid_system, neighborhood_search=GridNeighborhoodSearch)

    mock_state = MockState([mock_properties, fluid.coordinates])
    mock_sol = MockSolution([mock_state])

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

    @testset verbose=true "Interpolation Point" begin
        result = TrixiParticles.interpolate_point([1.0, 0.01], semi, fluid_system, mock_sol,
                                                  cut_off_bnd=true)
        println(result)

    #     expected_result = (density=1009.4921501393262, neighbor_count=16, coord=[1.0, 0.01],
    #                        velocity=[1.7405906444065288e-14, 0.005661512837091638],
    #                        pressure=8623.500764832543)
    #     compare_interpolation_result(result, expected_result)

    #     # with deactivated cut_off_bnd
    #     # results extend outside of the fluid domain
    #     result = TrixiParticles.interpolate_point([1.0, 0.01], semi, fluid_system, sol,
    #                                               cut_off_bnd=false)

    #     expected_result = (density=1009.4921501393262, neighbor_count=8, coord=[1.0, 0.01],
    #                        velocity=[1.7405906444065288e-14, 0.005661512837091638],
    #                        pressure=8623.500764832543)
    #     compare_interpolation_result(result, expected_result)
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
