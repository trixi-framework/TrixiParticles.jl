using CairoMakie

@testset verbose=true "Makie Extension" begin
    initial_condition = RectangularShape(0.1, (2, 2), (0.0, 0.0); density=1.0)
    fluid_system = WeaklyCompressibleSPHSystem(initial_condition;
                                               smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                               smoothing_length=0.1,
                                               density_calculator=SummationDensity(),
                                               state_equation=nothing)
    semi = Semidiscretization(fluid_system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x

    makie_extension = Base.get_extension(TrixiParticles, :TrixiParticlesMakieExt)
    @test makie_extension !== nothing

    figure, axis, plot_object = plot(v_ode, u_ode, semi)
    @test figure isa Figure
    @test axis isa Axis
    @test length(plot_object.plots) == 1
    @test only(plot_object.plots) isa CairoMakie.MeshScatter
    @test length(only(plot_object.plots)[1][]) == nparticles(fluid_system)

    figure = Figure(; size=(320, 240))
    axis = Axis(figure[1, 1])
    plot_object = trixi2makie!(axis, v_ode, u_ode, semi)
    @test plot_object isa makie_extension.Trixi2Makie

    solution = TrixiParticles.SciMLBase.build_solution(ode, :NoAlgorithm,
                                                       [first(ode.tspan)], [ode.u0])
    figure, axis, plot_object = plot(solution)
    @test figure isa Figure
    @test axis isa Axis
    @test plot_object isa makie_extension.Trixi2Makie

    figure = Figure(; size=(320, 240))
    axis = Axis(figure[1, 1])
    @test plot!(axis, solution) isa makie_extension.Trixi2Makie

    initial_condition_3d = RectangularShape(0.1, (2, 2, 2), (0.0, 0.0, 0.0);
                                            density=1.0)
    fluid_system_3d = WeaklyCompressibleSPHSystem(initial_condition_3d;
                                                  smoothing_kernel=SchoenbergCubicSplineKernel{3}(),
                                                  smoothing_length=0.1,
                                                  density_calculator=SummationDensity(),
                                                  state_equation=nothing)
    semi_3d = Semidiscretization(fluid_system_3d)
    ode_3d = semidiscretize(semi_3d, (0.0, 0.01))
    v_ode_3d, u_ode_3d = ode_3d.u0.x

    figure, axis, plot_object = plot(v_ode_3d, u_ode_3d, semi_3d)
    @test figure isa Figure
    @test axis isa Axis3
    @test only(plot_object.plots) isa CairoMakie.MeshScatter
    @test length(only(plot_object.plots)[1][]) == nparticles(fluid_system_3d)

    initial_condition_3d_2 = RectangularShape(0.1, (2, 2, 2), (0.3, 0.0, 0.0);
                                              density=1.0)
    fluid_system_3d_2 = WeaklyCompressibleSPHSystem(initial_condition_3d_2;
                                                    smoothing_kernel=SchoenbergCubicSplineKernel{3}(),
                                                    smoothing_length=0.1,
                                                    density_calculator=SummationDensity(),
                                                    state_equation=nothing)
    semi_3d_2 = Semidiscretization(fluid_system_3d, fluid_system_3d_2)
    ode_3d_2 = semidiscretize(semi_3d_2, (0.0, 0.01))
    v_ode_3d_2, u_ode_3d_2 = ode_3d_2.u0.x

    _, _, plot_object = plot(v_ode_3d_2, u_ode_3d_2, semi_3d_2;
                             system_colors=[:blue, :orange],
                             marker_size_scales=[0.8, 0.5])
    @test length(plot_object.plots) == 1
    meshscatter = only(plot_object.plots)
    @test length(meshscatter[1][]) ==
          nparticles(fluid_system_3d) + nparticles(fluid_system_3d_2)
    @test length(meshscatter.color[]) == length(meshscatter[1][])
    @test length(meshscatter.markersize[]) == length(meshscatter[1][])
end
