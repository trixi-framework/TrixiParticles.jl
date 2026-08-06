using CairoMakie

@testset verbose=true "Makie Extension" begin
    initial_condition = RectangularShape(0.1, (2, 2, 2), (0.0, 0.0, 0.0);
                                         density=1.0)
    fluid_system = WeaklyCompressibleSPHSystem(initial_condition;
                                               smoothing_kernel=SchoenbergCubicSplineKernel{3}(),
                                               smoothing_length=0.1,
                                               density_calculator=SummationDensity(),
                                               state_equation=nothing)
    semi = Semidiscretization(fluid_system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x

    figure = Figure(; size=(320, 240))
    axis = LScene(figure[1, 1]; show_axis=false)
    plots = trixi2makie(axis, v_ode, u_ode, semi)

    @test Base.get_extension(TrixiParticles, :TrixiParticlesMakieExt) !== nothing
    @test length(plots) == 1
    @test only(plots) isa CairoMakie.MeshScatter
    @test length(only(plots)[1][]) == nparticles(fluid_system)
end
