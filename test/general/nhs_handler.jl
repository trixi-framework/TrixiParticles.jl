using OrdinaryDiffEq

@testset verbose=true "Test NHS Handlers" begin
    nhs_handlers = (
                    # PairsNHSHandler, 
                    # GridNHSHandler, 
                    VariableNHSHandler,)

    @testset verbose=true "$nhs_handler" for nhs_handler in nhs_handlers

        # Run the `dam_break_gate_2d.jl` file for 0 timesteps to instantiate all systems
        sol = trixi_include(@__MODULE__,
                            joinpath(examples_dir(), "fsi", "dam_break_gate_2d.jl"),
                            tspan=(0.0, 0.0),
                            callbacks=OrdinaryDiffEq.CallbackSet())

        # sol.prob.p is the original semidiscretization
        semi = sol.prob.p
        systems = semi.systems
        nhs = TrixiParticles.get_neighborhood_search(first(systems), semi)
        neighborhood_search_handler = nhs_handler(nhs, systems)

        new_semi = Semidiscretization(systems...,
                                      neighborhood_search_handler=neighborhood_search_handler,
                                      parallelization_backend=PolyesterBackend())

        tspan_test = (0.0, 0.1)
        ode = semidiscretize(new_semi, tspan_test)

        new_sol = Base.invokelatest(solve, ode, RDPK3SpFSAL49();
                                    abstol=1e-6,
                                    reltol=1e-4,
                                    dtmax=1e-3,
                                    save_everystep=false)

        @test new_sol.retcode == ReturnCode.Success
    end
end
