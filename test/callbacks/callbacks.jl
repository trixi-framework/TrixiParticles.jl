@testset verbose=true "Callbacks" begin
    @testset verbose=true "set_callbacks_used!" begin
        # Test that the `set_callbacks_used!` correctly detects the presence
        # of a `SplitIntegrationCallback` and an `UpdateCallback` in the list of callbacks,
        # and sets `integrate_tlsph` and `update_callback_used` accordingly.

        # We need a mock `integrator` that supports
        # `integrator.opts.callback.discrete_callbacks`.
        function make_integrator(discrete_callbacks)
            # We also need a mock `Semidiscretization` that supports `semi.integrate_tlsph[]`
            # and `semi.update_callback_used[]`.
            semi = (; integrate_tlsph=Ref(true), update_callback_used=Ref(false))

            return (; opts=(; callback=(; discrete_callbacks)), p=semi)
        end

        stepsize_cb = StepsizeCallback(cfl=1.2)
        split_cb = SplitIntegrationCallback(nothing)
        update_cb = UpdateCallback()
        update_cb_periodic = UpdateCallback(dt=0.1)

        integrator_without_split = make_integrator([stepsize_cb])
        integrator_with_split = make_integrator([stepsize_cb, split_cb])
        integrator_with_update = make_integrator([stepsize_cb, update_cb])
        integrator_with_update_periodic = make_integrator([stepsize_cb, update_cb_periodic])
        integrator_with_both = make_integrator([stepsize_cb, split_cb, update_cb])

        semi_without_split = integrator_without_split.p
        semi_with_split = integrator_with_split.p
        semi_with_update = integrator_with_update.p
        semi_with_update_periodic = integrator_with_update_periodic.p
        semi_with_both = integrator_with_both.p

        TrixiParticles.set_callbacks_used!(semi_without_split, integrator_without_split)
        TrixiParticles.set_callbacks_used!(semi_with_split, integrator_with_split)
        TrixiParticles.set_callbacks_used!(semi_with_update, integrator_with_update)
        TrixiParticles.set_callbacks_used!(semi_with_update_periodic,
                                           integrator_with_update_periodic)
        TrixiParticles.set_callbacks_used!(semi_with_both, integrator_with_both)

        @test semi_without_split.integrate_tlsph[] == true
        @test semi_without_split.update_callback_used[] == false
        @test semi_with_split.integrate_tlsph[] == false
        @test semi_with_split.update_callback_used[] == false
        @test semi_with_update.integrate_tlsph[] == true
        @test semi_with_update.update_callback_used[] == true
        @test semi_with_update_periodic.integrate_tlsph[] == true
        @test semi_with_update_periodic.update_callback_used[] == true
        @test semi_with_both.integrate_tlsph[] == false
        @test semi_with_both.update_callback_used[] == true
    end

    include("info.jl")
    include("stepsize.jl")
    include("postprocess.jl")
    include("update.jl")
    include("solution_saving.jl")
    include("steady_state_reached.jl")
end
