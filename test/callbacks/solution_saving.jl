@testset verbose=true "SolutionSavingCallback" begin
    @testset verbose=true "show" begin
        out = joinpath(pkgdir(TrixiParticles), "out")
        output_directory_padded = out * " "^(65 - length(out))

        @testset verbose=true "dt" begin
            callback = SolutionSavingCallback(dt=0.02, prefix="test", output_directory=out)

            show_compact = "SolutionSavingCallback(dt=0.02)"
            @test repr(callback) == show_compact

            show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ SolutionSavingCallback                                                                           │
            │ ══════════════════════                                                                           │
            │ dt: ……………………………………………………………………… 0.02                                                             │
            │ custom quantities: ……………………………… nothing                                                          │
            │ save initial solution: …………………… yes                                                              │
            │ save final solution: ………………………… yes                                                              │
            │ output directory: ………………………………… $(output_directory_padded)│
            │ prefix: …………………………………………………………… test                                                             │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
            @test repr("text/plain", callback) == show_box
        end

        @testset verbose=true "interval" begin
            callback = SolutionSavingCallback(interval=100, prefix="test",
                                              output_directory=out)

            show_compact = "SolutionSavingCallback(interval=100)"
            @test repr(callback) == show_compact

            show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ SolutionSavingCallback                                                                           │
            │ ══════════════════════                                                                           │
            │ interval: ……………………………………………………… 100                                                              │
            │ custom quantities: ……………………………… nothing                                                          │
            │ save initial solution: …………………… yes                                                              │
            │ save final solution: ………………………… yes                                                              │
            │ output directory: ………………………………… $(output_directory_padded)│
            │ prefix: …………………………………………………………… test                                                             │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
            @test repr("text/plain", callback) == show_box
        end

        @testset verbose=true "save_times" begin
            callback = SolutionSavingCallback(save_times=[1.0, 2.0, 3.0], prefix="test",
                                              output_directory=out)

            show_compact = "SolutionSavingCallback(save_times=[1.0, 2.0, 3.0])"
            @test repr(callback) == show_compact

            show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ SolutionSavingCallback                                                                           │
            │ ══════════════════════                                                                           │
            │ save_times: ………………………………………………… [1.0, 2.0, 3.0]                                                  │
            │ custom quantities: ……………………………… nothing                                                          │
            │ save initial solution: …………………… yes                                                              │
            │ save final solution: ………………………… yes                                                              │
            │ output directory: ………………………………… $(output_directory_padded)│
            │ prefix: …………………………………………………………… test                                                             │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
            @test repr("text/plain", callback) == show_box
        end
    end

    @testset verbose=true "custom quantities" begin
        # Test that `custom_quantity` correctly chooses the correct method
        quantity1(system, data, t) = data
        quantity2(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = 2
        quantity3() = 3

        system = Val(:mock_system)
        TrixiParticles.system_data(::Val{:mock_system}, dv_ode, du_ode, v_ode, u_ode,
                                   semi) = 1

        data = v_ode = u_ode = dv_ode = du_ode = semi = t = nothing

        @test TrixiParticles.custom_quantity(quantity1, system, dv_ode, du_ode, v_ode,
                                             u_ode, semi, t) == 1
        @test TrixiParticles.custom_quantity(quantity2, system, dv_ode, du_ode, v_ode,
                                             u_ode, semi, t) == 2
        @test_throws MethodError TrixiParticles.custom_quantity(quantity3, system, dv_ode,
                                                                du_ode, v_ode, u_ode,
                                                                semi, t)
    end
end
