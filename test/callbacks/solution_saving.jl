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
end
