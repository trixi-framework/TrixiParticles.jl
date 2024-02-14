@testset verbose=true "PostprocessCallback" begin
    @testset verbose=true "show" begin
        function example_function(v, u, t, system)
            return 0
        end

        callback = PostprocessCallback(example_function, interval=10)

        show_compact = "PostprocessCallback(interval=10, functions=[example_function])"
        @test repr(callback) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ interval: ……………………………………………………… 10                                                               │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box
    end
end
