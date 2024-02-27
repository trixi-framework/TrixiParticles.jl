@testset verbose=true "PostprocessCallback" begin
    @testset verbose=true "show" begin
        function example_function(v, u, t, system)
            return 0
        end

        callback = PostprocessCallback(another_function=(v, u, t, system) -> 1; interval=10,
                                       example_function, write_file_interval=0)

        show_compact = "PostprocessCallback(interval=10, functions=[another_function, example_function])"
        @test repr(callback) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ interval: ……………………………………………………… 10                                                               │
        │ write file: ………………………………………………… no                                                               │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… another_function                                                 │
        │ function2: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; dt=0.1, example_function, write_file_interval=0)

        show_compact = "PostprocessCallback(dt=0.1, functions=[example_function])"
        @test repr(callback) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ dt: ……………………………………………………………………… 0.1                                                              │
        │ write file: ………………………………………………… no                                                               │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; dt=0.1, example_function, write_file_interval=3)
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ dt: ……………………………………………………………………… 0.1                                                              │
        │ write file: ………………………………………………… every 3 * dt                                                     │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; interval=23, example_function,
                                       write_file_interval=4)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ interval: ……………………………………………………… 23                                                               │
        │ write file: ………………………………………………… every 4 * interval                                               │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; interval=23, example_function)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ interval: ……………………………………………………… 23                                                               │
        │ write file: ………………………………………………… always                                                           │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; dt=0.2, example_function)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ dt: ……………………………………………………………………… 0.2                                                              │
        │ write file: ………………………………………………… always                                                           │
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
