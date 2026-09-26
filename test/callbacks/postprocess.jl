@testset verbose=true "PostprocessCallback" begin
    @testset "shared series JSON/CSV writer" begin
        mktempdir() do directory
            json_path = joinpath(directory, "series.json")
            csv_path = joinpath(directory, "series.csv")
            values = TrixiParticles.create_series_dict([1.0, 2.0], [0.0, 0.1],
                                                       "fluid_1")
            data = Dict("meta" => Dict("source" => "test"),
                        "measure_fluid_1" => values)
            TrixiParticles.write_time_series_files(json_path, csv_path, data;
                                                   save_csv=false)
            @test isfile(json_path) && !isfile(csv_path)
            @test TrixiParticles.JSON.parsefile(json_path)["measure_fluid_1"]["values"] ==
                  [1.0, 2.0]
            TrixiParticles.write_time_series_files(json_path, csv_path, data;
                                                   save_json=false)
            @test isfile(csv_path)
            @test occursin("measure_fluid_1", first(readlines(csv_path)))
        end
    end

    @testset verbose=true "errors" begin
        error_str1 = "`funcs` cannot be empty"
        @test_throws ArgumentError(error_str1) PostprocessCallback(interval=10,
                                                                   write_file_interval=0)

        error_str2 = "setting both `interval` and `dt` is not supported"
        @test_throws ArgumentError(error_str2) PostprocessCallback(interval=10,
                                                                   write_file_interval=0,
                                                                   dt=0.1,
                                                                   another_function=(system,
                                                                                     v_ode,
                                                                                     u_ode,
                                                                                     semi,
                                                                                     t) -> 1)
    end

    @testset verbose=true "reset custom quantity" begin
        @test TrixiParticles.reset_custom_quantity!(identity) === identity

        callback = PostprocessCallback(; interval=1, example_quantity=identity)
        postprocess_callback = callback.affect!

        # Add some data to make sure it gets cleared.
        postprocess_callback.data["example_quantity"] = Any[1.0]
        push!(postprocess_callback.times, 1.0)

        semi = nothing
        integrator = (; p=(; semi), opts=(; callback=(; discrete_callbacks=Any[])))

        function (callback::TrixiParticles.PostprocessCallback)(::typeof(integrator))
            return callback
        end

        TrixiParticles.set_callbacks_used!(semi::Nothing, integrator) = nothing

        TrixiParticles.initialize_postprocess_callback!(postprocess_callback, nothing,
                                                        0.0, integrator)
        @test isempty(postprocess_callback.data)
        @test isempty(postprocess_callback.times)
    end

    @testset verbose=true "show" begin
        function example_function(system, v_ode, u_ode, semi, t)
            return 0
        end

        callback = PostprocessCallback(another_function=(system, v_ode, u_ode, semi,
                                                         t) -> 1;
                                       interval=10,
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
