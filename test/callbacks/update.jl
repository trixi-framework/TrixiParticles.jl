@testset verbose=true "UpdateCallback" begin
    @testset verbose=true "show" begin
        # Default
        callback0 = UpdateCallback()

        show_compact = "UpdateCallback(interval=1)"
        @test repr(callback0) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ UpdateCallback                                                                                   │
        │ ══════════════                                                                                   │
        │ interval: ……………………………………………………… 1                                                                │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback0) == show_box

        callback1 = UpdateCallback(interval=11)

        show_compact = "UpdateCallback(interval=11)"
        @test repr(callback1) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ UpdateCallback                                                                                   │
        │ ══════════════                                                                                   │
        │ interval: ……………………………………………………… 11                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback1) == show_box

        callback2 = UpdateCallback(dt=1.2)

        show_compact = "UpdateCallback(dt=1.2)"
        @test repr(callback2) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ UpdateCallback                                                                                   │
        │ ══════════════                                                                                   │
        │ dt: ……………………………………………………………………… 1.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback2) == show_box
    end

    @testset "Illegal Input" begin
        error_str = "Setting both interval and dt is not supported!"
        @test_throws ArgumentError(error_str) UpdateCallback(dt=0.1, interval=1)
    end
end
