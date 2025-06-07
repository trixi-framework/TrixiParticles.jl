@testset verbose=true "WindingNumberJacobson" begin
    @testset verbose=true "Show" begin
        data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
        geometry = load_geometry(joinpath(data_dir, "circle.asc"))

        winding = WindingNumberJacobson(geometry; winding=NaiveWinding())

        show_compact = "WindingNumberJacobson{NaiveWinding}()"
        @test repr(winding) == show_compact

        show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ WindingNumberJacobson                                                                            │
            │ ═════════════════════                                                                            │
            │ winding number factor: …………………… 0.0                                                              │
            │ winding: ………………………………………………………… NaiveWinding                                                     │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", winding) == show_box

        winding = WindingNumberJacobson(geometry; winding_number_factor=pi)
        show_compact = "WindingNumberJacobson{HierarchicalWinding}()"
        @test repr(winding) == show_compact

        show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ WindingNumberJacobson                                                                            │
            │ ═════════════════════                                                                            │
            │ winding number factor: …………………… 3.142                                                            │
            │ winding: ………………………………………………………… HierarchicalWinding                                              │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", winding) == show_box
    end
end
