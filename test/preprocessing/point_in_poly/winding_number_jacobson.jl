@testset verbose=true "WindingNumberJacobson" begin
    @testset verbose=true "Show" begin
        data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
        geometry = load_geometry(joinpath(data_dir, "circle.asc"))

        winding = WindingNumberJacobson()

        show_compact = "WindingNumberJacobson{NaiveWinding}()"
        @test repr(winding) == show_compact

        winding = WindingNumberJacobson(; hierarchical_winding=false)

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

        winding = WindingNumberJacobson(; geometry, winding_number_factor=pi,
                                        hierarchical_winding=true)
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

    @testset verbose=true "Point Matrix Input" begin
        geometry = TrixiParticles.Polygon([0.0 1.0 1.0 0.0 0.0;
                                           0.0 0.0 1.0 1.0 0.0])
        points = [0.5 1.5;
                  0.5 1.5]

        expected = Bool[true, false]

        inpoly_jacobson, _ = WindingNumberJacobson()(geometry, points)
        inpoly_hormann, _ = WindingNumberHormann()(geometry, points)

        @test inpoly_jacobson == expected
        @test inpoly_hormann == expected
    end
end
