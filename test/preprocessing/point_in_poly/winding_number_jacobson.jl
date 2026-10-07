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

        winding = WindingNumberJacobson(; geometry, winding_number_factor=pi)
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
        point_storage = [0.5 0.0 1.5;
                         0.5 0.0 1.5]
        points = @view point_storage[:, 1:2:3]

        expected = Bool[true, false]

        inpoly_jacobson, _ = WindingNumberJacobson()(geometry, points)
        inpoly_hormann, _ = WindingNumberHormann()(geometry, points)

        @test inpoly_jacobson == expected
        @test inpoly_hormann == expected
    end

    @testset verbose=true "Open Geometry Validation" begin
        open_square = [0.0 1.0 1.0 0.0;
                       0.0 0.0 1.0 1.0]
        geometry = TrixiParticles.Polygon(open_square; close_curve=false)
        points = [SVector(0.5, 0.5)]

        jacobson = WindingNumberJacobson(; hierarchical_winding=false)
        hormann = WindingNumberHormann()

        @test jacobson(geometry, points)[1] isa Vector{Bool}
        @test hormann(geometry, points)[1] isa Vector{Bool}
    end
end
