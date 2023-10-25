
@testset verbose=true "Dummy Particles" begin
    @testset "show" begin
        boundary_model = BoundaryModelMonaghanKajtar(10.0, 3.0, 0.1, [1.0])

        show_compact = "BoundaryModelMonaghanKajtar(10.0, 3.0, NoViscosity)"
        @test repr(boundary_model) == show_compact
    end
end
