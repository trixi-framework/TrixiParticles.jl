@testset verbose=true "TrivialNeighborhoodSearch" begin
    # Setup with 5 particles
    nhs = TrixiParticles.TrivialNeighborhoodSearch{2}(1.0, Base.OneTo(5))

    # Get each neighbor for arbitrary coordinates
    neighbors = collect(TrixiParticles.eachneighbor([1.0, 2.0], nhs))

    #### Verification
    @test neighbors == [1, 2, 3, 4, 5]
end
