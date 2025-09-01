@testset verbose=true "Shifting Techniques" begin
    @testset "Constructors" begin
        @test_nowarn TransportVelocityAdami()
        @test_nowarn ParticleShiftingTechniqueSun2017()
        @test_nowarn ConsistentShiftingSun2019()
    end
end
