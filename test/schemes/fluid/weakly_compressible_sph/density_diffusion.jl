@testset verbose=true "Density Diffusion" begin
    @testset verbose=true "DensityDiffusionAntuono" begin
        # Use `@trixi_testset` to isolate the mock functions in a separate namespace
        @trixi_testset "show" begin
            density_diffusion = DensityDiffusionAntuono(delta=0.1)

            @test repr(density_diffusion) == "DensityDiffusionAntuono(0.1)"
        end
    end
end
