@testset verbose=true "Density Diffusion" begin
    @testset verbose=true "DensityDiffusionAntuono" begin
        # Use `@trixi_testset` to isolate the mock functions in a separate namespace
        @trixi_testset "show" begin
            density_diffusion = DensityDiffusionAntuono(delta=0.1)

            @test repr(density_diffusion) ==
                  "DensityDiffusionAntuono(delta=0.1, update_everystage=true)"

            callback_density_diffusion = DensityDiffusionAntuono(delta=0.1,
                                                                 update_everystage=false)
            @test repr(callback_density_diffusion) ==
                  "DensityDiffusionAntuono(delta=0.1, update_everystage=false)"
            @test !TrixiParticles.requires_update_callback(density_diffusion)
            @test TrixiParticles.requires_update_callback(callback_density_diffusion)
        end
    end
end
