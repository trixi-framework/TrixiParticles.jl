@testset verbose=true "Density Diffusion" begin
    @testset verbose=true "DensityDiffusionAntuono" begin
        # Use `@trixi_testset` to isolate the mock functions in a separate namespace
        @trixi_testset "show" begin
            # Mock initial condition
            initial_condition = Val(:initial_condition)
            Base.ndims(::Val{:initial_condition}) = 2
            Base.eltype(::Val{:initial_condition}) = Float64
            TrixiParticles.nparticles(::Val{:initial_condition}) = 15

            density_diffusion = DensityDiffusionAntuono(delta=0.1, initial_condition)

            @test repr(density_diffusion) == "DensityDiffusionAntuono(0.1)"
        end
    end
end
