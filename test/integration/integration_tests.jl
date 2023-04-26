include("../test_util.jl")

@testset verbose=true "Integration Tests" begin
    include("containers/fluid_container.jl")
    include("interactions/interactions.jl")
end
