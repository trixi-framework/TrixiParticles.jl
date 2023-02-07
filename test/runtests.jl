using Test
using Pixie
using LinearAlgebra
using Printf

include("test_util.jl")

@testset "Tests" verbose=true begin
    @testset "Unit and Integration Tests" verbose=true begin
        include("containers/solid_container.jl")
        include("interactions/solid.jl")
    end

    include("system_tests.jl")
end
