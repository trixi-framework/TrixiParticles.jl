using Test
using Pixie
using LinearAlgebra
using Printf

include("test_util.jl")

@testset "All Tests" verbose=true begin
    include("unit/unit_tests.jl")
    include("integration/integration_tests.jl")
    include("system/system_tests.jl")
end
