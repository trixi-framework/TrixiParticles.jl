using Test
using Pixie
using LinearAlgebra

include("containers/solid_container.jl")
include("interactions/solid.jl")
include("system_tests.jl")
include("sph/neighborhood_search.jl") # include last since mocked variables are not isolated yet.
