using Test
using TrixiParticles
using LinearAlgebra
using Printf
using QuadGK: quadgk # For integration in smoothing kernel tests
using Random: Random # For rectangular patch
using Polyester: disable_polyester_threads # For `count_rhs_allocations`

"""
    @trixi_testset "name of the testset" #= code to test #=

Similar to `@testset`, but wraps the code inside a temporary module to avoid
namespace pollution.
"""
macro trixi_testset(name, expr)
    @assert name isa String

    mod = gensym()

    # TODO: `@eval` is evil
    quote
        println("═"^100)
        println($name)

        @eval module $mod
        using Test
        using TrixiParticles

        # We also include this file again to provide the definition of
        # the other testing macros. This allows to use `@trixi_testset`
        # in a nested fashion and also call `@test_nowarn_mod` from
        # there.
        include(@__FILE__)

        @testset verbose=true $name $expr
        end

        nothing
    end
end

include("count_allocations.jl")
include("rectangular_patch.jl")
