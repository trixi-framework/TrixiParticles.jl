using Test
using TrixiParticles
using LinearAlgebra
using Printf
using QuadGK: quadgk

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

        local time_start = time_ns()

        @eval module $mod
        using Test
        using TrixiParticles

        @testset verbose=true $name $expr
        end

        nothing
    end
end
