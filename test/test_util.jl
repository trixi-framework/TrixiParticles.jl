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
        local time_start = time_ns()

        @eval module $mod
        using Test
        using TrixiParticles

        @testset $name $expr
        end

        nothing
    end
end
