using Test
using TrixiTest: @trixi_test_nowarn
using TrixiParticles
using TrixiParticles: PointNeighbors
using LinearAlgebra
using Printf
using CSV: CSV
using DataFrames: DataFrame
using JSON: JSON
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
        # in a nested fashion and also call `@trixi_test_nowarn` from
        # there.
        include(@__FILE__)

        @testset verbose=true $name $expr
        end

        nothing
    end
end

struct DummySemidiscretization
    parallelization_backend :: Any
    integrate_tlsph         :: Any

    function DummySemidiscretization(; parallelization_backend=SerialBackend())
        new(parallelization_backend, Ref(true))
    end
end

@inline function PointNeighbors.parallel_foreach(f, iterator, semi::DummySemidiscretization)
    PointNeighbors.parallel_foreach(f, iterator, semi.parallelization_backend)
end

@inline function TrixiParticles.get_neighborhood_search(system, neighbor_system,
                                                        ::DummySemidiscretization)
    search_radius = TrixiParticles.compact_support(system, neighbor_system)
    eachpoint = TrixiParticles.eachparticle(neighbor_system)
    return TrixiParticles.TrivialNeighborhoodSearch{ndims(system)}(; search_radius,
                                                                   eachpoint)
end

@inline function TrixiParticles.get_neighborhood_search(system,
                                                        semi::DummySemidiscretization)
    return get_neighborhood_search(system, system, semi)
end

include("count_allocations.jl")
include("rectangular_patch.jl")
