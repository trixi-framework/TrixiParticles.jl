using Test
using TrixiParticles
using LinearAlgebra
using Printf
using QuadGK: quadgk
using Random: Random

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

function perturbate!(data, amplitude)
    for i in eachindex(data)
        # Perturbation in the interval (-amplitude, amplitude)
        data[i] .+= 2amplitude * rand() .- amplitude
    end

    return data
end

# Rectangular patch of particles, optionally with a perturbation in position and quantities
function rectangular_patch(particle_spacing, size, density=1000.0, pressure=0.0, seed=1,
                           perturbation_factor=1.0)
    # Fixed seed to ensure reproducibility
    Random.seed!(seed)

    # Center particle at the origin (assuming odd size)
    min_corner = -particle_spacing / 2 .* size
    ic = RectangularShape(particle_spacing, size, min_corner, density,
                                         pressure=pressure)

    perturbate!(ic.mass, perturbation_factor * 0.1 * ic.mass[1])
    perturbate!(ic.density, perturbation_factor * 0.1density)
    perturbate!(ic.pressure, perturbation_factor * 2000)
    perturbate!(ic.velocity, perturbation_factor * 0.5particle_spacing)
    perturbate!(ic.coordinates, perturbation_factor * 0.5particle_spacing)

    # Don't perturbate center particle position
    center_particle = ceil(prod(size))
    ic.coordinates[:, center_particle] .= 0.0

    return ic
end
