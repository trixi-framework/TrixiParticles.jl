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

function perturb!(data, amplitude)
    for i in eachindex(data)
        # Perturbation in the interval (-amplitude, amplitude)
        data[i] += 2amplitude * rand() - amplitude
    end

    return data
end

# Rectangular patch of particles, optionally with a perturbation in position and quantities
function rectangular_patch(particle_spacing, size; density=1000.0, pressure=0.0, seed=1,
                           perturbation_factor=1.0)
    # Fixed seed to ensure reproducibility
    Random.seed!(seed)

    # Center particle at the origin (assuming odd size)
    min_corner = -particle_spacing / 2 .* size
    ic = RectangularShape(particle_spacing, size, min_corner, density, pressure=pressure)

    perturb!(ic.mass, perturbation_factor * 0.1 * ic.mass[1])
    perturb!(ic.density, perturbation_factor * 0.1density)
    perturb!(ic.pressure, perturbation_factor * 2000)
    perturb!(ic.velocity, perturbation_factor * 0.5particle_spacing)
    perturb!(ic.coordinates, perturbation_factor * 0.5particle_spacing)

    # Don't perturb center particle position
    center_particle = ceil(Int, prod(size) / 2)
    ic.coordinates[:, center_particle] .= 0.0

    return ic
end
