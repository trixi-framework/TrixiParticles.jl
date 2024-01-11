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
        data[i] += 2 * amplitude * rand() - amplitude
    end

    return data
end

# Rectangular patch of particles, optionally with a perturbation in position and/or quantities
function rectangular_patch(particle_spacing, size; density=1000.0, pressure=0.0, seed=1,
                           perturbation_factor=1.0, perturbation_factor_position=0.0,
                           set_function=nothing, offset=ntuple(_ -> 0.0, length(size)))
    # Fixed seed to ensure reproducibility
    Random.seed!(seed)

    if perturbation_factor_position < eps()
        perturbation_factor_position = perturbation_factor
    end

    # Center particle at the origin (assuming odd size)
    min_corner = -particle_spacing / 2 .* size
    ic = RectangularShape(particle_spacing, size, min_corner, density, pressure=pressure)

    perturb!(ic.coordinates, perturbation_factor_position * 0.5 * particle_spacing)

    # Don't perturb center particle position
    center_particle = ceil(Int, prod(size) / 2)
    ic.coordinates[:, center_particle] .= 0.0

    for i in 1:Base.size(ic.coordinates, 2)
        ic.coordinates[:, i] .+= offset
    end

    if set_function === nothing
        perturb!(ic.mass, perturbation_factor * 0.1 * ic.mass[1])
        perturb!(ic.density, perturbation_factor * 0.1 * density)
        perturb!(ic.pressure, perturbation_factor * 2000)
        perturb!(ic.velocity, perturbation_factor * 0.5 * particle_spacing)
    else
        for i in 1:Base.size(ic.coordinates, 2)
            coord = ic.coordinates[:, i]
            ic.mass[i], ic.density[i], ic.pressure[i], ic.velocity[:, i] = set_function(coord)
        end
    end

    return ic
end
