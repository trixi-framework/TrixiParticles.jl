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

# Copied from TrixiBase.jl. See https://github.com/trixi-framework/TrixiBase.jl/issues/9.
"""
    @test_nowarn_mod expr

Modified version of `@test_nowarn expr` that prints the content of `stderr` when
it is not empty and ignores some common info statements printed in Trixi.jl
uses.
"""
macro test_nowarn_mod(expr, additional_ignore_content = String[])
    quote
        let fname = tempname()
            try
                ret = open(fname, "w") do f
                    redirect_stderr(f) do
                        $(esc(expr))
                    end
                end
                stderr_content = read(fname, String)
                if !isempty(stderr_content)
                    println("Content of `stderr`:\n", stderr_content)
                end

                # Patterns matching the following ones will be ignored. Additional patterns
                # passed as arguments can also be regular expressions, so we just use the
                # type `Any` for `ignore_content`.
                ignore_content = Any["[ Info: You just called `trixi_include`. Julia may now compile the code, please be patient.\n"]
                append!(ignore_content, $additional_ignore_content)
                for pattern in ignore_content
                    stderr_content = replace(stderr_content, pattern => "")
                end

                # We also ignore simple module redefinitions for convenience. Thus, we
                # check whether every line of `stderr_content` is of the form of a
                # module replacement warning.
                @test occursin(r"^(WARNING: replacing module .+\.\n)*$", stderr_content)
                ret
            finally
                rm(fname, force = true)
            end
        end
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
                           perturbation_factor=1.0, perturbation_factor_position=1.0,
                           set_function=nothing, offset=ntuple(_ -> 0.0, length(size)))
    # Fixed seed to ensure reproducibility
    Random.seed!(seed)

    # Center particle at the origin (assuming odd size)
    min_corner = -particle_spacing / 2 .* size
    ic = RectangularShape(particle_spacing, size, min_corner,
                          density=density, pressure=pressure)

    perturb!(ic.coordinates, perturbation_factor_position * 0.5 * particle_spacing)

    # Don't perturb center particle position
    center_particle = ceil(Int, prod(size) / 2)
    ic.coordinates[:, center_particle] .= 0.0

    for i in 1:Base.size(ic.coordinates, 2)
        ic.coordinates[:, i] .+= offset
    end

    if set_function !== nothing
        for i in 1:Base.size(ic.coordinates, 2)
            coord = ic.coordinates[:, i]
            ic.mass[i], ic.density[i], ic.pressure[i], ic.velocity[:, i] = set_function(coord)
        end
    end

    if perturbation_factor > eps()
        perturb!(ic.mass, perturbation_factor * 0.1 * ic.mass[1])
        perturb!(ic.density, perturbation_factor * 0.1 * density)
        perturb!(ic.pressure, perturbation_factor * 2000)
        perturb!(ic.velocity, perturbation_factor * 0.5 * particle_spacing)
    end

    return ic
end
