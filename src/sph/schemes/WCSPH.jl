"""
    WCSPH(state_equation)

Weakly-compressible SPH introduced by (Monaghan, 1994). This formulation relies on a stiff
equation of state (see  [`StateEquationCole`](@ref)) that generates large pressure changes
for small density variations.

## References:
- Joseph J. Monaghan. "Simulating Free Surface Flows in SPH".
  In: Journal of Computational Physics 110 (1994), pages 399-406.
  [doi: 10.1006/jcph.1994.1034](https://doi.org/10.1006/jcph.1994.1034)
"""
struct WCSPH{SE}
    state_equation::SE
    function WCSPH(state_equation)
        new{typeof(state_equation)}(state_equation)
    end
end

function Base.show(io::IO, container::FluidParticleContainer{<:WCSPH})
    @nospecialize container # reduce precompilation time

    print(io, "FluidParticleContainer{WCSPH, ", ndims(container), "}(")
    print(io, container.density_calculator)
    print(io, ", ", container.SPH_scheme.state_equation)
    print(io, ", ", container.smoothing_kernel)
    print(io, ", ", container.viscosity)
    print(io, ", ", container.acceleration)
    print(io, ") with ", nparticles(container), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", container::FluidParticleContainer{<:WCSPH})
    @nospecialize container # reduce precompilation time

    if get(io, :compact, false)
        show(io, container)
    else
        summary_header(io, "FluidParticleContainer{WCSPH, $(ndims(container))}")
        summary_line(io, "#particles", nparticles(container))
        summary_line(io, "density calculator",
                     container.density_calculator |> typeof |> nameof)
        summary_line(io, "state equation",
                     container.SPH_scheme.state_equation |> typeof |> nameof)
        summary_line(io, "smoothing kernel", container.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "viscosity", container.viscosity)
        summary_line(io, "acceleration", container.acceleration)
        summary_footer(io)
    end
end

function compute_quantities(container::FluidParticleContainer{<:WCSPH}, container_index,
                            density_calculator, v, u, u_ode, semi, t)
    compute_density!(container, container_index, density_calculator, v, u, u_ode, semi)
    compute_pressure!(container, v)
end

function compute_pressure!(container::FluidParticleContainer{<:WCSPH}, v)
    @unpack SPH_scheme, pressure = container
    @unpack state_equation = SPH_scheme

    # Note that @threaded makes this slower
    for particle in eachparticle(container)
        pressure[particle] = state_equation(get_particle_density(particle, v, container))
    end
end
