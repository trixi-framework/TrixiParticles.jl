@doc raw"""
    BoundaryModelMonaghanKajtar(K, beta, boundary_particle_spacing, mass;
                                viscosity=nothing)

`boundary_model` for `BoundarySPHSystem`.

# Arguments
- `K`: Scaling factor for repulsive force.
- `beta`: Ratio of fluid particle spacing to boundary particle spacing.
- `boundary_particle_spacing`: Boundary particle spacing.
- `mass`: Vector holding the mass of each boundary particle.

# Keywords
- `viscosity`:  Free-slip (default) or no-slip condition. See description above for further
                information.
"""
struct BoundaryModelMonaghanKajtar{ELTYPE <: Real, V}
    K                         :: ELTYPE
    beta                      :: ELTYPE
    boundary_particle_spacing :: ELTYPE
    hydrodynamic_mass         :: Vector{ELTYPE}
    viscosity                 :: V

    function BoundaryModelMonaghanKajtar(K, beta, boundary_particle_spacing, mass;
                                         viscosity=nothing)
        return new{typeof(K), typeof(viscosity)}(K, beta, boundary_particle_spacing, mass,
                                                 viscosity)
    end
end

function Base.show(io::IO, model::BoundaryModelMonaghanKajtar)
    @nospecialize model # reduce precompilation time

    print(io, "BoundaryModelMonaghanKajtar(")
    print(io, model.K)
    print(io, ", ")
    print(io, model.beta)
    print(io, ", ")
    print(io, model.viscosity |> typeof |> nameof)
    print(io, ")")
end

@inline function pressure_acceleration(particle_system,
                                       neighbor_system::Union{BoundarySPHSystem{<:BoundaryModelMonaghanKajtar},
                                                              TotalLagrangianSPHSystem{<:BoundaryModelMonaghanKajtar}},
                                       neighbor, m_a, m_b, p_a, p_b, rho_a, rho_b,
                                       pos_diff, distance, grad_kernel, correction)
    (; K, beta, boundary_particle_spacing) = neighbor_system.boundary_model

    # This is `distance - boundary_particle_spacing` in the paper. This factor makes
    # the force grow infinitely close to the boundary, with a singularity where
    # `distance` = `boundary_particle_spacing`. However, when the time step is large
    # enough for a particle to end up behind the singularity in a time integration stage,
    # the force will switch sign and become smaller again.
    #
    # In order to avoid this, we clip the force at a "large" value, large enough to prevent
    # penetration when a reasonable `K` is used, but small enough to not cause instabilites
    # or super small time steps.
    distance_from_singularity = max(0.01 * boundary_particle_spacing,
                                    distance - boundary_particle_spacing)

    return K / beta^(ndims(particle_system) - 1) * pos_diff /
           (distance * distance_from_singularity) *
           boundary_kernel(distance, particle_system.smoothing_length)
end

@fastpow @inline function boundary_kernel(r, h)
    q = r / h

    # TODO The neighborhood search fluid->boundary should use this search distance
    if q >= 2
        return 0.0
    end

    # (Monaghan, Kajtar, 2009, Section 4): The kernel should be normalized to 1.77 for q=0
    return 1.77 / 32 * (1 + 5 / 2 * q + 2 * q^2) * (2 - q)^5
end

@inline function particle_density(v, model::BoundaryModelMonaghanKajtar, system, particle)
    (; hydrodynamic_mass, boundary_particle_spacing) = model

    # This model does not use any particle density. However, a mean density is used for
    # `ArtificialViscosityMonaghan` in the fluid interaction.
    return hydrodynamic_mass[particle] / boundary_particle_spacing^ndims(system)
end

# This model does not not use any particle pressure
particle_pressure(v, model::BoundaryModelMonaghanKajtar, system, particle) = zero(eltype(v))

@inline function update_pressure!(boundary_model::BoundaryModelMonaghanKajtar, system,
                                  v, u, v_ode, u_ode, semi)
    # Nothing to do in the update step
    return boundary_model
end

@inline function update_density!(boundary_model::BoundaryModelMonaghanKajtar, system,
                                 v, u, v_ode, u_ode, semi)
    # Nothing to do in the update step
    return boundary_model
end
