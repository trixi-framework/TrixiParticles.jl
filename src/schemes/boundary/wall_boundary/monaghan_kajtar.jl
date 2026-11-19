@doc raw"""
    BoundaryModelMonaghanKajtar(K, beta, boundary_particle_spacing, mass;
                                viscosity=nothing)

Boundary model for [`WallBoundarySystem`](@ref).

# Arguments
- `K`: Scaling factor for repulsive force.
- `beta`: Ratio of fluid particle spacing to boundary particle spacing.
- `boundary_particle_spacing`: Boundary particle spacing.
- `mass`: Vector holding the mass of each boundary particle.

# Keywords
- `viscosity`:  Free-slip (default) or no-slip condition. See description above for further
                information.
"""
struct BoundaryModelMonaghanKajtar{ELTYPE <: Real, VECTOR, V}
    K                         :: ELTYPE
    beta                      :: ELTYPE
    boundary_particle_spacing :: ELTYPE
    hydrodynamic_mass         :: VECTOR # Vector{ELTYPE}
    viscosity                 :: V
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function BoundaryModelMonaghanKajtar(K, beta, boundary_particle_spacing, mass;
                                     viscosity=nothing)
    return BoundaryModelMonaghanKajtar(K, convert(typeof(K), beta),
                                       boundary_particle_spacing,
                                       mass, viscosity)
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

@inline function density_calculator(model::BoundaryModelMonaghanKajtar)
    return nothing
end

# The radial boundary law replaces the physical pressure force for every fluid
# correction, including corrections with asymmetric gradients. TVF calls the
# inter-particle operator separately and does not apply this law.
@inline function pressure_acceleration(particle_system,
                                       neighbor_system::Union{WallBoundarySystem{<:BoundaryModelMonaghanKajtar},
                                                              TotalLagrangianSPHSystem{<:BoundaryModelMonaghanKajtar}},
                                       particle, neighbor, m_a, m_b, p_a, p_b, rho_a, rho_b,
                                       pos_diff, distance, grad_kernel, correction)
    (; K, beta, boundary_particle_spacing) = neighbor_system.boundary_model
    h = smoothing_length(particle_system, particle)

    # A coincident pair has no radial direction. A corrected fluid gradient can
    # still be nonzero there, so only skip the repulsive acceleration.
    distance < almostzero(h) && return zero(pos_diff)

    # This is `distance - boundary_particle_spacing` in the paper. This factor makes
    # the force grow infinitely close to the boundary, with a singularity where
    # `distance` = `boundary_particle_spacing`. However, when the time step is large
    # enough for a particle to end up behind the singularity in a time integration stage,
    # the force will switch sign and become smaller again.
    #
    # In order to avoid this, we clip the force at a "large" value, large enough to prevent
    # penetration when a reasonable `K` is used, but small enough to not cause instabilites
    # or super small time steps.
    distance_from_singularity = max(boundary_particle_spacing / 100,
                                    distance - boundary_particle_spacing)

    return K / beta^(ndims(particle_system) - 1) * pos_diff /
           (distance * distance_from_singularity) *
           boundary_kernel(distance, h)
end

@inline function neighbor_pressure_kernel_grad(neighbor_system::Union{WallBoundarySystem{<:BoundaryModelMonaghanKajtar},
                                                                      TotalLagrangianSPHSystem{<:BoundaryModelMonaghanKajtar}},
                                               pos_diff, distance, neighbor, W_a)
    # Repulsive particles have no hydrodynamic kernel of their own. TVF uses the
    # reversed fluid gradient, not an elastic structure kernel or the repulsion law.
    return -W_a
end

@inline function tvf_shifting_quadrature_factors(system::Union{WallBoundarySystem{<:BoundaryModelMonaghanKajtar},
                                                               TotalLagrangianSPHSystem{<:BoundaryModelMonaghanKajtar}})
    # Let d be the wall marker spacing and dx_fluid = beta*d the nominal fluid spacing.
    # At matched spacing (beta=1), use the marker's hydrodynamic mass m_b and weight 1.
    # When refining the wall, keep the volume used in the pair operator at dx_fluid^D.
    # Since rho_b = m_b/d^D, the required mass is rho_b * dx_fluid^D = m_b * beta^D.
    # Weight the result by the marker's share of a wall patch of size dx_fluid^(D-1).
    # Halving d in 3D gives four times as many markers on that patch, each contributing
    # one quarter of its former weight at its own kernel gradient.
    beta = system.boundary_model.beta
    return beta^ndims(system), inv(beta^(ndims(system) - 1))
end

@inline contributes_to_pressure_average(::Union{WallBoundarySystem{<:BoundaryModelMonaghanKajtar},
                                                TotalLagrangianSPHSystem{<:BoundaryModelMonaghanKajtar}}) = false

@fastpow @inline function boundary_kernel(r, h)
    q = r / h

    # TODO The neighborhood search fluid->boundary should use this search distance
    if q >= 2
        return zero(eltype(r))
    end

    # (Monaghan, Kajtar, 2009, Section 4): The kernel should be normalized to 1.77 for q=0
    return (177 // 100) // 32 * (1 + 5 // 2 * q + 2 * q^2) * (2 - q)^5
end

@inline function current_density(v,
                                 system::Union{WallBoundarySystem{<:BoundaryModelMonaghanKajtar},
                                               TotalLagrangianSPHSystem{<:BoundaryModelMonaghanKajtar}},
                                 particle)
    (; hydrodynamic_mass, boundary_particle_spacing) = system.boundary_model

    # The radial repulsion law does not use particle density. The fluid interaction
    # still needs a mean density for `ArtificialViscosityMonaghan` and TVF pair weights.
    return hydrodynamic_mass[particle] / boundary_particle_spacing^ndims(system)
end

@inline function current_density(v, model::BoundaryModelMonaghanKajtar, system)
    # We cannot make this an array without allocating
    error("`current_density` not implemented for `BoundaryModelMonaghanKajtar`")
end

# The radial repulsion law does not use particle pressure. This placeholder is not
# a prescribed boundary pressure for EDAC diffusion.
@inline function current_pressure(v,
                                  system::Union{WallBoundarySystem{<:BoundaryModelMonaghanKajtar},
                                                TotalLagrangianSPHSystem{<:BoundaryModelMonaghanKajtar}},
                                  particle)
    return zero(eltype(v))
end

@inline function current_pressure(v, model::BoundaryModelMonaghanKajtar, system)
    # We cannot make this an array without allocating
    error("`current_pressure` not implemented for `BoundaryModelMonaghanKajtar`")
end

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
