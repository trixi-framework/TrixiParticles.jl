@doc raw"""
    BoundaryModelMonaghanKajtar(K, beta, boundary_particle_spacing, mass;
                                viscosity=nothing)

Boundaries modeled as boundary particles which exert forces on the fluid particles (Monaghan, Kajtar, 2009).
The force on fluid particle ``a`` due to boundary particle ``b`` is given by
```math
f_{ab} = m_a \left(\tilde{f}_{ab} - m_b \Pi_{ab} \nabla_{r_a} W(\Vert r_a - r_b \Vert, h)\right)
```
with
```math
\tilde{f}_{ab} = \frac{K}{\beta^{n-1}} \frac{r_{ab}}{\Vert r_{ab} \Vert (\Vert r_{ab} \Vert - d)} \Phi(\Vert r_{ab} \Vert, h)
\frac{2 m_b}{m_a + m_b},
```
where ``m_a`` and ``m_b`` are the masses of fluid particle ``a`` and boundary particle ``b``
respectively, ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles
``a`` and ``b``, ``d`` denotes the boundary particle spacing and ``n`` denotes the number of
dimensions (see (Monaghan, Kajtar, 2009, Equation (3.1)) and (Valizadeh, Monaghan, 2015)).
Note that the repulsive acceleration $\tilde{f}_{ab}$ does not depend on the masses of
the boundary particles.
Here, ``\Phi`` denotes the 1D Wendland C4 kernel, normalized to ``1.77`` for ``q=0``
(Monaghan, Kajtar, 2009, Section 4), with ``\Phi(r, h) = w(r/h)`` and
```math
w(q) =
\begin{cases}
  (1.77/32) (1 + (5/2)q + 2q^2)(2 - q)^5  & \text{if } 0 \leq q < 2 \\
  0                                       & \text{if } q \geq 2.
\end{cases}
```

The boundary particles are assumed to have uniform spacing by the factor ``\beta`` smaller
than the expected fluid particle spacing.
For example, if the fluid particles have an expected spacing of ``0.3`` and the boundary particles
have a uniform spacing of ``0.1``, then this parameter should be set to ``\beta = 3``.
According to (Monaghan, Kajtar, 2009), a value of ``\beta = 3`` for the Wendland C4 that
we use here is reasonable for most computing purposes.

The parameter ``K`` is used to scale the force exerted by the boundary particles.
In (Monaghan, Kajtar, 2009), a value of ``gD`` is used for static tank simulations,
where ``g`` is the gravitational acceleration and ``D`` is the depth of the fluid.

The viscosity ``\Pi_{ab}`` is calculated according to the viscosity used in the
simulation, where the density of the boundary particle if needed is assumed to be
identical to the density of the fluid particle.

# No-slip condition

By choosing the viscosity model [`ArtificialViscosityMonaghan`](@ref) for `viscosity`,
a no-slip condition is imposed. When omitting the viscous interaction
(default `viscosity=nothing`), a free-slip wall boundary condition is applied.

# Arguments
- `K`: Scaling factor for repulsive force.
- `beta`: Ratio of fluid particle spacing to boundary particle spacing.
- `boundary_particle_spacing`: Boundary particle spacing.
- `mass`: Vector holding the mass of each boundary particle.

# Keywords
- `viscosity`:  Free-slip (default) or no-slip condition. See description above for further
                information.

!!! warning
    The no-slip conditions for `BoundaryModelMonaghanKajtar` have not been verified yet.

## References:
- Joseph J. Monaghan, Jules B. Kajtar. "SPH particle boundary forces for arbitrary boundaries".
  In: Computer Physics Communications 180.10 (2009), pages 1811–1820.
  [doi: 10.1016/j.cpc.2009.05.008](https://doi.org/10.1016/j.cpc.2009.05.008)
- Alireza Valizadeh, Joseph J. Monaghan. "A study of solid wall models for weakly compressible SPH."
  In: Journal of Computational Physics 300 (2015), pages 5–19.
  [doi: 10.1016/J.JCP.2015.07.033](https://doi.org/10.1016/J.JCP.2015.07.033)
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
