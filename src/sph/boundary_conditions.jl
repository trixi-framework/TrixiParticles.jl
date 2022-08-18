# Create Tuple of BCs from single BC
digest_boundary_conditions(boundary_condition) = (boundary_condition, )
digest_boundary_conditions(boundary_condition::Tuple) = boundary_condition
digest_boundary_conditions(::Nothing) = ()


@doc raw"""
    BoundaryConditionMonaghanKajtar(K, coordinates, masses, spacings;
                                    neighborhood_search=nothing)

Boundaries modeled as boundary particles which exert forces on the fluid particles (Monaghan, Kajtar, 2009).
The force on fluid particle ``a`` is given by
```math
f_a = m_a \sum_{b \in B} f_{ab} - m_b \Pi_{ab} \nabla_{r_a} W(\Vert r_a - r_b \Vert, h)
```
with
```math
f_{ab} = \frac{K}{\beta} \frac{r_{ab}}{\Vert r_{ab} \Vert^2} W(\Vert r_{ab} \Vert, h)
\frac{2 m_b}{m_a + m_b},
```
where ``B`` denotes the set of boundary particles, ``m_a`` and ``m_b`` are the masses of
fluid particle ``a`` and boundary particle ``b`` respectively, and
``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``.

The boundary particles are assumed to have uniform spacing by the factor ``\beta`` smaller
than the expected fluid particle spacing.
For example, if the fluid particles have an expected spacing of `0.1` and the boundary particles
have a uniform spacing of `0.05`, then this parameter should be set to ``\beta = 2``.
According to (Monaghan, Kajtar, 2009), a value of ``\beta = 3`` for cubic smoothing_kernels
and ``\beta = 2`` for quintic kernels is reasonable for most computing purposes.

The parameter ``K`` is used to scale the force exerted by the boundary particles.
In (Monaghan, Kajtar, 2009), a value of ``gD`` is used for static tank simulations,
where ``g`` is the gravitational acceleration and ``D`` is the depth of the fluid.

The viscosity ``\Pi_{ab}`` is calculated according to the viscosity used in the
simulation, where the density of the boundary particle if needed is assumed to be
identical to the density of the fluid particle.

References:
- Joseph J. Monaghan, Jules B. Kajtar. "SPH particle boundary forces for arbitrary boundaries".
  In: Computer Physics Communications 180.10 (2009), pages 1811–1820.
  [doi: 10.1016/j.cpc.2009.05.008](https://doi.org/10.1016/j.cpc.2009.05.008)
"""
struct BoundaryConditionMonaghanKajtar{ELTYPE<:Real, NS}
    K                   ::ELTYPE
    coordinates         ::Array{ELTYPE, 2}
    mass                ::Vector{ELTYPE}
    beta                ::Float64
    neighborhood_search ::NS

    function BoundaryConditionMonaghanKajtar(K, coordinates, masses, beta;
                                             neighborhood_search=nothing)
        new{typeof(K), typeof(neighborhood_search)}(K, coordinates, masses,
                                                    beta, neighborhood_search)
    end
end

function initialize!(boundary_conditions::BoundaryConditionMonaghanKajtar, semi)
    @unpack neighborhood_search = boundary_conditions

    initialize!(neighborhood_search, boundary_conditions,
                semi, particles=eachparticle(boundary_conditions))
end

@inline nparticles(boundary_container::BoundaryConditionMonaghanKajtar) = length(boundary_container.mass)
