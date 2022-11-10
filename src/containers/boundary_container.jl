abstract type BoundaryParticleContainer{NDIMS} <: ParticleContainer{NDIMS} end


@inline n_moving_particles(container::BoundaryParticleContainer) = 0


@inline function get_current_coords(particle, u, container::BoundaryParticleContainer)
    @unpack coordinates = container

    return get_particle_coords(particle, coordinates, container)
end


function initialize!(container::BoundaryParticleContainer)
    @unpack coordinates, neighborhood_search = container

    # Initialize neighborhood search
    initialize!(neighborhood_search, coordinates, container)
end

function update!(container::BoundaryParticleContainer, u, u_ode, semi)
    return container
end


function write_variables!(u0, container::BoundaryParticleContainer)
    return u0
end


# Boundary-fluid interaction
function calc_du!(du, u_particle_container, u_neighbor_container,
                  particle_container::BoundaryParticleContainer,
                  neighbor_container::FluidParticleContainer)
    # No interaction (only fluid-boundary interaction)
    return du
end

# Boundary-boundary interaction
function calc_du!(du, u_particle_container, u_neighbor_container,
                  particle_container::BoundaryParticleContainer,
                  neighbor_container::BoundaryParticleContainer)
    # No interaction
    return du
end


@doc raw"""
    BoundaryParticlesMonaghanKajtar(coordinates, masses, K, beta,
                                    boundary_particle_spacing;
                                    neighborhood_search=nothing)

Boundaries modeled as boundary particles which exert forces on the fluid particles (Monaghan, Kajtar, 2009).
The force on fluid particle ``a`` is given by
```math
f_a = m_a \sum_{b \in B} f_{ab} - m_b \Pi_{ab} \nabla_{r_a} W(\Vert r_a - r_b \Vert, h)
```
with
```math
f_{ab} = \frac{K}{\beta} \frac{r_{ab}}{\Vert r_{ab} \Vert (\Vert r_{ab} \Vert - d)} \Phi(\Vert r_{ab} \Vert, h)
\frac{2 m_b}{m_a + m_b},
```
where ``B`` denotes the set of boundary particles, ``m_a`` and ``m_b`` are the masses of
fluid particle ``a`` and boundary particle ``b`` respectively,
``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``,
and ``d`` denotes the boundary particle spacing
(see (Monaghan, Kajtar, 2009, Equation (3.1)) and (Valizadeh, Monaghan, 2015)).
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

References:
- Joseph J. Monaghan, Jules B. Kajtar. "SPH particle boundary forces for arbitrary boundaries".
  In: Computer Physics Communications 180.10 (2009), pages 1811–1820.
  [doi: 10.1016/j.cpc.2009.05.008](https://doi.org/10.1016/j.cpc.2009.05.008)
- Alireza Valizadeh, Joseph J. Monaghan. "A study of solid wall models for weakly compressible SPH."
  In: Journal of Computational Physics 300 (2015), pages 5–19.
  [doi: 10.1016/J.JCP.2015.07.033](https://doi.org/10.1016/J.JCP.2015.07.033)
"""
struct BoundaryParticlesMonaghanKajtar{NDIMS, ELTYPE<:Real, NS} <: BoundaryParticleContainer{NDIMS}
    coordinates                 ::Array{ELTYPE, 2}
    mass                        ::Vector{ELTYPE}
    K                           ::ELTYPE
    beta                        ::ELTYPE
    boundary_particle_spacing   ::ELTYPE
    neighborhood_search         ::NS

    function BoundaryParticlesMonaghanKajtar(coordinates, masses, K, beta,
                                             boundary_particle_spacing;
                                             neighborhood_search=nothing)
        NDIMS = size(coordinates, 1)

        new{NDIMS, typeof(K), typeof(neighborhood_search)}(coordinates, masses, K, beta,
                                                           boundary_particle_spacing,
                                                           neighborhood_search)
    end
end


@inline function boundary_kernel(r, h)
    q = r / h

    if q >= 2
        return 0.0
    end

    # (Monaghan, Kajtar, 2009, Section 4): The kernel should be normalized to 1.77 for q=0
    return 1.77/32 * (1 + 5/2 * q + 2 * q^2) * (2 - q)^5
end


@inline function boundary_particle_impact(particle, particle_container,
                                          boundary_container::BoundaryParticlesMonaghanKajtar,
                                          pos_diff, distance, m_a, m_b, density_a, v_a)
    @unpack state_equation, viscosity, smoothing_kernel, smoothing_length = particle_container
    @unpack K, beta, boundary_particle_spacing = boundary_container

    pi_ab = viscosity(state_equation.sound_speed, v_a, pos_diff, distance, density_a, smoothing_length)

    dv_viscosity = m_b * pi_ab * kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

    dv_repulsive = K / beta * pos_diff / (distance * (distance - boundary_particle_spacing)) *
        boundary_kernel(distance, smoothing_length) * 2 * m_b / (m_a + m_b)

    return dv_viscosity + dv_repulsive
end


struct BoundaryParticlesFrozen{NDIMS, ELTYPE<:Real, NS} <: BoundaryParticleContainer{NDIMS}
    coordinates                 ::Array{ELTYPE, 2}
    mass                        ::Vector{ELTYPE}
    rest_density                ::ELTYPE
    neighborhood_search         ::NS

    function BoundaryParticlesFrozen(coordinates, masses, rest_density;
                                     neighborhood_search=nothing)
        NDIMS = size(coordinates, 1)

        new{NDIMS, eltype(coordinates), typeof(neighborhood_search)}(coordinates, masses, rest_density,
                                                                     neighborhood_search)
    end
end


@inline function boundary_particle_impact(particle, particle_container,
                                          boundary_container::BoundaryParticlesFrozen,
                                          pos_diff, distance, m_a, m_b, density_a, v_a)
    @unpack pressure, state_equation, viscosity, smoothing_kernel, smoothing_length = particle_container
    @unpack rest_density = boundary_container

    pi_ab = viscosity(state_equation.sound_speed, v_a, pos_diff, distance, density_a, smoothing_length)

    grad_kernel = kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

    # Use 0 as boundary particle pressure
    dv_pressure = -m_b * (pressure[particle] / density_a^2 +
                          0 / rest_density^2) * grad_kernel

    dv_viscosity = m_b * pi_ab * grad_kernel

    return dv_pressure + dv_viscosity
end
