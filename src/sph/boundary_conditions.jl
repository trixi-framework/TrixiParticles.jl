# Create Tuple of BCs from single BC
digest_boundary_conditions(boundary_condition) = (boundary_condition,)
digest_boundary_conditions(boundary_condition::Tuple) = boundary_condition
digest_boundary_conditions(::Nothing) = ()


@doc raw"""
    BoundaryConditionMonaghanKajtar(coordinates, masses, K, beta,
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
struct BoundaryConditionMonaghanKajtar{ELTYPE<:Real,NS}
    coordinates::Array{ELTYPE,2}
    mass::Vector{ELTYPE}
    K::ELTYPE
    beta::ELTYPE
    boundary_particle_spacing::ELTYPE
    neighborhood_search::NS

    function BoundaryConditionMonaghanKajtar(coordinates, masses, K, beta,
                                                boundary_particle_spacing;
                                                neighborhood_search=nothing)
        new{typeof(K),typeof(neighborhood_search)}(coordinates, masses, K, beta,
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
    return 1.77 / 32 * (1 + 5 / 2 * q + 2 * q^2) * (2 - q)^5
end

function initialize!(boundary_conditions, semi)
    @unpack neighborhood_search = boundary_conditions

    initialize!(neighborhood_search, boundary_conditions,
        semi, particles=eachparticle(boundary_conditions))
end

@inline nparticles(boundary_container::BoundaryConditionMonaghanKajtar) = length(boundary_container.mass)

@inline function boundary_impact_normal(boundary_condition::BoundaryConditionMonaghanKajtar,
                                        semi, distance, pos_diff, particle, m_a, m_b)
    @unpack smoothing_length = semi
    @unpack coordinates, mass, K, beta, boundary_particle_spacing = boundary_condition

    return K / beta * pos_diff / (distance * (distance - boundary_particle_spacing)) *
           boundary_kernel(distance, smoothing_length) * 2 * m_b / (m_a + m_b)

end

@inline function boundary_impact_tangential(boundary_condition,
                                            semi::WCSPHSemidiscretization, u, particle, pos_diff, distance, m_b)
    @unpack smoothing_kernel, smoothing_length, viscosity, cache, density_calculator = semi
    @unpack boundary_particle_spacing = boundary_condition
    particle_density = get_particle_density(u, cache, density_calculator, particle)
    v_rel = get_particle_vel(u, semi, particle)
    pi_ab = viscosity(v_rel, pos_diff, distance, particle_density, particle_density, smoothing_length)


    return - m_b * pi_ab * kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

end


@doc raw"""
    BoundaryConditionCrespo(coordinates, masses, c;
                            neighborhood_search=nothing)
The boundary particles verify the same equations of continuity and of state as the fluid particles,
but their position remains unchanged (Crespo, 2007). 
Assuming the same mass ``m`` for boundary and fluid particle and a constant speed of sound ``c`` the force on a fluid particle ``a`` is given by
```math
f_{ab} = -\sum_{b \in B} \left( 2 c^2 \frac{W_{ab}}{(W_{ab}+W_0)^2} + m \Pi_{ab} \right) \nabla W_{ab}
```            
where ``B`` denotes the set of boundary particles and the subindices ``a`` and ``b`` the
fluid particle and boundary particle respectively.
``W_{ab} = W(r_a-r_b, h)`` denotes a weight or kernel function. Being ``W_0 = W_{aa} = W_{bb} = W(r_{ab}=0)`` with ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``.
The viscosity ``\Pi_{ab}`` is calculated according to the viscosity used in the
simulation, where the density of the boundary particle if needed is assumed to be
identical to the density of the fluid particle.

According to (Crespo, 2007) a boundary particle spacing of ``h/1.3`` is reasonable. There is no need to scale the repulsive force by any factor.
Nevertheless, it is recommended to place the boundary particles in two rows forming a staggered grid. Otherwise, one needs to set a small particle spacing 
which may cause unexpected large repulsive forces. 

References:
- A. J. C. Crespo, M. Gòmez-Gesteira and R.A. Dalrymple. "Boundary Conditions Generated by Dynamic Particles in SPH Methods".
  In: Computers, Materials and Continua vol. 5 (2007), pages 173-184.
  [doi: 10.3970/cmc.2007.005.173](https://doi.org/10.3970/cmc.2007.005.173)
"""
struct BoundaryConditionCrespo{ELTYPE<:Real,NS}
    coordinates::Array{ELTYPE,2}
    mass::Vector{ELTYPE}
    c::ELTYPE
    neighborhood_search::NS

    function BoundaryConditionCrespo(coordinates, masses, c; neighborhood_search=nothing)
        new{typeof(c),typeof(neighborhood_search)}(coordinates, masses, c, neighborhood_search)
    end
end

@inline nparticles(boundary_container::BoundaryConditionCrespo) = length(boundary_container.mass)

@inline function boundary_impact_normal(boundary_condition::BoundaryConditionCrespo,
    semi, distance, pos_diff, particle, m_a, m_b)
    @unpack smoothing_kernel, smoothing_length = semi
    @unpack coordinates, mass, c = boundary_condition

    W_ab = kernel(smoothing_kernel, distance, smoothing_length)
    W_0 = kernel(smoothing_kernel, 0.0, smoothing_length)

    return -2 * c^2 * W_ab * kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / (distance * (W_ab + W_0)^2)
end




struct BoundaryConditionFixedParticleLiu{ELTYPE<:Real,NS}
    coordinates::Array{ELTYPE,2}
    mass::Vector{ELTYPE}
    rho_0              ::ELTYPE
    neighborhood_search::NS

    function BoundaryConditionFixedParticleLiu(coordinates, masses, rho_0; neighborhood_search=nothing)
        new{typeof(rho_0),typeof(neighborhood_search)}(coordinates, masses, rho_0, neighborhood_search)
    end

end
@inline nparticles(boundary_container::BoundaryConditionFixedParticleLiu) = length(boundary_container.mass)

@inline function boundary_impact_normal(boundary_condition::BoundaryConditionFixedParticleLiu,
    semi, distance, pos_diff, particle, m_a, m_b)
    @unpack smoothing_kernel, smoothing_length, cache, gravity = semi
    @unpack rho_0 = boundary_condition
    @unpack pressure = cache

    # TBD: normal vector is set to (0,1) to test the implementation
    n_B = SVector(0,1)
    n_dot_g = sum(n_B .* gravity)
    P_B = pressure[particle] - rho_0*(distance+0.5*smoothing_length)*n_dot_g


    return m_a*(pressure[particle]+P_B)*n_B*dimensionless_contribution_normal(distance, smoothing_length)/(rho_0^2*smoothing_length^4)
end

@inline function boundary_impact_tangential(boundary_condition::BoundaryConditionFixedParticleLiu,
                                            semi, u, particle, pos_diff, distance, m_b)
    @unpack smoothing_kernel, smoothing_length, viscosity, cache, density_calculator = semi
    @unpack rho_0 = boundary_condition
    @unpack nu = viscosity

    v_rel = get_particle_vel(u, semi, particle)
    if ndims(semi) == 3
        projection_vec = SVector(1,1,0)
    else
        projection_vec = SVector(1,0)
    end
    u_tang = v_rel.*projection_vec

    return -m_b*2*nu*u_tang*dimensionless_contribution_tangential(distance, smoothing_length)/(rho_0^2*smoothing_length^5)
end


@inline function dimensionless_contribution_normal(r, h)
    q = r / h

    if q >= 1.45
        return 0.0
    end

    return  -0.686*q^4 + 2.93*q^3 -3.49*q^2 -0.067*q + 1.55
end

@inline function dimensionless_contribution_tangential(r, h)
    q = r / h

    if q >= 1.43
        return 0.0
    end

    return  -0.645*q^4 + 1.597*q^3 +0.703*q^2 -4.43*q + 2.93
end


# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function calc_boundary_condition_per_particle!(du, u, particle,
    boundary_condition,
    semi)
    @unpack smoothing_kernel, smoothing_length,
    density_calculator, cache = semi
    @unpack coordinates, neighborhood_search, mass = boundary_condition

    m_a = mass[particle]
    for boundary_particle in eachneighbor(particle, u, neighborhood_search, semi, particles=eachparticle(boundary_condition))
        pos_diff = get_particle_coords(u, semi, particle) -
                   get_particle_coords(boundary_condition, semi, boundary_particle)
        distance = norm(pos_diff)

        if eps() < distance <= compact_support(smoothing_kernel, smoothing_length)
            # Viscosity
            m_b = mass[boundary_particle]
            f_n = boundary_impact_normal(boundary_condition, semi, distance, pos_diff, particle, m_a, m_b)
            f_t = boundary_impact_tangential(boundary_condition, semi, u, particle, pos_diff, distance, m_b)

            dv = f_n + f_t

            for i in 1:ndims(semi)
                du[ndims(semi)+i, particle] += dv[i]
            end
        end
    end
end
