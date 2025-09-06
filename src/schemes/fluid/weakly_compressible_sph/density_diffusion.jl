@doc raw"""
    DensityDiffusion

An abstract supertype of all density diffusion formulations.

Currently, the following formulations are available:

| Formulation                                 | Suitable for Steady-State Simulations | Low Computational Cost |
| :------------------------------------------ | :------------------------------------ | :--------------------- |
| [`DensityDiffusionMolteniColagrossi`](@ref) | ❌                                    | ✅                     |
| [`DensityDiffusionFerrari`](@ref)           | ❌                                    | ✅                     |
| [`DensityDiffusionAntuono`](@ref)           | ✅                                    | ❌                     |

See [Density Diffusion](@ref density_diffusion) for a comparison and more details.
"""
abstract type DensityDiffusion end

# Most density diffusion formulations don't need updating
function update!(density_diffusion, v, u, system, v_ode, u_ode, semi)
    return density_diffusion
end

@doc raw"""
    DensityDiffusionMolteniColagrossi(; delta)

The commonly used density diffusion term by [Molteni (2009)](@cite Molteni2009).

The term ``\psi_{ab}`` in the continuity equation in [`DensityDiffusion`](@ref) is defined
by
```math
\psi_{ab} = 2(\rho_a - \rho_b) \frac{r_{ab}}{\Vert r_{ab} \Vert^2},
```
where ``\rho_a`` and ``\rho_b`` denote the densities of particles ``a`` and ``b`` respectively
and ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``.

See [`DensityDiffusion`](@ref) for an overview and comparison of implemented density
diffusion terms.
"""
struct DensityDiffusionMolteniColagrossi{ELTYPE} <: DensityDiffusion
    delta::ELTYPE

    function DensityDiffusionMolteniColagrossi(; delta)
        new{typeof(delta)}(delta)
    end
end

@inline function density_diffusion_psi(::DensityDiffusionMolteniColagrossi,
                                       system, neighbor_system,
                                       particle, neighbor, pos_diff, distance, rho_a, rho_b)
    return 2 * (rho_a - rho_b) * pos_diff / distance^2
end

@doc raw"""
    DensityDiffusionFerrari()

A density diffusion term by [Ferrari (2009)](@cite Ferrari2009).

The term ``\psi_{ab}`` in the continuity equation in [`DensityDiffusion`](@ref) is defined
by
```math
\psi_{ab} = \frac{\rho_a - \rho_b}{h_a + h_b} \frac{r_{ab}}{\Vert r_{ab} \Vert},
```
where ``\rho_a`` and ``\rho_b`` denote the densities of particles ``a`` and ``b`` respectively,
``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b`` and
``h_a`` and ``h_b`` are the smoothing lengths of particles ``a`` and ``b`` respectively.

See [`DensityDiffusion`](@ref) for an overview and comparison of implemented density
diffusion terms.
"""
struct DensityDiffusionFerrari <: DensityDiffusion
    delta::Int

    # δ is always 1 in this formulation
    DensityDiffusionFerrari() = new(1)
end

@inline function density_diffusion_psi(::DensityDiffusionFerrari,
                                       system, neighbor_system,
                                       particle, neighbor, pos_diff, distance, rho_a, rho_b)
    return ((rho_a - rho_b) /
            (smoothing_length(system, particle) + smoothing_length(system, neighbor))) *
           pos_diff / distance
end

@doc raw"""
    DensityDiffusionAntuono(initial_condition; delta)

The commonly used density diffusion terms by [Antuono (2010)](@cite Antuono2010), also referred to as
δ-SPH. The density diffusion term by [Molteni (2009)](@cite Molteni2009) is extended by a second
term, which is nicely written down by [Antuono (2012)](@cite Antuono2012).

The term ``\psi_{ab}`` in the continuity equation in [`DensityDiffusion`](@ref) is defined
by
```math
\psi_{ab} = 2\left(\rho_a - \rho_b - \frac{1}{2}\big(\nabla\rho^L_a + \nabla\rho^L_b\big) \cdot r_{ab}\right)
    \frac{r_{ab}}{\Vert r_{ab} \Vert^2},
```
where ``\rho_a`` and ``\rho_b`` denote the densities of particles ``a`` and ``b`` respectively
and ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``.
The symbol ``\nabla\rho^L_a`` denotes the renormalized density gradient defined as
```math
\nabla\rho^L_a = -\sum_b (\rho_a - \rho_b) V_b L_a \nabla W_{ab}
```
with
```math
L_a := \left( -\sum_{b} V_b r_{ab} \otimes \nabla W_{ab} \right)^{-1} \in \R^{d \times d},
```
where ``d`` is the number of dimensions.

See [`DensityDiffusion`](@ref) for an overview and comparison of implemented density
diffusion terms.
"""
struct DensityDiffusionAntuono{NDIMS, ELTYPE, ARRAY2D, ARRAY3D} <: DensityDiffusion
    delta                       :: ELTYPE
    correction_matrix           :: ARRAY3D # Array{ELTYPE, 3}: [i, j, particle]
    normalized_density_gradient :: ARRAY2D # Array{ELTYPE, 2}: [i, particle]

    function DensityDiffusionAntuono(delta, correction_matrix, normalized_density_gradient)
        new{size(correction_matrix, 1), typeof(delta),
            typeof(normalized_density_gradient),
            typeof(correction_matrix)}(delta, correction_matrix,
                                       normalized_density_gradient)
    end
end

function DensityDiffusionAntuono(initial_condition; delta)
    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)
    correction_matrix = Array{ELTYPE, 3}(undef, NDIMS, NDIMS,
                                         nparticles(initial_condition))

    normalized_density_gradient = Array{ELTYPE, 2}(undef, NDIMS,
                                                   nparticles(initial_condition))

    return DensityDiffusionAntuono(delta, correction_matrix, normalized_density_gradient)
end

@inline Base.ndims(::DensityDiffusionAntuono{NDIMS}) where {NDIMS} = NDIMS

function Base.show(io::IO, density_diffusion::DensityDiffusionAntuono)
    @nospecialize density_diffusion # reduce precompilation time

    print(io, "DensityDiffusionAntuono(")
    print(io, density_diffusion.delta)
    print(io, ")")
end

function allocate_buffer(initial_condition, density_diffusion, buffer)
    return allocate_buffer(initial_condition, buffer), density_diffusion
end

function allocate_buffer(ic, dd::DensityDiffusionAntuono, buffer::SystemBuffer)
    initial_condition = allocate_buffer(ic, buffer)
    return initial_condition, DensityDiffusionAntuono(initial_condition; delta=dd.delta)
end

@inline function density_diffusion_psi(density_diffusion::DensityDiffusionAntuono,
                                       system, neighbor_system,
                                       particle, neighbor, pos_diff, distance, rho_a, rho_b)
    normalized_gradient_a = normalized_density_gradient(system, particle)
    normalized_gradient_b = normalized_density_gradient(neighbor_system, neighbor)

    # First term by Molteni & Colagrossi
    result = 2 * (rho_a - rho_b)

    # Second correction term
    result -= dot(normalized_gradient_a + normalized_gradient_b, pos_diff)

    return result * pos_diff / distance^2
end

@propagate_inbounds function normalized_density_gradient(system::FluidSystem,
                                                         particle)
    (; normalized_density_gradient) = system.density_diffusion

    return extract_svector(normalized_density_gradient, system, particle)
end

# TODO clean dispatch
@propagate_inbounds function normalized_density_gradient(system::BoundarySystem,
                                                         particle)
    (; normalized_density_gradient) = system.boundary_model.density_calculator.density_diffusion

    return extract_svector(normalized_density_gradient, system, particle)
end

function update!(density_diffusion::DensityDiffusionAntuono, v, u, system, v_ode, u_ode, semi)
    (; normalized_density_gradient) = density_diffusion

    # Compute correction matrix
    # density_fun = @inline(particle->current_density(v, system, particle))
    system_coords = current_coordinates(u, system)

    # compute_gradient_correction_matrix!(density_diffusion.correction_matrix,
    #                                     system, system_coords, density_fun, semi)
    compute_gradient_correction_matrix!(density_diffusion.correction_matrix, system,
                                             system_coords, v_ode, u_ode, semi,
                                             nothing, smoothing_kernel)

    # Compute normalized density gradient
    set_zero!(normalized_density_gradient)

    foreach_system(semi) do neighbor_system
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                            points=each_moving_particle(system)) do particle, neighbor,
                                                                    pos_diff, distance
            # Only consider particles with a distance > 0
            distance < sqrt(eps(typeof(distance))) && return

            rho_a = current_density(v, system, particle)
            rho_b = current_density(v_neighbor, neighbor_system, neighbor)

            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
            L = correction_matrix(density_diffusion, particle)

            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            volume_b = m_b / rho_b

            normalized_gradient = -(rho_a - rho_b) * L * grad_kernel * volume_b

            for i in eachindex(normalized_gradient)
                normalized_density_gradient[i, particle] += normalized_gradient[i]
            end
        end
    end

    return density_diffusion
end

@propagate_inbounds function density_diffusion!(dv, density_diffusion::DensityDiffusion,
                                                system, neighbor_system, particle, neighbor,
                                                pos_diff, distance, m_b, rho_a, rho_b,
                                                grad_kernel)
    # Density diffusion terms are all zero for distance zero
    distance < sqrt(eps(typeof(distance))) && return

    (; delta) = density_diffusion

    volume_b = m_b / rho_b

    psi = density_diffusion_psi(density_diffusion,
                                system, neighbor_system,
                                particle, neighbor, pos_diff, distance, rho_a, rho_b)
    density_diffusion_term = dot(psi, grad_kernel) * volume_b

    smoothing_length_avg = (smoothing_length(system, particle) +
                            smoothing_length(neighbor_system, neighbor)) / 2
    dv[end, particle] += delta * smoothing_length_avg * 10.0 *
                         density_diffusion_term
end

# Density diffusion `nothing` or interaction other than fluid-fluid
@inline function density_diffusion!(dv, density_diffusion,
                                    system, neighbor_system, particle, neighbor,
                                    pos_diff, distance, m_b, rho_a, rho_b, grad_kernel)
    return dv
end
