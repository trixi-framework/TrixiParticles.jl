"""
    ShiftingHaftu(; max_iterations=100, tolerance=2e-3, theta=0.5, max_shift_factor=0.05)

Shifting of the particles after splitting and merging by Haftu et al. (Eq. 15 and 36),
to be passed as `shifting_technique` to `ParticleRefinement`.
The particle properties are corrected with a Taylor series expansion (Eq. 37).

The particles are shifted iteratively. In each iteration, they are shifted by `theta`
times the shift of Eq. 15, but by at most `max_shift_factor` times their smoothing length.
The iteration stops when the largest shift of Eq. 15 is smaller than `tolerance` times
the smoothing length of its particle, or after `max_iterations` iterations.
The code of Haftu et al. instead uses three iterations with `theta=1` and a maximum shift
of `0.00625` times the smoothing length.

Velocity and pressure are corrected for all density calculators. The density is only
corrected with [`ContinuityDensity`](@ref), since [`SummationDensity`](@ref) computes
the density from the shifted positions before it is used again.
"""
struct ShiftingHaftu{ELTYPE}
    max_iterations   :: Int
    tolerance        :: ELTYPE
    theta            :: ELTYPE
    max_shift_factor :: ELTYPE
end

function ShiftingHaftu(; max_iterations=100, tolerance=2e-3, theta=0.5,
                       max_shift_factor=0.05)
    return ShiftingHaftu(max_iterations, promote(tolerance, theta, max_shift_factor)...)
end

"""
    shift_particles!(semi, v_ode, u_ode)

Shift the particles of fluid systems with particle refinement after splitting and merging
with the `shifting_technique` of their `ParticleRefinement` (see [`ShiftingHaftu`](@ref)).
After shifting, the smoothing lengths are updated with the average mass of the neighbors
(Eq. 35 in Haftu et al.). This is also done with `shifting_technique=nothing`.
"""
function shift_particles!(semi, v_ode, u_ode)
    foreach_system(semi) do system
        shift_particles!(system, v_ode, u_ode, semi)
    end

    # Update all neighborhood searches that contain the shifted particles
    update_nhs!(semi, u_ode)

    return semi
end

@inline shift_particles!(system, v_ode, u_ode, semi) = system

@inline function shift_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi)
    return shift_particles!(system, get_refinement(system), v_ode, u_ode, semi)
end

@inline shift_particles!(system, ::Nothing, v_ode, u_ode, semi) = system

@inline function shift_particles!(system, refinement, v_ode, u_ode, semi)
    return apply_refinement_shifting!(system, refinement.shifting_technique,
                                      v_ode, u_ode, semi)
end

# Without shifting, only the smoothing lengths are updated to match the masses
# after splitting and merging
function apply_refinement_shifting!(system, ::Nothing, v_ode, u_ode, semi)
    u = wrap_u(u_ode, system, semi)
    update_smoothing_lengths!(system, u, semi)

    return system
end

function apply_refinement_shifting!(system, shifting::ShiftingHaftu, v_ode, u_ode, semi)
    (; total_shift) = system.cache
    (; max_iterations) = shifting
    ELTYPE = eltype(system)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    tolerance = convert(ELTYPE, shifting.tolerance)
    theta = convert(ELTYPE, shifting.theta)
    max_shift_factor = convert(ELTYPE, shifting.max_shift_factor)

    set_zero!(total_shift)

    # Shift with smoothing lengths that match the masses after splitting and merging
    update_smoothing_lengths!(system, u, semi)

    for _ in 1:max_iterations
        compute_shift!(system, u, u_ode, semi)
        max_relative_shift = apply_shift!(system, u, semi, theta, max_shift_factor)

        max_relative_shift < tolerance && break
    end

    correct_properties!(system, v, u, v_ode, u_ode, semi)

    # Smoothing lengths for the shifted positions
    update_smoothing_lengths!(system, u, semi)

    return system
end

# The shift of Eq. 15 in Haftu et al. As in their code, all systems are included,
# so that particles are also shifted away from boundaries.
@fastpow function compute_shift!(system, u, u_ode, semi)
    (; shift) = system.cache
    (; reference_density) = get_refinement(system)
    smoothing_kernel = system_smoothing_kernel(system)
    xi = convert(eltype(system), kernel_inflection_point(smoothing_kernel))

    set_zero!(shift)

    system_coords = current_coordinates(u, system)

    foreach_system(semi) do neighbor_system
        has_system_interaction(system, neighbor_system, semi) || return

        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            h_a, h_b = pair_smoothing_lengths(system, neighbor_system, particle, neighbor)
            h_ab = (h_a + h_b) / 2
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

            W = kernel(smoothing_kernel, distance, h_ab)
            W_xi = kernel(smoothing_kernel, xi * h_ab, h_ab)
            grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, h_ab)

            # Eq. 15
            shift_ = -h_a^2 / 2 * m_b / reference_density *
                     (1 + 24 * (W / W_xi)^4 / 100) * grad_kernel

            @inbounds add_svector_flat!(shift, shift_, particle)
        end
    end

    return system
end

# Shift the particles by `theta` times the shift of Eq. 15, but by at most
# `max_shift_factor` times their smoothing length. Return the largest shift of Eq. 15
# divided by the smoothing length of its particle.
function apply_shift!(system, u, semi, theta, max_shift_factor)
    (; shift, total_shift, relative_shift) = system.cache
    NDIMS = ndims(system)

    set_zero!(relative_shift)

    @threaded semi for particle in each_integrated_particle(system)
        shift_a = extract_svector_flat(shift, Val(NDIMS), particle)
        shift_norm = norm(shift_a)
        h_a = smoothing_length(system, particle)

        max_shift = max_shift_factor * h_a
        factor = theta * shift_norm > max_shift ? max_shift / shift_norm : theta
        applied_shift = factor * shift_a

        position = current_coords(u, system, particle) + applied_shift
        set_particle_position!(u, system, particle, position)
        add_svector_flat!(total_shift, applied_shift, particle)

        relative_shift[particle] = shift_norm / h_a
    end

    # Only the self-interaction neighborhood search is used in the next iteration
    neighborhood_search = get_neighborhood_search(system, semi)
    update_nhs!(neighborhood_search, system, system, u, u, semi)

    return maximum(relative_shift)
end

# Correct the particle properties after shifting with a Taylor series expansion (Eq. 37),
# `ϕ_a' = ϕ_a + (∇ϕ)_a ⋅ δr_a` with `(∇ϕ)_a = ∑_b V_b (ϕ_b - ϕ_a) ∇W_ab`, where `δr_a`
# is the total shift of all iterations. As in the code of Haftu et al., the gradients are
# computed at the shifted positions and all systems are included.
function correct_properties!(system, v, u, v_ode, u_ode, semi)
    (; total_shift, velocity_correction, pressure_correction,
     density_correction) = system.cache
    smoothing_kernel = system_smoothing_kernel(system)
    NDIMS = ndims(system)

    set_zero!(velocity_correction)
    set_zero!(pressure_correction)
    set_zero!(density_correction)

    system_coords = current_coordinates(u, system)

    foreach_system(semi) do neighbor_system
        has_system_interaction(system, neighbor_system, semi) || return

        v_neighbor = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            h_a, h_b = pair_smoothing_lengths(system, neighbor_system, particle, neighbor)
            grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, (h_a + h_b) / 2)

            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            rho_a = @inbounds current_density(v, system, particle)
            rho_b = @inbounds current_density(v_neighbor, neighbor_system, neighbor)
            v_a = @inbounds current_velocity(v, system, particle)
            v_b = @inbounds current_velocity(v_neighbor, neighbor_system, neighbor)
            p_a = @inbounds current_pressure(v, system, particle)
            p_b = @inbounds current_pressure(v_neighbor, neighbor_system, neighbor)

            shift_a = @inbounds extract_svector_flat(total_shift, Val(NDIMS), particle)
            weight = m_b / rho_b * dot(grad_kernel, shift_a)

            @inbounds add_svector_flat!(velocity_correction, weight * (v_b - v_a),
                                        particle)
            @inbounds pressure_correction[particle] += weight * (p_b - p_a)
            @inbounds density_correction[particle] += weight * (rho_b - rho_a)
        end
    end

    @threaded semi for particle in each_integrated_particle(system)
        velocity = current_velocity(v, system, particle) +
                   extract_svector_flat(velocity_correction, Val(NDIMS), particle)
        pressure = current_pressure(v, system, particle) + pressure_correction[particle]

        set_particle_velocity!(v, system, particle, velocity)
        set_particle_pressure!(v, system, particle, pressure)
        correct_density!(v, system, system.density_calculator, particle,
                         density_correction[particle])
    end

    return system
end

@inline function correct_density!(v, system, ::ContinuityDensity, particle, correction)
    density = current_density(v, system, particle) + correction
    set_particle_density!(v, system, particle, density)

    return system
end

# The summation density is computed from the shifted positions before it is used again
@inline correct_density!(v, system, density_calculator, particle, correction) = system

# Update the smoothing lengths with the average mass of the neighbors (Eq. 35).
# As in `neighborhood_spacing`, only the particles of the same system within the compact
# support of `particle` are considered, including the particle itself.
function update_smoothing_lengths!(system, u, semi)
    (; reference_density) = get_refinement(system)
    smoothing_kernel = system_smoothing_kernel(system)
    ELTYPE = eltype(system)

    system_coords = current_coordinates(u, system)
    neighborhood_search = get_neighborhood_search(system, semi)

    # Sum of the masses and number of neighbors
    init = (zero(ELTYPE), 0)
    op(a, b) = (a[1] + b[1], a[2] + b[2])

    # Each thread only reads and writes the smoothing length of its own particle
    @threaded semi for particle in each_integrated_particle(system)
        radius = compact_support(smoothing_kernel, smoothing_length(system, particle))

        mass_sum,
        n_neighbors = mapreduce_neighbor(op, system_coords, system_coords,
                                         neighborhood_search,
                                         semi.parallelization_backend,
                                         particle; init) do particle, neighbor, _, distance
            distance > radius && return init

            return (hydrodynamic_mass(system, neighbor), 1)
        end

        # The particle itself is always a neighbor
        mass_avg = mass_sum / n_neighbors
        spacing = (mass_avg / reference_density)^(1 / ndims(system))
        set_particle_smoothing_length!(system, particle,
                                       spacing_to_smoothing_length(system, spacing))
    end

    return system
end

# The point of inflection `ξ` of the kernel in units of the smoothing length,
# where the second derivative of the kernel is zero (see Eq. 15 in Haftu et al.)
function kernel_inflection_point(smoothing_kernel)
    throw(ArgumentError("shifting with particle refinement is not supported for the " *
                        "kernel `$(nameof(typeof(smoothing_kernel)))`"))
end

@inline kernel_inflection_point(::SchoenbergCubicSplineKernel) = 2 / 3
@inline kernel_inflection_point(::SchoenbergQuinticSplineKernel) = 0.759298480738450
@inline kernel_inflection_point(::WendlandC2Kernel) = 1 / 2
