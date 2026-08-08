# Structure-structure interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::TotalLagrangianSPHSystem, semi;
                   eachparticle=each_integrated_particle(particle_system))
    # Different structures do not interact with each other (yet)
    particle_system === neighbor_system || return dv

    interact_structure_structure!(dv, v_particle_system, particle_system, semi;
                                  eachparticle)
end

# Function barrier without dispatch for unit testing
@inline function interact_structure_structure!(dv, v_system, system, semi;
                                               eachparticle=each_integrated_particle(system))
    (; penalty_force) = system

    # Everything here is done in the initial coordinates
    system_coords = initial_coordinates(system)
    neighborhood_search = get_neighborhood_search(system, semi)
    backend = semi.parallelization_backend

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient
    # and the density diffusion divide by zero.
    # To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the smoothing length `h`, `distance^2` is in
    # the order of `h^2`, so we need to check `distance < sqrt(eps(h^2))`.
    # Note that `sqrt(eps(h^2)) != eps(h)`.
    h = initial_smoothing_length(system)
    almostzero = sqrt(eps(h^2))

    # Check alignment of deformation gradient and `pk1_rho2` arrays before the `@threaded`
    # loop to be able to use aligned loads safely inside the loop.
    use_aligned_matrix_load_ = Val(use_aligned_matrix_load(system))

    @threaded semi for particle in eachparticle
        # We are looping over the particles of `system`, so it is guaranteed
        # that `particle` is in bounds of `system`.
        m_a = @inbounds system.mass[particle]
        rho_a = @inbounds system.material_density[particle]
        # PK1 / rho^2
        pk1_rho2_a = @inbounds pk1_rho2(system, use_aligned_matrix_load_, particle)
        current_coords_a = @inbounds current_coords(system, particle)
        F_a = @inbounds deformation_gradient(system, use_aligned_matrix_load_, particle)

        # Accumulate the RHS contributions over all neighbors before writing to `dv`
        # to reduce the number of memory writes.
        # Make sure that the returned name `dv_particle_` is not used inside the closure
        # to avoid allocations.
        dv_particle_ = @inbounds mapreduce_neighbor(+, system_coords, system_coords,
                                                    neighborhood_search, backend, particle;
                                                    init=zero(current_coords_a)) do particle,
                                                                                    neighbor,
                                                                                    initial_pos_diff,
                                                                                    initial_distance

            # Skip neighbors with the same position because the kernel gradient is zero.
            # Note that `return` only exits the closure, i.e., skips the current neighbor.
            if skip_zero_distance(system) && initial_distance < almostzero
                return zero(initial_pos_diff)
            end

            # Now that we know that `distance` is not zero, we can safely call the unsafe
            # version of the kernel gradient to avoid redundant zero checks.
            grad_kernel = smoothing_kernel_grad_unsafe(system, initial_pos_diff,
                                                       initial_distance, particle)

            rho_b = @inbounds system.material_density[neighbor]
            m_b = @inbounds system.mass[neighbor]
            # PK1 / rho^2
            pk1_rho2_b = @inbounds pk1_rho2(system, use_aligned_matrix_load_, neighbor)
            current_coords_b = @inbounds current_coords(system, neighbor)

            # The compiler is smart enough to optimize this away if no penalty force is used
            F_b = @inbounds deformation_gradient(system, use_aligned_matrix_load_,
                                                 neighbor)

            current_pos_diff_ = current_coords_a - current_coords_b
            # In mixed-precision simulations, convert from `coordinates_eltype(system)`
            # to `eltype(system)` immediately after computing the difference.
            current_pos_diff = convert.(eltype(system), current_pos_diff_)
            current_distance = norm(current_pos_diff)

            dv_particle = m_b * (pk1_rho2_a + pk1_rho2_b) * grad_kernel

            dv_particle = @inbounds dv_penalty_force(dv_particle, penalty_force,
                                                     particle, neighbor,
                                                     initial_pos_diff, initial_distance,
                                                     current_pos_diff, current_distance,
                                                     system, m_a, m_b, rho_a, rho_b,
                                                     F_a, F_b)

            dv_particle = @inbounds dv_viscosity_tlsph(dv_particle, system, v_system,
                                                       particle, neighbor,
                                                       current_pos_diff, current_distance,
                                                       m_a, m_b, rho_a, rho_b, F_a,
                                                       grad_kernel)

            return dv_particle
        end

        for i in 1:ndims(system)
            @inbounds dv[i, particle] += dv_particle_[i]
        end

        # TODO continuity equation for boundary model with `ContinuityDensity`?
    end

    return dv
end

# Structure-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::AbstractFluidSystem, semi;
                   eachparticle=each_integrated_particle(particle_system))
    return interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                     v_neighbor_system, u_neighbor_system,
                                     particle_system, neighbor_system, semi; eachparticle)
end

# Structure-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::Union{WallBoundarySystem, OpenBoundarySystem}, semi;
                   eachparticle=each_integrated_particle(particle_system))
    # TODO continuity equation?
    return dv
end

function use_aligned_matrix_load(system)
    return use_aligned_matrix_load(system.deformation_grad, system.pk1_rho2)
end

function use_aligned_matrix_load(deformation_grad::AbstractGPUArray,
                                 pk1_rho2::AbstractGPUArray)
    # Aligned loads should always be possible on GPUs because GPU arrays are always aligned
    # to full pages, and these arrays are not slices of larger arrays.
    if !can_use_aligned_load(deformation_grad, 4)
        error("illegal alignment of deformation gradient array. Please report this issue.")
    end
    if !can_use_aligned_load(pk1_rho2, 4)
        error("illegal alignment of `pk1_rho2` array. Please report this issue.")
    end

    return true
end

# Don't use aligned vector loads on the CPU. For large arrays, alignment to 32 bytes
# (4 * Float64) is usually given, but it is not guaranteed, as Julia only guarantees
# alignment to 16 bytes. However, the non-aligned `vload` used in `extract_smatrix` in 2D
# has the same performance as the aligned `vloada` in `extract_smatrix_aligned` on the CPU.
use_aligned_matrix_load(deformation_grad, pk1_rho2) = false

# Aligned vector load versions for deformation gradient and `pk1_rho2`.
# These are only used on GPUs, which is checked by `use_aligned_matrix_load`.
@propagate_inbounds function pk1_rho2(system, ::Val{true}, particle)
    return extract_smatrix_aligned(system.pk1_rho2, system, particle)
end

@propagate_inbounds function pk1_rho2(system, ::Val{false}, particle)
    return pk1_rho2(system, particle)
end

@propagate_inbounds function deformation_gradient(system, ::Val{true}, particle)
    return extract_smatrix_aligned(system.deformation_grad, system, particle)
end

@propagate_inbounds function deformation_gradient(system, ::Val{false}, particle)
    return deformation_gradient(system, particle)
end
