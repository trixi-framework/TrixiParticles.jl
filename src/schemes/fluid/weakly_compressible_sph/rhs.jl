# Computes the forces that particles in `particle_system` experience from particles
# in `neighbor_system` and updates `dv` accordingly.
# It takes into account pressure forces, viscosity, and for `ContinuityDensity` updates
# the density using the continuity equation.
@inline @fastpow function dualsphysics_pressure(rho, pressure_constant, inverse_reference_density)
    density_ratio = rho * inverse_reference_density

    return pressure_constant * (density_ratio^7 - one(rho))
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi;
                   eachparticle=each_integrated_particle(particle_system),
                   kwargs...)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)
    state_equation = particle_system.state_equation
    reference_density = state_equation.reference_density
    pressure_constant = reference_density * sound_speed^2 / state_equation.exponent
    inverse_reference_density = inv(reference_density)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    backend = semi.parallelization_backend

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient divides
    # by zero. To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the compact support `c`, `distance^2` is in
    # the order of `c^2`, so we need to check `distance < sqrt(eps(c^2))`.
    # Note that `sqrt(eps(c^2)) != eps(c)`.
    compact_support_ = compact_support(particle_system, neighbor_system)
    almostzero = sqrt(eps(compact_support_^2))

    use_aligned_load_system = Val(use_aligned_vrho_load(v_particle_system, particle_system))
    use_aligned_load_neighbor = Val(use_aligned_vrho_load(v_neighbor_system,
                                                          neighbor_system))

    @threaded semi for particle in eachparticle
        # We are looping over the particles of `particle_system`, so it is guaranteed
        # that `particle` is in bounds of `particle_system`.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)

        # In 3D, this function can combine velocity and density load into one wide load,
        # which gives a significant speedup on GPUs.
        # Note that we can only safely use `@inbounds` after checking alignment
        # with `use_aligned_vrho_load` before the `@threaded` loop.
        (v_a,
         rho_a) = @inbounds velocity_and_density(v_particle_system, particle_system,
                                                 use_aligned_load_system, particle)
        p_a = dualsphysics_pressure(rho_a, pressure_constant,
                                    inverse_reference_density)

        # Accumulate the RHS contributions over all neighbors before writing to `dv`,
        # to reduce the number of memory writes.
        @inline function dv_drho_sum(a, b)
            dv_a, drho_a = a
            dv_b, drho_b = b
            return dv_a + dv_b, drho_a + drho_b
        end
        init = (zero(v_a), zero(rho_a))

        # Loop over all neighbors within the kernel cutoff.
        # Make sure that the returned names `dv_particle_` and `drho_particle_`
        # are not used inside the closure to avoid allocations.
        (dv_particle_,
         drho_particle_) = @inbounds mapreduce_neighbor(dv_drho_sum, system_coords,
                                                        neighbor_system_coords,
                                                        neighborhood_search,
                                                        backend, particle;
                                                        init) do particle, neighbor,
                                                                 pos_diff, distance
            # Skip neighbors with the same position because the kernel gradient is zero.
            # Note that `return` only exits the closure, i.e., skips the current neighbor.
            skip_zero_distance(particle_system) && distance < almostzero && return init

            # Now that we know that `distance` is not zero, we can safely call the unsafe
            # version of the kernel gradient to avoid redundant zero checks.
            grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                       distance, particle)

            # `foreach_neighbor` makes sure that `neighbor` is in bounds of `neighbor_system`
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            # Note that we can only safely use `@inbounds` after checking alignment
            # with `use_aligned_vrho_load` before the `@threaded` loop.
            (v_b,
             rho_b) = @inbounds velocity_and_density(v_neighbor_system, neighbor_system,
                                                     use_aligned_load_neighbor, neighbor)

            # DualSPHysics applies the state equation to every accepted fluid or boundary
            # neighbor directly in the interaction kernel.
            p_b = dualsphysics_pressure(rho_b, pressure_constant,
                                        inverse_reference_density)

            # Determine correction factors.
            # This can usually be ignored, as these are all 1 when no correction is used.
            (viscosity_correction, pressure_correction,
             surface_tension_correction) = free_surface_correction(correction,
                                                                   particle_system,
                                                                   rho_a, rho_b)

            # For `ContinuityDensity` without correction, this is equivalent to
            # dv_pressure = -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
            dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                                particle, neighbor,
                                                m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                distance, grad_kernel, correction)
            dv_particle = dv_pressure * pressure_correction

            # Propagate `@inbounds` to the viscosity function, which accesses particle data
            dv_particle = @inbounds dv_viscosity(dv_particle, particle_system,
                                                 neighbor_system,
                                                 v_particle_system, v_neighbor_system,
                                                 particle, neighbor, pos_diff, distance,
                                                 sound_speed, m_a, m_b, rho_a, rho_b,
                                                 v_a, v_b, grad_kernel,
                                                 viscosity_correction)

            # Extra terms in the momentum equation when using a shifting technique
            dv_particle = @inbounds dv_shifting(dv_particle,
                                                shifting_technique(particle_system),
                                                particle_system, neighbor_system,
                                                v_particle_system, v_neighbor_system,
                                                particle, neighbor, m_a, m_b, rho_a, rho_b,
                                                v_a, v_b, pos_diff, distance,
                                                grad_kernel, correction)

            dv_particle = @inbounds surface_tension_force(dv_particle,
                                                          surface_tension_a,
                                                          surface_tension_b,
                                                          particle_system, neighbor_system,
                                                          particle, neighbor, pos_diff,
                                                          distance,
                                                          rho_a, rho_b, grad_kernel,
                                                          surface_tension_correction)

            dv_particle = @inbounds adhesion_force(dv_particle, surface_tension_a,
                                                   particle_system, neighbor_system,
                                                   particle, neighbor, pos_diff, distance)

            drho_particle = zero(rho_a)

            # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
            # Propagate `@inbounds` to the continuity equation, which accesses particle data
            drho_particle = @inbounds continuity_equation(drho_particle, density_calculator,
                                                          particle_system, neighbor_system,
                                                          particle, neighbor, pos_diff,
                                                          distance,
                                                          m_b, rho_a, rho_b, v_a, v_b,
                                                          grad_kernel)

            return dv_particle, drho_particle
        end

        for i in eachindex(dv_particle_)
            @inbounds dv[i, particle] += dv_particle_[i]
        end
        @inbounds write_drho_particle!(dv, density_calculator, drho_particle_, particle)
    end

    return dv
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem{NDIMS},
                   neighbor_system::WeaklyCompressibleSPHSystem, semi) where NDIMS
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    # system_coords = vcat(system_coords, zero(drho)')
    # neighbor_system_coords = vcat(neighbor_system_coords, zero(drho)')

    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    cell_list = neighborhood_search.cell_list
    search_radius2 = PointNeighbors.search_radius(neighborhood_search)^2
    state_equation = particle_system.state_equation
    reference_density = state_equation.reference_density
    pressure_constant = reference_density * state_equation.sound_speed^2 /
                        state_equation.exponent
    inverse_reference_density = inv(reference_density)
    smoothing_length = particle_system.cache.smoothing_length
    inverse_smoothing_length = inv(smoothing_length)
    kernel_normalization = -2.7852f0 /
                           (smoothing_length^2 * smoothing_length^2)
    eta2 = particle_system.viscosity.epsilon * smoothing_length^2
    density_diffusion_factor = 2 * particle_system.density_diffusion.delta *
                               smoothing_length * state_equation.sound_speed

    backend = semi.parallelization_backend
    ndrange = length(each_integrated_particle(particle_system))
    mykernel(backend)(dv, system_coords, neighbor_system_coords, neighborhood_search,
                      cell_list, search_radius2, pressure_constant,
                      inverse_reference_density, smoothing_length,
                      inverse_smoothing_length, kernel_normalization, eta2,
                      density_diffusion_factor, v_particle_system, v_neighbor_system,
                      particle_system, neighbor_system; ndrange=ndrange,
                      workgroupsize=128)

    KernelAbstractions.synchronize(backend)

    return dv
end

@kernel function mykernel(dv,
                          system_coords, neighbor_system_coords,
                          nhs, cell_list, search_radius2,
                          pressure_constant, inverse_reference_density,
                          smoothing_length, inverse_smoothing_length,
                          kernel_normalization, eta2, density_diffusion_factor,
                          v_particle_system, v_neighbor_system,
                          particle_system::WeaklyCompressibleSPHSystem{NDIMS},
                          neighbor_system::WeaklyCompressibleSPHSystem) where NDIMS
    # `SymplecticPositionVerletWithSorting` deactivates out-of-bounds particles before
    # sorting, so active particles occupy the prefix used as the kernel launch range.
    particle = @index(Global)

    sound_speed = particle_system.state_equation.sound_speed
    # VT_coords = Vec{4, eltype(system_coords)}
    # point_coords_ = vloada(VT_coords, pointer(system_coords, 4*(particle-1)+1))
    # a, b, c, d = Tuple(point_coords_)
    # point_coords = SVector(a, b, c)
    point_coords = @inbounds extract_svector(system_coords, Val(NDIMS), particle)
    VT = SIMD.Vec{4, eltype(v_particle_system)}
    vrho_a = SIMD.vloada(VT, pointer(v_particle_system, 4*(particle-1)+1))
    a, b, c, d = Tuple(vrho_a)
    v_a = SVector(a, b, c)
    rho_a = d
    p_a = dualsphysics_pressure(rho_a, pressure_constant,
                                inverse_reference_density)
    # v_a = @inbounds extract_svector(v_particle_system, Val(NDIMS), particle)
    # rho_a = @inbounds v_particle_system[end, particle]

    dv_particle = zero(v_a)
    drho_particle = zero(rho_a)
    m_b = @inbounds neighbor_system.mass[1]

    cell = PointNeighbors.cell_coords(point_coords, nhs)
    VT_poscell = SIMD.Vec{4, eltype(nhs.relative_coords)}
    poscell_a = SIMD.vloada(VT_poscell,
                            pointer(nhs.relative_coords, 4 * (particle - 1) + 1))
    pos_a_x, pos_a_y, pos_a_z, encoded_cell_a = Tuple(poscell_a)
    cell_code_a = reinterpret(UInt32, encoded_cell_a)
    cell_a_x = Int32(cell_code_a >> 19)

    # Benchmark-only 3D traversal: x is the contiguous dimension of the full grid, so
    # visit the three x cells as one particle range for each y/z pair.
    for cell_z in (cell[3] - 1):(cell[3] + 1),
        cell_y in (cell[2] - 1):(cell[2] + 1)
        block_start = (cell[1] - 1, cell_y, cell_z)
        cell_index = @inbounds PointNeighbors.cell_index(cell_list, block_start)
        start = @inbounds cell_list.cells.first_bin_index[cell_index]
        stop = @inbounds cell_list.cells.first_bin_index[cell_index + 3] - 1

        offset_y = (cell[2] - cell_y) * nhs.cell_size[2]
        offset_z = (cell[3] - cell_z) * nhs.cell_size[3]

        for neighbor in start:stop
    # for neighbor_cell_ in PointNeighbors.neighboring_cells(cell, nhs)
    #     neighbor_cell = Tuple(neighbor_cell_)
    #     neighbors = @inbounds PointNeighbors.points_in_cell(neighbor_cell, nhs)

    #     for neighbor_ in eachindex(neighbors)
    #         neighbor = @inbounds neighbors[neighbor_]

            # neighbor_coords_ = vloada(VT_coords, pointer(neighbor_system_coords, 4*(neighbor-1)+1))
            # a, b, c, d = Tuple(neighbor_coords_)
            # neighbor_coords = SVector(a, b, c)
            poscell_b = SIMD.vloada(VT_poscell,
                                    pointer(nhs.relative_coords, 4 * (neighbor - 1) + 1))
            pos_b_x, pos_b_y, pos_b_z, encoded_cell_b = Tuple(poscell_b)
            cell_code_b = reinterpret(UInt32, encoded_cell_b)
            cell_b_x = Int32(cell_code_b >> 19)

            pos_diff = SVector(pos_a_x - pos_b_x +
                               (cell_a_x - cell_b_x) * nhs.cell_size[1],
                               pos_a_y - pos_b_y + offset_y,
                               pos_a_z - pos_b_z + offset_z)
            distance2 = dot(pos_diff, pos_diff)

            if eps(search_radius2) <= distance2 <= search_radius2
                distance = @fastmath sqrt(distance2)

                vrho_b = SIMD.vloada(VT, pointer(v_neighbor_system, 4*(neighbor-1)+1))
                a, b, c, d = Tuple(vrho_b)
                v_b = SVector(a, b, c)
                rho_b = d
                p_b = dualsphysics_pressure(rho_b, pressure_constant,
                                            inverse_reference_density)

                # v_b = @inbounds extract_svector(v_neighbor_system, Val(NDIMS), neighbor)
                # rho_b = @inbounds v_neighbor_system[end, neighbor]

                grad_kernel = kernel_grad_ds(pos_diff, distance,
                                             inverse_smoothing_length,
                                             kernel_normalization)

                # dv_particle += -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
                @fastmath dv_particle += -m_b * Base.FastMath.div_fast(p_a + p_b,
                                                             rho_a * rho_b) * grad_kernel

                @fastmath vdiff = v_a - v_b
                rho_ratio = Base.FastMath.div_fast(rho_a, rho_b)
                # drho_particle += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)
                @fastmath drho_particle += rho_ratio * m_b * dot(vdiff, grad_kernel)

                alpha = particle_system.viscosity.alpha
                dot3 = dot(pos_diff, grad_kernel)
                @fastmath diffusion = Base.FastMath.div_fast(density_diffusion_factor *
                                                   (rho_ratio - one(rho_ratio)),
                                                   distance2 + eta2)
                @fastmath drho_particle += diffusion * dot3 * m_b

                vr = dot(vdiff, pos_diff)
                if vr < 0
                    mu = Base.FastMath.div_fast(smoothing_length * vr,
                                                distance2 + eta2)
                    rho_mean = (rho_a + rho_b) / 2
                    # @fastmath pi_ab = (alpha * sound_speed * mu) / rho_mean * grad_kernel
                    pi_ab = Base.FastMath.div_fast(alpha * sound_speed * mu, rho_mean) * grad_kernel
                    @fastmath dv_particle += m_b * pi_ab
                end
            end
        end
    end

    for i in eachindex(dv_particle)
        @inbounds dv[i, particle] += dv_particle[i]
        # Debug example
        # debug_array[i, particle] += dv_pressure[i]
    end
    @inbounds dv[end, particle] += drho_particle
end

@inline function kernel_grad_ds(pos_diff, r, inverse_smoothing_length,
                                kernel_normalization)
    q = r * inverse_smoothing_length
    wqq1 = (1 - q / 2)
    return kernel_normalization * wqq1 * wqq1 * wqq1 * pos_diff
end

@propagate_inbounds function neighbor_pressure(v_neighbor_system, neighbor_system,
                                               neighbor, p_a)
    return current_pressure(v_neighbor_system, neighbor_system, neighbor)
end

@inline function neighbor_pressure(v_neighbor_system,
                                   neighbor_system::WallBoundarySystem{<:BoundaryModelDummyParticles{PressureMirroring}},
                                   neighbor, p_a)
    return p_a
end

# Default method, which simply calls `current_velocity` and `current_density` separately.
@propagate_inbounds function velocity_and_density(v, system, ::Val{false}, particle)
    v_particle = current_velocity(v, system, particle)
    rho_particle = current_density(v, system, particle)

    return v_particle, rho_particle
end

# Optimized version for WCSPH with `ContinuityDensity` in 3D,
# which combines the velocity and density load into one wide load.
# This is significantly faster on GPUs than the 4 individual loads of `extract_svector`.
# WARNING: this requires that the pointer of `v` is aligned to `4 * sizeof(eltype(v))`,
#          which is checked by `use_aligned_vrho_load`.
#          Only call this function after checking `use_aligned_vrho_load` to avoid
#          segmentation faults from illegal accesses.
@propagate_inbounds function velocity_and_density(v, system, ::Val{true}, particle)
    vrho_particle = extract_svector_aligned(v, Val(4), particle)

    # The columns of `v` are ordered as (v_x, v_y, v_z, rho)
    v..., rho = Tuple(vrho_particle)
    v_particle = SVector(v)

    return v_particle, rho
end

# By default, don't use aligned loads
use_aligned_vrho_load(v, system) = false

function use_aligned_vrho_load(v::AbstractGPUArray, system::WeaklyCompressibleSPHSystem{3})
    use_aligned_vrho_load(v, system, system.density_calculator)
end

use_aligned_vrho_load(v, system, density_calculator) = false

# Only use aligned loads when all of these conditions are satisfied:
# - WCSPH with `ContinuityDensity` in 3D. Only then, the columns of `v` are of length 4.
# - We are on a GPU, where the aligned load gives a significant speedup.
# - The velocity array is aligned for aligned loads, which requires that the pointer of `v`
#   is aligned to `4 * sizeof(eltype(v))`
#   Otherwise, we cannot use `vloada`, which is an *aligned* load.
#   The unaligned version `vload` does not produce wide load instructions on GPUs.
function use_aligned_vrho_load(v::AbstractGPUArray, system, ::ContinuityDensity)
    if !can_use_aligned_load(v, 4)
        # Aligned loads should always be possible on GPUs because the slices of `v_ode`
        # are aligned to 64 bytes in `Semidiscretization` and arrays on GPUs are always
        # aligned to full pages.
        error("illegal alignment of `v` integration array. Please report this issue.")
    end

    return true
end
