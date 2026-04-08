# Computes the forces that particles in `particle_system` experience from particles
# in `neighbor_system` and updates `dv` accordingly.
# It takes into account pressure forces, viscosity, and for `ContinuityDensity` updates
# the density using the continuity equation.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)

    # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient divides
    # by zero. To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the compact support `c`, `distance^2` is in
    # the order of `c^2`, so we need to check `distance < sqrt(eps(c^2))`.
    # Note that `sqrt(eps(c^2)) != eps(c)`.
    compact_support_ = compact_support(particle_system, neighbor_system)
    almostzero = sqrt(eps(compact_support_^2))

    @threaded semi for particle in each_integrated_particle(particle_system)
        # We are looping over the particles of `particle_system`, so it is guaranteed
        # that `particle` is in bounds of `particle_system`.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)

        # In 3D, this function can combine velocity and density load into one wide load,
        # which gives a significant speedup on GPUs.
        (v_a,
         rho_a) = @inbounds velocity_and_density(v_particle_system, particle_system,
                                                 particle)

        # Accumulate the RHS contributions over all neighbors before writing to `dv`,
        # to reduce the number of memory writes.
        # Note that we need a `Ref` in order to be able to update these variables
        # inside the closure in the `foreach_neighbor` loop.
        dv_particle = Ref(zero(v_a))
        drho_particle = Ref(zero(rho_a))

        # Loop over all neighbors within the kernel cutoff
        @inbounds PointNeighbors.foreach_neighbor(system_coords, neighbor_system_coords,
                                                  neighborhood_search,
                                                  particle) do particle, neighbor,
                                                               pos_diff, distance
            # Skip neighbors with the same position because the kernel gradient is zero.
            # Note that `return` only exits the closure, i.e., skips the current neighbor.
            skip_zero_distance(particle_system, distance, almostzero) && return

            # Now that we know that `distance` is not zero, we can safely call the unsafe
            # version of the kernel gradient to avoid redundant zero checks.
            grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                       distance, particle)

            # `foreach_neighbor` makes sure that `neighbor` is in bounds of `neighbor_system`
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            (v_b,
             rho_b) = @inbounds velocity_and_density(v_neighbor_system, neighbor_system,
                                                     neighbor)
            rho_mean = (rho_a + rho_b) / 2
            vdiff = v_a - v_b

            # The following call is equivalent to
            #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
            # Only when the neighbor system is a `WallBoundarySystem`
            # or a `TotalLagrangianSPHSystem` with the boundary model `PressureMirroring`,
            # this will return `p_b = p_a`, which is the pressure of the fluid particle.
            p_b = @inbounds neighbor_pressure(v_neighbor_system, neighbor_system,
                                              neighbor, p_a)

            # For `ContinuityDensity` without correction, this is equivalent to
            # dv_pressure = -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
            dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                                particle, neighbor,
                                                m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                distance, grad_kernel, correction)

            # Determine correction factors.
            # This can usually be ignored, as these are all 1 when no correction is used.
            (viscosity_correction, pressure_correction,
             surface_tension_correction) = free_surface_correction(correction,
                                                                   particle_system,
                                                                   rho_a, rho_b)

            # Accumulate contributions over all neighbors
            dv_particle[] += dv_pressure * pressure_correction

            # Propagate `@inbounds` to the viscosity function, which accesses particle data
            @inbounds dv_viscosity(dv_particle,
                                   particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, pos_diff, distance,
                                   sound_speed, m_a, m_b, rho_a, rho_b,
                                   v_a, v_b, grad_kernel)

            # Extra terms in the momentum equation when using a shifting technique
            @inbounds dv_shifting!(dv_particle, shifting_technique(particle_system),
                                   particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, m_a, m_b, rho_a, rho_b,
                                   pos_diff, distance, grad_kernel, correction)

            # TODO surface_tension_correction
            @inbounds surface_tension_force!(dv_particle, surface_tension_a,
                                             surface_tension_b,
                                             particle_system, neighbor_system,
                                             particle, neighbor, pos_diff, distance,
                                             rho_a, rho_b, grad_kernel)

            @inbounds adhesion_force!(dv_particle, surface_tension_a, particle_system,
                                      neighbor_system,
                                      particle, neighbor, pos_diff, distance)

            # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
            # Propagate `@inbounds` to the continuity equation, which accesses particle data
            @inbounds continuity_equation!(drho_particle, density_calculator,
                                           particle_system, neighbor_system,
                                           v_particle_system, v_neighbor_system,
                                           particle, neighbor, pos_diff, distance,
                                           m_b, rho_a, rho_b, vdiff, grad_kernel)
        end

        for i in eachindex(dv_particle[])
            @inbounds dv[i, particle] += dv_particle[][i]
        end
        @inbounds dv[end, particle] += drho_particle[]
    end

    return dv
end


# for cell in cells
#     for point in cell
#         for neighbor_cell in neighbor_cells

#             for neighbor in neighbor_cell
#
# 3.84 ms vs 2.92 ms for the interact! above
function interact_localmem!(dv, v_particle_system, u_particle_system,
                            v_neighbor_system, u_neighbor_system,
                            particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    search_radius = PointNeighbors.search_radius(neighborhood_search)

    almostzero = eps(search_radius^2)

    backend = semi.parallelization_backend
    # max_particles_per_cell = 64
    # nhs_size = size(neighborhood_search.cell_list.linear_indices)
    # cells = CartesianIndices(ntuple(i -> 2:(nhs_size[i] - 1), ndims(neighborhood_search)))
    linear_indices = neighborhood_search.cell_list.linear_indices
    cartesian_indices = CartesianIndices(size(linear_indices))
    lengths = Array(neighborhood_search.cell_list.cells.lengths)
    max_particles_per_cell = maximum(lengths)
    # max_particles_per_cell = 64
    nonempty_cells = Adapt.adapt(backend, filter(index -> lengths[linear_indices[index]] > 0, cartesian_indices))
    ndrange = max_particles_per_cell * length(nonempty_cells)
    kernel = foreach_neighbor_localmem(backend, Int64(max_particles_per_cell))
    kernel(dv, system_coords, neighbor_coords, neighborhood_search, nonempty_cells,
           Val(max_particles_per_cell), search_radius, almostzero,
           particle_system, neighbor_system, v_particle_system, v_neighbor_system,
           sound_speed, density_calculator; ndrange)

    KernelAbstractions.synchronize(backend)

    return nothing
end

@inline function copy_to_localmem!(local_points, local_neighbor_coords,
                                    local_neighbor_vrho, local_neighbor_mass,
                                    local_neighbor_pressure,
                                    v_neighbor_system, neighbor_system,
                                   neighbor_cell, neighbor_system_coords,
                                   neighborhood_search, particleidx)
    points_view = @inbounds PointNeighbors.points_in_cell(neighbor_cell, neighborhood_search)
    n_particles_in_neighbor_cell = length(points_view)

    # First use all threads to load the neighbors into local memory in parallel
    if particleidx <= n_particles_in_neighbor_cell
        @inbounds p = local_points[particleidx] = points_view[particleidx]
        for d in 1:ndims(neighborhood_search)
            @inbounds local_neighbor_coords[d, particleidx] = neighbor_system_coords[d, p]
        end
        v, rho = @inbounds velocity_and_density(v_neighbor_system, neighbor_system, p)
        for d in 1:ndims(neighborhood_search)
            @inbounds local_neighbor_vrho[d, particleidx] = v[d]
        end
        @inbounds local_neighbor_vrho[ndims(neighborhood_search) + 1, particleidx] = rho

        m = @inbounds hydrodynamic_mass(neighbor_system, p)
        @inbounds local_neighbor_mass[particleidx] = m
        pressure = @inbounds current_pressure(v_neighbor_system, neighbor_system, p)
        @inbounds local_neighbor_pressure[particleidx] = pressure
    end
    return n_particles_in_neighbor_cell
end

# @parallel(block) for cell in cells
#     for neighbor_cell in neighboring_cells
#         @parallel(thread) for neighbor in neighbor_cell
#             copy_coordinates_to_localmem(neighbor)
#
#         # Make sure all threads finished the copying
#         @synchronize
#
#         @parallel(thread) for particle in cell
#             for neighbor in neighbor_cell
#                 # This uses the neighbor coordinates from the local memory
#                 compute(point, neighbor)
#
#         # Make sure all threads finished computing before we continue with copying
#         @synchronize
@kernel cpu=false function foreach_neighbor_localmem(dv, system_coords, neighbor_system_coords,
                               neighborhood_search, cells, ::Val{MAX}, search_radius,
                               almostzero, particle_system, neighbor_system,
                               v_particle_system, v_neighbor_system, sound_speed,
                               density_calculator) where {MAX}
    cell_ = @index(Group)
    cell = @inbounds Tuple(cells[cell_])
    particleidx = @index(Local)
    @assert 1 <= particleidx <= MAX

    # Coordinate buffer in local memory
    local_points = @localmem Int32 MAX
    local_neighbor_coords = @localmem eltype(system_coords) (ndims(neighborhood_search), MAX)
    local_neighbor_vrho = @localmem eltype(system_coords) (ndims(neighborhood_search) + 1, MAX)
    local_neighbor_mass = @localmem eltype(system_coords) MAX
    local_neighbor_pressure = @localmem eltype(system_coords) MAX

    points = @inbounds PointNeighbors.points_in_cell(cell, neighborhood_search)
    n_particles_in_current_cell = length(points)

    # Extract point coordinates if a point lies on this thread
    if particleidx <= n_particles_in_current_cell
        particle = @inbounds points[particleidx]
        point_coords = @inbounds extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                                 particle)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)

        # In 3D, this function can combine velocity and density load into one wide load,
        # which gives a significant speedup on GPUs.
        (v_a,
         rho_a) = @inbounds velocity_and_density(v_particle_system, particle_system,
                                                 particle)
    else
        particle = zero(Int32)
        point_coords = zero(SVector{ndims(neighborhood_search), eltype(system_coords)})

        m_a = zero(eltype(system_coords))
        p_a = zero(eltype(system_coords))
        v_a = zero(SVector{ndims(neighborhood_search), eltype(system_coords)})
        rho_a = zero(eltype(system_coords))
    end

    dv_particle = zero(v_a)
    drho_particle = zero(rho_a)

    for neighbor_cell_ in PointNeighbors.neighboring_cells(cell, neighborhood_search)
        neighbor_cell = Tuple(neighbor_cell_)

        n_particles_in_neighbor_cell = copy_to_localmem!(local_points, local_neighbor_coords,
                                                         local_neighbor_vrho, local_neighbor_mass,
                                                         local_neighbor_pressure,
                                                            v_neighbor_system, neighbor_system,
                                                         neighbor_cell, neighbor_system_coords,
                                                         neighborhood_search, particleidx)

        # Make sure all threads finished the copying
        @synchronize

        # Now each thread works on one particle again
        if particleidx <= n_particles_in_current_cell
            for local_neighbor in 1:n_particles_in_neighbor_cell
                @inbounds neighbor = local_points[local_neighbor]
                @inbounds neighbor_coords = extract_svector(local_neighbor_coords,
                                                            Val(ndims(neighborhood_search)),
                                                            local_neighbor)

                pos_diff = point_coords - neighbor_coords
                distance2 = dot(pos_diff, pos_diff)

                # TODO periodic

                if almostzero < distance2 <= search_radius^2
                    distance = sqrt(distance2)

                    # Now that we know that `distance` is not zero, we can safely call the unsafe
                    # version of the kernel gradient to avoid redundant zero checks.
                    grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                               distance, particle)

                    m_b = @inbounds local_neighbor_mass[local_neighbor]
                    p_b = @inbounds local_neighbor_pressure[local_neighbor]
                    # v_rho_b = @inbounds extract_svector(local_neighbor_vrho, Val(ndims(neighborhood_search) + 1),
                    #                                     local_neighbor)
                    # v_b = v_rho_b[1:ndims(neighborhood_search)]
                    # v_b = @inbounds SVector(v_rho_b[1], v_rho_b[2], v_rho_b[3])
                    # rho_b = @inbounds v_rho_b[ndims(neighborhood_search) + 1]
                    (v_b, rho_b) = @inbounds velocity_and_density(local_neighbor_vrho, neighbor_system,
                                                                  local_neighbor)

                    # `foreach_neighbor` makes sure that `neighbor` is in bounds of `neighbor_system`
                    # m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
                    # (v_b,
                    # rho_b) = @inbounds velocity_and_density(v_neighbor_system, neighbor_system,
                    #                                         neighbor)
                    # v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)
                    # rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
                    vdiff = v_a - v_b

                    # The following call is equivalent to
                    #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
                    # Only when the neighbor system is a `WallBoundarySystem`
                    # or a `TotalLagrangianSPHSystem` with the boundary model `PressureMirroring`,
                    # this will return `p_b = p_a`, which is the pressure of the fluid particle.
                    # p_b = @inbounds neighbor_pressure(v_neighbor_system, neighbor_system,
                    #                                 neighbor, p_a)

                    # For `ContinuityDensity` without correction, this is equivalent to
                    # dv_pressure = -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
                    dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                                        particle, neighbor,
                                                        m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                        distance, grad_kernel, nothing)

                    # Accumulate contributions over all neighbors
                    dv_particle += dv_pressure

                    # Propagate `@inbounds` to the viscosity function, which accesses particle data
                    dv_viscosity_ = Ref(zero(dv_particle))
                    @inbounds dv_viscosity(dv_viscosity_,
                                        particle_system, neighbor_system,
                                        v_particle_system, v_neighbor_system,
                                        particle, neighbor, pos_diff, distance,
                                        sound_speed, m_a, m_b, rho_a, rho_b,
                                        v_a, v_b, grad_kernel)
                    dv_particle += dv_viscosity_[]

                    # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
                    # Propagate `@inbounds` to the continuity equation, which accesses particle data
                    drho_ = Ref(zero(drho_particle))
                    @inbounds continuity_equation!(drho_, density_calculator,
                                                particle_system, neighbor_system,
                                                v_particle_system, v_neighbor_system,
                                                particle, neighbor, pos_diff, distance,
                                                m_b, rho_a, rho_b, vdiff, grad_kernel)
                    drho_particle += drho_[]
                end
            end
        end

        # Make sure all threads finished computing before we continue with copying
        @synchronize()
    end

    if particleidx <= n_particles_in_current_cell
        for i in eachindex(dv_particle)
            @inbounds dv[i, particle] += dv_particle[i]
        end
        @inbounds dv[end, particle] += drho_particle
    end
end

# Same as above, but local memory is used for 3 cells at once to reduce the number of synchronizations.
# 3.64 ms
function interact_localmem2!(dv, v_particle_system, u_particle_system,
                            v_neighbor_system, u_neighbor_system,
                            particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    search_radius = PointNeighbors.search_radius(neighborhood_search)

    almostzero = eps(search_radius^2)

    backend = semi.parallelization_backend
    # max_particles_per_cell = 64
    # nhs_size = size(neighborhood_search.cell_list.linear_indices)
    # cells = CartesianIndices(ntuple(i -> 2:(nhs_size[i] - 1), ndims(neighborhood_search)))
    linear_indices = neighborhood_search.cell_list.linear_indices
    cartesian_indices = CartesianIndices(size(linear_indices))
    lengths = Array(neighborhood_search.cell_list.cells.lengths)
    max_particles_per_cell = maximum(lengths)
    # max_particles_per_cell = 64
    nonempty_cells = Adapt.adapt(backend, filter(index -> lengths[linear_indices[index]] > 0, cartesian_indices))
    ndrange = max_particles_per_cell * length(nonempty_cells)
    kernel = foreach_neighbor_localmem2(backend, Int64(max_particles_per_cell))
    kernel(dv, system_coords, neighbor_coords, neighborhood_search, nonempty_cells,
           Val(max_particles_per_cell), search_radius, almostzero,
           particle_system, neighbor_system, v_particle_system, v_neighbor_system,
           sound_speed, density_calculator; ndrange)

    KernelAbstractions.synchronize(backend)

    return nothing
end

@inline function copy_to_localmem2!(local_points, local_neighbor_coords,
                                    local_neighbor_vrho, local_neighbor_mass,
                                    local_neighbor_pressure,
                                    v_neighbor_system, neighbor_system,
                                   neighbor_cell, neighbor_system_coords,
                                   neighborhood_search, particleidx, offset)
    points_view = @inbounds PointNeighbors.points_in_cell(neighbor_cell, neighborhood_search)
    n_particles_in_neighbor_cell = length(points_view)

    # First use all threads to load the neighbors into local memory in parallel
    if particleidx <= n_particles_in_neighbor_cell
        write_idx = particleidx + offset
        @inbounds p = local_points[write_idx] = points_view[particleidx]
        for d in 1:ndims(neighborhood_search)
            @inbounds local_neighbor_coords[d, write_idx] = neighbor_system_coords[d, p]
        end
        v, rho = @inbounds velocity_and_density(v_neighbor_system, neighbor_system, p)
        for d in 1:ndims(neighborhood_search)
            @inbounds local_neighbor_vrho[d, write_idx] = v[d]
        end
        @inbounds local_neighbor_vrho[ndims(neighborhood_search) + 1, write_idx] = rho

        m = @inbounds hydrodynamic_mass(neighbor_system, p)
        @inbounds local_neighbor_mass[write_idx] = m
        pressure = @inbounds current_pressure(v_neighbor_system, neighbor_system, p)
        @inbounds local_neighbor_pressure[write_idx] = pressure
    end
    return n_particles_in_neighbor_cell
end

# @parallel(block) for cell in cells
#     for neighbor_cell in neighboring_cells
#         @parallel(thread) for neighbor in neighbor_cell
#             copy_coordinates_to_localmem(neighbor)
#
#         # Make sure all threads finished the copying
#         @synchronize
#
#         @parallel(thread) for particle in cell
#             for neighbor in neighbor_cell
#                 # This uses the neighbor coordinates from the local memory
#                 compute(point, neighbor)
#
#         # Make sure all threads finished computing before we continue with copying
#         @synchronize
@kernel cpu=false function foreach_neighbor_localmem2(dv, system_coords, neighbor_system_coords,
                               neighborhood_search, cells, ::Val{MAX}, search_radius,
                               almostzero, particle_system, neighbor_system,
                               v_particle_system, v_neighbor_system, sound_speed,
                               density_calculator) where {MAX}
    cell_ = @index(Group)
    cell = @inbounds Tuple(cells[cell_])
    particleidx = @index(Local)
    @assert 1 <= particleidx <= MAX

    # Coordinate buffer in local memory
    local_points = @localmem Int32 3 * MAX
    local_neighbor_coords = @localmem eltype(system_coords) (ndims(neighborhood_search), 3 * MAX)
    local_neighbor_vrho = @localmem eltype(system_coords) (ndims(neighborhood_search) + 1, 3 * MAX)
    local_neighbor_mass = @localmem eltype(system_coords) 3 * MAX
    local_neighbor_pressure = @localmem eltype(system_coords) 3 * MAX

    points = @inbounds PointNeighbors.points_in_cell(cell, neighborhood_search)
    n_particles_in_current_cell = length(points)

    # Extract point coordinates if a point lies on this thread
    if particleidx <= n_particles_in_current_cell
        particle = @inbounds points[particleidx]
        point_coords = @inbounds extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                                 particle)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)

        # In 3D, this function can combine velocity and density load into one wide load,
        # which gives a significant speedup on GPUs.
        (v_a,
         rho_a) = @inbounds velocity_and_density(v_particle_system, particle_system,
                                                 particle)
    else
        particle = zero(Int32)
        point_coords = zero(SVector{ndims(neighborhood_search), eltype(system_coords)})

        m_a = zero(eltype(system_coords))
        p_a = zero(eltype(system_coords))
        v_a = zero(SVector{ndims(neighborhood_search), eltype(system_coords)})
        rho_a = zero(eltype(system_coords))
    end

    dv_particle = zero(v_a)
    drho_particle = zero(rho_a)

    for neighbor_cell_block in ((cell[2] - 1, cell[3] - 1), (cell[2], cell[3] - 1), (cell[2] + 1, cell[3] - 1),
                                 (cell[2] - 1, cell[3]),     (cell[2], cell[3]),     (cell[2] + 1, cell[3]),
                                 (cell[2] - 1, cell[3] + 1), (cell[2], cell[3] + 1), (cell[2] + 1, cell[3] + 1))
    # for neighbor_cell_ in PointNeighbors.neighboring_cells(cell, neighborhood_search)
        # neighbor_cell_block = Tuple(neighbor_cell_block_)
        neighbor_cells = ((cell[1] - 1, neighbor_cell_block...), (cell[1], neighbor_cell_block...), (cell[1] + 1, neighbor_cell_block...))

        neighbor_cell = @inbounds neighbor_cells[1]
        n_particles_in_neighbor_cell1 = copy_to_localmem2!(local_points, local_neighbor_coords,
                                                        local_neighbor_vrho, local_neighbor_mass,
                                                        local_neighbor_pressure,
                                                            v_neighbor_system, neighbor_system,
                                                        neighbor_cell, neighbor_system_coords,
                                                        neighborhood_search, particleidx, 0)

        neighbor_cell = @inbounds neighbor_cells[2]
        n_particles_in_neighbor_cell2 = copy_to_localmem2!(local_points, local_neighbor_coords,
                                                        local_neighbor_vrho, local_neighbor_mass,
                                                        local_neighbor_pressure,
                                                            v_neighbor_system, neighbor_system,
                                                        neighbor_cell, neighbor_system_coords,
                                                        neighborhood_search, particleidx, MAX)

        neighbor_cell = @inbounds neighbor_cells[3]
        n_particles_in_neighbor_cell3 = copy_to_localmem2!(local_points, local_neighbor_coords,
                                                        local_neighbor_vrho, local_neighbor_mass,
                                                        local_neighbor_pressure,
                                                            v_neighbor_system, neighbor_system,
                                                        neighbor_cell, neighbor_system_coords,
                                                        neighborhood_search, particleidx, 2 * MAX)

        # Make sure all threads finished the copying
        @synchronize

        # Now each thread works on one particle again
        if particleidx <= n_particles_in_current_cell
            n_particles_in_cells = (n_particles_in_neighbor_cell1, n_particles_in_neighbor_cell2, n_particles_in_neighbor_cell3)
            for neighbor_cell_idx in eachindex(neighbor_cells), local_neighbor in 1:n_particles_in_cells[neighbor_cell_idx]
                local_neighbor = local_neighbor + (neighbor_cell_idx - 1) * MAX
                @inbounds neighbor = local_points[local_neighbor]
                @inbounds neighbor_coords = extract_svector(local_neighbor_coords,
                                                            Val(ndims(neighborhood_search)),
                                                            local_neighbor)

                pos_diff = point_coords - neighbor_coords
                distance2 = dot(pos_diff, pos_diff)

                # TODO periodic

                if almostzero < distance2 <= search_radius^2
                    distance = sqrt(distance2)

                    # Now that we know that `distance` is not zero, we can safely call the unsafe
                    # version of the kernel gradient to avoid redundant zero checks.
                    grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                               distance, particle)

                    m_b = @inbounds local_neighbor_mass[local_neighbor]
                    p_b = @inbounds local_neighbor_pressure[local_neighbor]
                    # v_rho_b = @inbounds extract_svector(local_neighbor_vrho, Val(ndims(neighborhood_search) + 1),
                    #                                     local_neighbor)
                    # v_b = v_rho_b[1:ndims(neighborhood_search)]
                    # v_b = @inbounds SVector(v_rho_b[1], v_rho_b[2], v_rho_b[3])
                    # rho_b = @inbounds v_rho_b[ndims(neighborhood_search) + 1]
                    (v_b, rho_b) = @inbounds velocity_and_density(local_neighbor_vrho, neighbor_system,
                                                                  local_neighbor)

                    # `foreach_neighbor` makes sure that `neighbor` is in bounds of `neighbor_system`
                    # m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
                    # (v_b,
                    # rho_b) = @inbounds velocity_and_density(v_neighbor_system, neighbor_system,
                    #                                         neighbor)
                    # v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)
                    # rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
                    vdiff = v_a - v_b

                    # The following call is equivalent to
                    #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
                    # Only when the neighbor system is a `WallBoundarySystem`
                    # or a `TotalLagrangianSPHSystem` with the boundary model `PressureMirroring`,
                    # this will return `p_b = p_a`, which is the pressure of the fluid particle.
                    # p_b = @inbounds neighbor_pressure(v_neighbor_system, neighbor_system,
                    #                                 neighbor, p_a)

                    # For `ContinuityDensity` without correction, this is equivalent to
                    # dv_pressure = -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
                    dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                                        particle, neighbor,
                                                        m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                        distance, grad_kernel, nothing)

                    # Accumulate contributions over all neighbors
                    dv_particle += dv_pressure

                    # Propagate `@inbounds` to the viscosity function, which accesses particle data
                    dv_viscosity_ = Ref(zero(dv_particle))
                    @inbounds dv_viscosity(dv_viscosity_,
                                        particle_system, neighbor_system,
                                        v_particle_system, v_neighbor_system,
                                        particle, neighbor, pos_diff, distance,
                                        sound_speed, m_a, m_b, rho_a, rho_b,
                                        v_a, v_b, grad_kernel)
                    dv_particle += dv_viscosity_[]

                    # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
                    # Propagate `@inbounds` to the continuity equation, which accesses particle data
                    drho_ = Ref(zero(drho_particle))
                    @inbounds continuity_equation!(drho_, density_calculator,
                                                particle_system, neighbor_system,
                                                v_particle_system, v_neighbor_system,
                                                particle, neighbor, pos_diff, distance,
                                                m_b, rho_a, rho_b, vdiff, grad_kernel)
                    drho_particle += drho_[]
                end
            end
        end

        # Make sure all threads finished computing before we continue with copying
        @synchronize()
    end

    if particleidx <= n_particles_in_current_cell
        for i in eachindex(dv_particle)
            @inbounds dv[i, particle] += dv_particle[i]
        end
        @inbounds dv[end, particle] += drho_particle
    end
end

# Double-buffered version of localmem2
# 7.89 ms
function interact_localmem3!(dv, v_particle_system, u_particle_system,
                            v_neighbor_system, u_neighbor_system,
                            particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    search_radius = PointNeighbors.search_radius(neighborhood_search)

    almostzero = eps(search_radius^2)

    backend = semi.parallelization_backend
    # max_particles_per_cell = 64
    # nhs_size = size(neighborhood_search.cell_list.linear_indices)
    # cells = CartesianIndices(ntuple(i -> 2:(nhs_size[i] - 1), ndims(neighborhood_search)))
    linear_indices = neighborhood_search.cell_list.linear_indices
    cartesian_indices = CartesianIndices(size(linear_indices))
    lengths = Array(neighborhood_search.cell_list.cells.lengths)
    max_particles_per_cell = maximum(lengths)
    # max_particles_per_cell = 64
    nonempty_cells = Adapt.adapt(backend, filter(index -> lengths[linear_indices[index]] > 0, cartesian_indices))
    ndrange = max_particles_per_cell * length(nonempty_cells)
    kernel = foreach_neighbor_localmem3(backend, Int64(max_particles_per_cell))
    kernel(dv, system_coords, neighbor_coords, neighborhood_search, nonempty_cells,
           Val(max_particles_per_cell), search_radius, almostzero,
           particle_system, neighbor_system, v_particle_system, v_neighbor_system,
           sound_speed, density_calculator; ndrange)

    KernelAbstractions.synchronize(backend)

    return nothing
end

# @parallel(block) for cell in cells
#     for neighbor_cell in neighboring_cells
#         @parallel(thread) for neighbor in neighbor_cell
#             copy_coordinates_to_localmem(neighbor)
#
#         # Make sure all threads finished the copying
#         @synchronize
#
#         @parallel(thread) for particle in cell
#             for neighbor in neighbor_cell
#                 # This uses the neighbor coordinates from the local memory
#                 compute(point, neighbor)
#
#         # Make sure all threads finished computing before we continue with copying
#         @synchronize
@kernel cpu=false function foreach_neighbor_localmem3(dv, system_coords, neighbor_system_coords,
                               neighborhood_search, cells, ::Val{MAX}, search_radius,
                               almostzero, particle_system, neighbor_system,
                               v_particle_system, v_neighbor_system, sound_speed,
                               density_calculator) where {MAX}
    cell_ = @index(Group)
    cell = @inbounds Tuple(cells[cell_])
    particleidx = @index(Local)
    @assert 1 <= particleidx <= MAX

    # Coordinate buffer in local memory
    local_points = @localmem Int32 3 * MAX
    local_neighbor_coords = @localmem eltype(system_coords) (ndims(neighborhood_search), 3 * MAX)
    local_neighbor_vrho = @localmem eltype(system_coords) (ndims(neighborhood_search) + 1, 3 * MAX)
    local_neighbor_mass = @localmem eltype(system_coords) 3 * MAX
    local_neighbor_pressure = @localmem eltype(system_coords) 3 * MAX

    next_local_points = @localmem Int32 3 * MAX
    next_local_neighbor_coords = @localmem eltype(system_coords) (ndims(neighborhood_search), 3 * MAX)
    next_local_neighbor_vrho = @localmem eltype(system_coords) (ndims(neighborhood_search) + 1, 3 * MAX)
    next_local_neighbor_mass = @localmem eltype(system_coords) 3 * MAX
    next_local_neighbor_pressure = @localmem eltype(system_coords) 3 * MAX

    points = @inbounds PointNeighbors.points_in_cell(cell, neighborhood_search)
    n_particles_in_current_cell = length(points)

    # Extract point coordinates if a point lies on this thread
    if particleidx <= n_particles_in_current_cell
        particle = @inbounds points[particleidx]
        point_coords = @inbounds extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                                 particle)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)

        # In 3D, this function can combine velocity and density load into one wide load,
        # which gives a significant speedup on GPUs.
        (v_a,
         rho_a) = @inbounds velocity_and_density(v_particle_system, particle_system,
                                                 particle)
    else
        particle = zero(Int32)
        point_coords = zero(SVector{ndims(neighborhood_search), eltype(system_coords)})

        m_a = zero(eltype(system_coords))
        p_a = zero(eltype(system_coords))
        v_a = zero(SVector{ndims(neighborhood_search), eltype(system_coords)})
        rho_a = zero(eltype(system_coords))
    end

    dv_particle = zero(v_a)
    drho_particle = zero(rho_a)

    # Load first cell block into local memory
    neighbor_cell_block = @inbounds (cell[2] - 1, cell[3] - 1)
    neighbor_cells = @inbounds ((cell[1] - 1, neighbor_cell_block...), (cell[1], neighbor_cell_block...), (cell[1] + 1, neighbor_cell_block...))

    neighbor_cell = @inbounds neighbor_cells[1]
    n_particles_in_neighbor_cell1 = copy_to_localmem2!(local_points, local_neighbor_coords,
                                                    local_neighbor_vrho, local_neighbor_mass,
                                                    local_neighbor_pressure,
                                                        v_neighbor_system, neighbor_system,
                                                    neighbor_cell, neighbor_system_coords,
                                                    neighborhood_search, particleidx, 0)

    neighbor_cell = @inbounds neighbor_cells[2]
    n_particles_in_neighbor_cell2 = copy_to_localmem2!(local_points, local_neighbor_coords,
                                                    local_neighbor_vrho, local_neighbor_mass,
                                                    local_neighbor_pressure,
                                                        v_neighbor_system, neighbor_system,
                                                    neighbor_cell, neighbor_system_coords,
                                                    neighborhood_search, particleidx, MAX)

    neighbor_cell = @inbounds neighbor_cells[3]
    n_particles_in_neighbor_cell3 = copy_to_localmem2!(local_points, local_neighbor_coords,
                                                    local_neighbor_vrho, local_neighbor_mass,
                                                    local_neighbor_pressure,
                                                        v_neighbor_system, neighbor_system,
                                                    neighbor_cell, neighbor_system_coords,
                                                    neighborhood_search, particleidx, 2 * MAX)

    n_particles_in_cells = (n_particles_in_neighbor_cell1, n_particles_in_neighbor_cell2, n_particles_in_neighbor_cell3)

    @synchronize

    neighbor_cell_blocks = @inbounds ((cell[2] - 1, cell[3] - 1), (cell[2], cell[3] - 1), (cell[2] + 1, cell[3] - 1),
                                 (cell[2] - 1, cell[3]),     (cell[2], cell[3]),     (cell[2] + 1, cell[3]),
                                 (cell[2] - 1, cell[3] + 1), (cell[2], cell[3] + 1), (cell[2] + 1, cell[3] + 1))
    for neighbor_cell_block_idx in eachindex(neighbor_cell_blocks)
        neighbor_cell_block = @inbounds neighbor_cell_blocks[neighbor_cell_block_idx]
        neighbor_cells = @inbounds ((cell[1] - 1, neighbor_cell_block...), (cell[1], neighbor_cell_block...), (cell[1] + 1, neighbor_cell_block...))

        if neighbor_cell_block_idx < length(neighbor_cell_blocks)
            neighbor_cell = @inbounds neighbor_cells[1]
            n_particles_in_neighbor_cell1 = copy_to_localmem2!(local_points, next_local_neighbor_coords,
                                                            next_local_neighbor_vrho, next_local_neighbor_mass,
                                                            next_local_neighbor_pressure,
                                                                v_neighbor_system, neighbor_system,
                                                            neighbor_cell, neighbor_system_coords,
                                                            neighborhood_search, particleidx, 0)

            neighbor_cell = @inbounds neighbor_cells[2]
            n_particles_in_neighbor_cell2 = copy_to_localmem2!(next_local_points, next_local_neighbor_coords,
                                                            next_local_neighbor_vrho, next_local_neighbor_mass,
                                                            next_local_neighbor_pressure,
                                                                v_neighbor_system, neighbor_system,
                                                            neighbor_cell, neighbor_system_coords,
                                                            neighborhood_search, particleidx, MAX)

            neighbor_cell = @inbounds neighbor_cells[3]
            n_particles_in_neighbor_cell3 = copy_to_localmem2!(next_local_points, next_local_neighbor_coords,
                                                            next_local_neighbor_vrho, next_local_neighbor_mass,
                                                            next_local_neighbor_pressure,
                                                                v_neighbor_system, neighbor_system,
                                                            neighbor_cell, neighbor_system_coords,
                                                            neighborhood_search, particleidx, 2 * MAX)

            next_n_particles_in_cells = (n_particles_in_neighbor_cell1, n_particles_in_neighbor_cell2, n_particles_in_neighbor_cell3)
        end

        # Now each thread works on one particle again
        if particleidx <= n_particles_in_current_cell
            for neighbor_cell_idx in eachindex(neighbor_cells), local_neighbor in 1:n_particles_in_cells[neighbor_cell_idx]
                local_neighbor = local_neighbor + (neighbor_cell_idx - 1) * MAX
                @inbounds neighbor = local_points[local_neighbor]
                @inbounds neighbor_coords = extract_svector(local_neighbor_coords,
                                                            Val(ndims(neighborhood_search)),
                                                            local_neighbor)

                pos_diff = point_coords - neighbor_coords
                distance2 = dot(pos_diff, pos_diff)

                # TODO periodic

                if almostzero < distance2 <= search_radius^2
                    distance = sqrt(distance2)

                    # Now that we know that `distance` is not zero, we can safely call the unsafe
                    # version of the kernel gradient to avoid redundant zero checks.
                    grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                               distance, particle)

                    m_b = @inbounds local_neighbor_mass[local_neighbor]
                    p_b = @inbounds local_neighbor_pressure[local_neighbor]
                    # v_rho_b = @inbounds extract_svector(local_neighbor_vrho, Val(ndims(neighborhood_search) + 1),
                    #                                     local_neighbor)
                    # v_b = v_rho_b[1:ndims(neighborhood_search)]
                    # v_b = @inbounds SVector(v_rho_b[1], v_rho_b[2], v_rho_b[3])
                    # rho_b = @inbounds v_rho_b[ndims(neighborhood_search) + 1]
                    (v_b, rho_b) = @inbounds velocity_and_density(local_neighbor_vrho, neighbor_system,
                                                                  local_neighbor)

                    # `foreach_neighbor` makes sure that `neighbor` is in bounds of `neighbor_system`
                    # m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
                    # (v_b,
                    # rho_b) = @inbounds velocity_and_density(v_neighbor_system, neighbor_system,
                    #                                         neighbor)
                    # v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)
                    # rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
                    vdiff = v_a - v_b

                    # The following call is equivalent to
                    #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
                    # Only when the neighbor system is a `WallBoundarySystem`
                    # or a `TotalLagrangianSPHSystem` with the boundary model `PressureMirroring`,
                    # this will return `p_b = p_a`, which is the pressure of the fluid particle.
                    # p_b = @inbounds neighbor_pressure(v_neighbor_system, neighbor_system,
                    #                                 neighbor, p_a)

                    # For `ContinuityDensity` without correction, this is equivalent to
                    # dv_pressure = -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
                    dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                                        particle, neighbor,
                                                        m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                        distance, grad_kernel, nothing)

                    # Accumulate contributions over all neighbors
                    dv_particle += dv_pressure

                    # Propagate `@inbounds` to the viscosity function, which accesses particle data
                    dv_viscosity_ = Ref(zero(dv_particle))
                    @inbounds dv_viscosity(dv_viscosity_,
                                        particle_system, neighbor_system,
                                        v_particle_system, v_neighbor_system,
                                        particle, neighbor, pos_diff, distance,
                                        sound_speed, m_a, m_b, rho_a, rho_b,
                                        v_a, v_b, grad_kernel)
                    dv_particle += dv_viscosity_[]

                    # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
                    # Propagate `@inbounds` to the continuity equation, which accesses particle data
                    drho_ = Ref(zero(drho_particle))
                    @inbounds continuity_equation!(drho_, density_calculator,
                                                particle_system, neighbor_system,
                                                v_particle_system, v_neighbor_system,
                                                particle, neighbor, pos_diff, distance,
                                                m_b, rho_a, rho_b, vdiff, grad_kernel)
                    drho_particle += drho_[]
                end
            end
        end

        # Make sure all threads finished computing before we continue with copying
        @synchronize()

        neighbor_cell_block_idx >= length(neighbor_cell_blocks) && break

        # Swap local memory buffers
        n_particles_in_cells = next_n_particles_in_cells
        local_points, next_local_points = next_local_points, local_points
        local_neighbor_coords, next_local_neighbor_coords = next_local_neighbor_coords, local_neighbor_coords
        local_neighbor_vrho, next_local_neighbor_vrho = next_local_neighbor_vrho, local_neighbor_vrho
        local_neighbor_mass, next_local_neighbor_mass = next_local_neighbor_mass, local_neighbor_mass
        local_neighbor_pressure, next_local_neighbor_pressure = next_local_neighbor_pressure, local_neighbor_pressure
    end

    if particleidx <= n_particles_in_current_cell
        for i in eachindex(dv_particle)
            @inbounds dv[i, particle] += dv_particle[i]
        end
        @inbounds dv[end, particle] += drho_particle
    end
end


# Same as localmem2, but local memory is used for 27 cells at once to reduce the number of synchronizations.
# 6.73 ms
function interact_localmem4!(dv, v_particle_system, u_particle_system,
                            v_neighbor_system, u_neighbor_system,
                            particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    search_radius = PointNeighbors.search_radius(neighborhood_search)

    almostzero = eps(search_radius^2)

    backend = semi.parallelization_backend
    # max_particles_per_cell = 64
    # nhs_size = size(neighborhood_search.cell_list.linear_indices)
    # cells = CartesianIndices(ntuple(i -> 2:(nhs_size[i] - 1), ndims(neighborhood_search)))
    linear_indices = neighborhood_search.cell_list.linear_indices
    cartesian_indices = CartesianIndices(size(linear_indices))
    lengths = Array(neighborhood_search.cell_list.cells.lengths)
    max_particles_per_cell = maximum(lengths)
    # max_particles_per_cell = 64
    nonempty_cells = Adapt.adapt(backend, filter(index -> lengths[linear_indices[index]] > 0, cartesian_indices))
    ndrange = max_particles_per_cell * length(nonempty_cells)
    kernel = foreach_neighbor_localmem4(backend, Int64(max_particles_per_cell))
    kernel(dv, system_coords, neighbor_coords, neighborhood_search, nonempty_cells,
           Val(max_particles_per_cell), search_radius, almostzero,
           particle_system, neighbor_system, v_particle_system, v_neighbor_system,
           sound_speed, density_calculator; ndrange)

    KernelAbstractions.synchronize(backend)

    return nothing
end

# @parallel(block) for cell in cells
#     for neighbor_cell in neighboring_cells
#         @parallel(thread) for neighbor in neighbor_cell
#             copy_coordinates_to_localmem(neighbor)
#
#         # Make sure all threads finished the copying
#         @synchronize
#
#         @parallel(thread) for particle in cell
#             for neighbor in neighbor_cell
#                 # This uses the neighbor coordinates from the local memory
#                 compute(point, neighbor)
#
#         # Make sure all threads finished computing before we continue with copying
#         @synchronize
@kernel cpu=false function foreach_neighbor_localmem4(dv, system_coords, neighbor_system_coords,
                               neighborhood_search, cells, ::Val{MAX}, search_radius,
                               almostzero, particle_system, neighbor_system,
                               v_particle_system, v_neighbor_system, sound_speed,
                               density_calculator) where {MAX}
    cell_ = @index(Group)
    cell = @inbounds Tuple(cells[cell_])
    particleidx = @index(Local)
    @assert 1 <= particleidx <= MAX

    # Coordinate buffer in local memory
    local_points = @localmem Int32 27 * MAX
    local_neighbor_coords = @localmem eltype(system_coords) (ndims(neighborhood_search), 27 * MAX)
    local_neighbor_vrho = @localmem eltype(system_coords) (ndims(neighborhood_search) + 1, 27 * MAX)
    local_neighbor_mass = @localmem eltype(system_coords) 27 * MAX
    local_neighbor_pressure = @localmem eltype(system_coords) 27 * MAX
    local_n_particles_in_neighbor_cell = @localmem Int32 27

    points = @inbounds PointNeighbors.points_in_cell(cell, neighborhood_search)
    n_particles_in_current_cell = length(points)

    # Extract point coordinates if a point lies on this thread
    if particleidx <= n_particles_in_current_cell
        particle = @inbounds points[particleidx]
        point_coords = @inbounds extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                                 particle)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)

        # In 3D, this function can combine velocity and density load into one wide load,
        # which gives a significant speedup on GPUs.
        (v_a,
         rho_a) = @inbounds velocity_and_density(v_particle_system, particle_system,
                                                 particle)
    else
        particle = zero(Int32)
        point_coords = zero(SVector{ndims(neighborhood_search), eltype(system_coords)})

        m_a = zero(eltype(system_coords))
        p_a = zero(eltype(system_coords))
        v_a = zero(SVector{ndims(neighborhood_search), eltype(system_coords)})
        rho_a = zero(eltype(system_coords))
    end

    dv_particle = zero(v_a)
    drho_particle = zero(rho_a)

    neighboring_cells = PointNeighbors.neighboring_cells(cell, neighborhood_search)
    for neighbor_cell_idx in 1:length(neighboring_cells)
        neighbor_cell = @inbounds Tuple(neighboring_cells[neighbor_cell_idx])
        n_particles_in_neighbor_cell = copy_to_localmem2!(local_points, local_neighbor_coords,
                                                        local_neighbor_vrho, local_neighbor_mass,
                                                        local_neighbor_pressure,
                                                        v_neighbor_system, neighbor_system,
                                                        neighbor_cell, neighbor_system_coords,
                                                        neighborhood_search, particleidx, (neighbor_cell_idx - 1) * MAX)
        @inbounds local_n_particles_in_neighbor_cell[neighbor_cell_idx] = n_particles_in_neighbor_cell
    end

    # Make sure all threads finished the copying
    @synchronize

    # Now each thread works on one particle again
    if particleidx <= n_particles_in_current_cell
        for neighbor_cell_idx in 1:length(neighboring_cells)
            n_cells = @inbounds local_n_particles_in_neighbor_cell[neighbor_cell_idx]
            for local_neighbor in 1:n_cells
                local_neighbor = local_neighbor + (neighbor_cell_idx - 1) * MAX
                @inbounds neighbor = local_points[local_neighbor]
                @inbounds neighbor_coords = extract_svector(local_neighbor_coords,
                                                            Val(ndims(neighborhood_search)),
                                                            local_neighbor)

                pos_diff = point_coords - neighbor_coords
                distance2 = dot(pos_diff, pos_diff)

                # TODO periodic

                if almostzero < distance2 <= search_radius^2
                    distance = sqrt(distance2)

                    # Now that we know that `distance` is not zero, we can safely call the unsafe
                    # version of the kernel gradient to avoid redundant zero checks.
                    grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                               distance, particle)

                    m_b = @inbounds local_neighbor_mass[local_neighbor]
                    p_b = @inbounds local_neighbor_pressure[local_neighbor]
                    # v_rho_b = @inbounds extract_svector(local_neighbor_vrho, Val(ndims(neighborhood_search) + 1),
                    #                                     local_neighbor)
                    # v_b = v_rho_b[1:ndims(neighborhood_search)]
                    # v_b = @inbounds SVector(v_rho_b[1], v_rho_b[2], v_rho_b[3])
                    # rho_b = @inbounds v_rho_b[ndims(neighborhood_search) + 1]
                    (v_b, rho_b) = @inbounds velocity_and_density(local_neighbor_vrho, neighbor_system,
                                                                  local_neighbor)

                    # `foreach_neighbor` makes sure that `neighbor` is in bounds of `neighbor_system`
                    # m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
                    # (v_b,
                    # rho_b) = @inbounds velocity_and_density(v_neighbor_system, neighbor_system,
                    #                                         neighbor)
                    # v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)
                    # rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
                    vdiff = v_a - v_b

                    # The following call is equivalent to
                    #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
                    # Only when the neighbor system is a `WallBoundarySystem`
                    # or a `TotalLagrangianSPHSystem` with the boundary model `PressureMirroring`,
                    # this will return `p_b = p_a`, which is the pressure of the fluid particle.
                    # p_b = @inbounds neighbor_pressure(v_neighbor_system, neighbor_system,
                    #                                 neighbor, p_a)

                    # For `ContinuityDensity` without correction, this is equivalent to
                    # dv_pressure = -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
                    dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                                        particle, neighbor,
                                                        m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                        distance, grad_kernel, nothing)

                    # Accumulate contributions over all neighbors
                    dv_particle += dv_pressure

                    # Propagate `@inbounds` to the viscosity function, which accesses particle data
                    dv_viscosity_ = Ref(zero(dv_particle))
                    @inbounds dv_viscosity(dv_viscosity_,
                                        particle_system, neighbor_system,
                                        v_particle_system, v_neighbor_system,
                                        particle, neighbor, pos_diff, distance,
                                        sound_speed, m_a, m_b, rho_a, rho_b,
                                        v_a, v_b, grad_kernel)
                    dv_particle += dv_viscosity_[]

                    # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
                    # Propagate `@inbounds` to the continuity equation, which accesses particle data
                    drho_ = Ref(zero(drho_particle))
                    @inbounds continuity_equation!(drho_, density_calculator,
                                                particle_system, neighbor_system,
                                                v_particle_system, v_neighbor_system,
                                                particle, neighbor, pos_diff, distance,
                                                m_b, rho_a, rho_b, vdiff, grad_kernel)
                    drho_particle += drho_[]
                end
            end
        end
    end

    if particleidx <= n_particles_in_current_cell
        for i in eachindex(dv_particle)
            @inbounds dv[i, particle] += dv_particle[i]
        end
        @inbounds dv[end, particle] += drho_particle
    end
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

@propagate_inbounds function velocity_and_density(v, system, particle)
    # For other systems, fall back to the default implementation
    return velocity_and_density(v, nothing, system, particle)
end

@propagate_inbounds function velocity_and_density(v, system::WeaklyCompressibleSPHSystem,
                                                  particle)
    (; density_calculator) = system

    return velocity_and_density(v, density_calculator, system, particle)
end

@propagate_inbounds function velocity_and_density(v, _, system, particle)
    v_particle = current_velocity(v, system, particle)
    rho_particle = current_density(v, system, particle)

    return v_particle, rho_particle
end

@inline function velocity_and_density(v, ::ContinuityDensity,
                                      ::WeaklyCompressibleSPHSystem{3}, particle)
    vrho_a = vloada(Vec{4, eltype(v)}, pointer(v, 4 * (particle - 1) + 1))
    a, b, c, d = Tuple(vrho_a)
    v_particle = SVector(a, b, c)
    rho_particle = d

    return v_particle, rho_particle
end
