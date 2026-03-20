# Computes the forces that particles in `particle_system` experience from particles
# in `neighbor_system` and updates `dv` accordingly.
# It takes into account pressure forces, viscosity, and for `ContinuityDensity` updates
# the density using the continuity equation.
function interact_old!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    (; density_calculator, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # In order to visualize quantities like pressure forces or viscosity forces, uncomment
    # the following code and the two other lines below that are marked as "debug example".
    # debug_array = zeros(ndims(particle_system), nparticles(particle_system))

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_system_coords, semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
        # `foreach_point_neighbor` makes sure that `particle` and `neighbor` are
        # in bounds of the respective system. For performance reasons, we use `@inbounds`
        # in this hot loop to avoid bounds checking when extracting particle quantities.
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
        rho_mean = (rho_a + rho_b) / 2

        # Determine correction factors.
        # This can be ignored, as these are all 1 when no correction is used.
        (viscosity_correction, pressure_correction,
         surface_tension_correction) = free_surface_correction(correction, particle_system,
                                                               rho_mean)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        # The following call is equivalent to
        #     `p_a = current_pressure(v_particle_system, particle_system, particle)`
        #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
        # Only when the neighbor system is a `WallBoundarySystem` or a `TotalLagrangianSPHSystem`
        # with the boundary model `PressureMirroring`, this will return `p_b = p_a`, which is
        # the pressure of the fluid particle.
        p_a,
        p_b = @inbounds particle_neighbor_pressure(v_particle_system,
                                                   v_neighbor_system,
                                                   particle_system, neighbor_system,
                                                   particle, neighbor)

        dv_pressure = pressure_correction *
                      pressure_acceleration(particle_system, neighbor_system,
                                            particle, neighbor,
                                            m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                            distance, grad_kernel, correction)

        # Propagate `@inbounds` to the viscosity function, which accesses particle data
        dv_viscosity_ = zero(pos_diff)
        # viscosity_correction *
        #                 @inbounds dv_viscosity(particle_system, neighbor_system,
        #                                        v_particle_system, v_neighbor_system,
        #                                        particle, neighbor, pos_diff, distance,
        #                                        sound_speed, m_a, m_b, rho_a, rho_b,
        #                                        grad_kernel)

        # Extra terms in the momentum equation when using a shifting technique
        dv_tvf = @inbounds dv_shifting(shifting_technique(particle_system),
                                       particle_system, neighbor_system,
                                       v_particle_system, v_neighbor_system,
                                       particle, neighbor, m_a, m_b, rho_a, rho_b,
                                       pos_diff, distance, grad_kernel, correction)

        dv_surface_tension = surface_tension_correction *
                             surface_tension_force(surface_tension_a, surface_tension_b,
                                                   particle_system, neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   rho_a, rho_b, grad_kernel)

        dv_adhesion = adhesion_force(surface_tension_a, particle_system, neighbor_system,
                                     particle, neighbor, pos_diff, distance)

        dv_particle = dv_pressure + dv_viscosity_ + dv_tvf + dv_surface_tension +
                      dv_adhesion

        for i in 1:ndims(particle_system)
            @inbounds dv[i, particle] += dv_particle[i]
            # Debug example
            # debug_array[i, particle] += dv_pressure[i]
        end

        # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
        # Propagate `@inbounds` to the continuity equation, which accesses particle data
        @inbounds continuity_equation!(dv, density_calculator, particle_system,
                                       neighbor_system, v_particle_system,
                                       v_neighbor_system, particle, neighbor,
                                       pos_diff, distance, m_b, rho_a, rho_b, grad_kernel)
    end
    # Debug example
    # periodic_box = neighborhood_search.periodic_box
    # Note: this saves a file in every stage of the integrator
    # if !@isdefined iter; iter = 0; end
    # TODO: This call should use public API. This requires some additional changes to simplify the calls.
    # trixi2vtk(v_particle_system, u_particle_system, -1.0, particle_system, periodic_box, debug=debug_array, prefix="debug", iter=iter += 1)

    return dv
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    interact_old!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system, neighbor_system, semi)
end

function interact2!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem,
                   neighbor_system::WeaklyCompressibleSPHSystem, semi)
    dv_ = view(dv, 1:ndims(particle_system), :)
    drho = view(dv, ndims(particle_system) + 1, :)
    interact!(dv_, drho, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system, neighbor_system, semi)
end

function interact!(dv, drho, v_particle_system, u_particle_system,
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

    backend = semi.parallelization_backend
    ndrange = n_integrated_particles(particle_system)
    mykernel(backend)(dv, drho, system_coords, neighbor_system_coords, neighborhood_search,
                      cell_list, search_radius2, v_particle_system, v_neighbor_system,
                      particle_system, neighbor_system; ndrange=ndrange)

    KernelAbstractions.synchronize(backend)

    return dv
end

@kernel function mykernel(dv, drho,
                          system_coords, neighbor_system_coords,
                          nhs, cell_list, search_radius2,
                          v_particle_system, v_neighbor_system,
                          particle_system::WeaklyCompressibleSPHSystem{NDIMS},
                          neighbor_system::WeaklyCompressibleSPHSystem) where NDIMS
    particle = @index(Global)

    sound_speed = particle_system.state_equation.sound_speed
    # VT_coords = Vec{4, eltype(system_coords)}
    # point_coords_ = vloada(VT_coords, pointer(system_coords, 4*(particle-1)+1))
    # a, b, c, d = Tuple(point_coords_)
    # point_coords = SVector(a, b, c)
    point_coords = @inbounds extract_svector(system_coords, Val(NDIMS), particle)
    p_a = @inbounds particle_system.pressure[particle]

    VT = Vec{4, eltype(v_particle_system)}
    vrho_a = vloada(VT, pointer(v_particle_system, 4*(particle-1)+1))
    a, b, c, d = Tuple(vrho_a)
    v_a = SVector(a, b, c)
    rho_a = d
    # v_a = @inbounds extract_svector(v_particle_system, Val(NDIMS), particle)
    # rho_a = @inbounds v_particle_system[end, particle]

    dv_particle = zero(v_a)
    drho_particle = zero(rho_a)

    cell = PointNeighbors.cell_coords(point_coords, nhs)

    # cell_blocks = ((cell[1] - 1, cell[2] - 1), (cell[1] - 1, cell[2]), (cell[1] - 1, cell[2] + 1))
    cell_blocks = CartesianIndices(ntuple(i -> (cell[i + 1] - 1):(cell[i + 1] + 1), Val(NDIMS - 1)))
    for cell_block in cell_blocks
        cell_block_start = (cell[1] - 1, Tuple(cell_block)...)
        cell_index = @inbounds PointNeighbors.cell_index(cell_list, cell_block_start)
        start = @inbounds cell_list.cells.first_bin_index[cell_index]
        stop = @inbounds cell_list.cells.first_bin_index[cell_index + 3] - 1

        for neighbor in start:stop

    # for neighbor_cell_ in PointNeighbors.neighboring_cells(cell, nhs)
    #     neighbor_cell = Tuple(neighbor_cell_)
    #     neighbors = @inbounds PointNeighbors.points_in_cell(neighbor_cell, nhs)

    #     for neighbor_ in eachindex(neighbors)
    #         neighbor = @inbounds neighbors[neighbor_]

            # neighbor_coords_ = vloada(VT_coords, pointer(neighbor_system_coords, 4*(neighbor-1)+1))
            # a, b, c, d = Tuple(neighbor_coords_)
            # neighbor_coords = SVector(a, b, c)
            neighbor_coords = @inbounds extract_svector(neighbor_system_coords,
                                                        Val(NDIMS), neighbor)

            # pos_diff = convert.(eltype(particle_system), point_coords - neighbor_coords)
            pos_diff = point_coords - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if eps(search_radius2) <= distance2 <= search_radius2
                distance = sqrt(distance2)

                m_b = @inbounds neighbor_system.mass[neighbor]
                p_b = @inbounds neighbor_system.pressure[neighbor]

                vrho_b = vloada(VT, pointer(v_neighbor_system, 4*(neighbor-1)+1))
                a, b, c, d = Tuple(vrho_b)
                v_b = SVector(a, b, c)
                rho_b = d

                # v_b = @inbounds extract_svector(v_neighbor_system, Val(NDIMS), neighbor)
                # rho_b = @inbounds v_neighbor_system[end, neighbor]

                grad_kernel = kernel_grad_ds(particle_system, pos_diff, distance)

                # dv_particle += -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
                dv_particle += -m_b * Base.FastMath.div_fast(p_a + p_b, rho_a * rho_b) * grad_kernel

                vdiff = v_a - v_b
                # drho_particle += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)
                drho_particle += Base.FastMath.div_fast(rho_a, rho_b) * m_b * dot(vdiff, grad_kernel)

                h = particle_system.cache.smoothing_length
                alpha = particle_system.viscosity.alpha
                epsilon = particle_system.viscosity.epsilon

                vr = dot(vdiff, pos_diff)
                if vr < 0
                    # mu = h * vr / (distance2 + epsilon)
                    mu = Base.FastMath.div_fast(h * vr, distance2 + epsilon)
                    rho_mean = (rho_a + rho_b) / 2
                    # @fastmath pi_ab = (alpha * sound_speed * mu) / rho_mean * grad_kernel
                    pi_ab = Base.FastMath.div_fast(alpha * sound_speed * mu, rho_mean) * grad_kernel
                    dv_particle += m_b * pi_ab
                end
            end
        end
    end

    for i in eachindex(dv_particle)
        @inbounds dv[i, particle] += dv_particle[i]
        # Debug example
        # debug_array[i, particle] += dv_pressure[i]
    end
    @inbounds drho[particle] += drho_particle
end

@inline function kernel_grad_ds(system, pos_diff, r)
    h = system.cache.smoothing_length
    normalization_factor = system.cache.normalization_factor

    # q = r / h
    q = Base.FastMath.div_fast(r, h)
    wqq1 = (1 - q / 2)
    return normalization_factor * wqq1 * wqq1 * wqq1 * pos_diff
end

function interact_reassembled!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem{NDIMS},
                   neighbor_system, semi) where NDIMS
    (; density_calculator, correction) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    sound_speed = particle_system.state_equation.sound_speed

    @threaded semi for particle in each_integrated_particle(particle_system)
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)

        # v_a = @inbounds extract_svector(v_particle_system, Val(NDIMS), particle)
        # rho_a = @inbounds v_particle_system[end, particle]
        v_a, rho_a = @inbounds velocity_and_density(v_particle_system, particle_system, particle)

        dv_particle = Ref(zero(v_a))
        drho_particle = Ref(zero(rho_a))

        PointNeighbors.foreach_neighbor(system_coords, neighbor_system_coords,
                         neighborhood_search, particle) do _, neighbor, pos_diff, distance
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)

            # v_b = @inbounds extract_svector(v_neighbor_system, Val(NDIMS), neighbor)
            # rho_b = @inbounds v_neighbor_system[end, neighbor]
            v_b, rho_b = @inbounds velocity_and_density(v_neighbor_system, neighbor_system, neighbor)

            # grad_kernel = kernel_grad_ds(particle_system, pos_diff, distance)
            grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

            # dv_particle += -m_b * (p_a + p_b) / (rho_a * rho_b) * grad_kernel
            # dv_particle[] += -m_b * Base.FastMath.div_fast(p_a + p_b, rho_a * rho_b) * grad_kernel

            dv_particle[] += pressure_acceleration(particle_system, neighbor_system,
                                                   particle, neighbor,
                                                   m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                   distance, grad_kernel, correction)

            vdiff = v_a - v_b
            # drho_particle += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)
            # drho_particle[] += Base.FastMath.div_fast(rho_a, rho_b) * m_b * dot(vdiff, grad_kernel)

            @inbounds dv_viscosity(dv_particle, particle_system, neighbor_system,
                                               vdiff,
                                               particle, neighbor, pos_diff, distance,
                                               sound_speed, m_a, m_b, rho_a, rho_b,
                                               grad_kernel)

            @inbounds continuity_equation!(drho_particle, density_calculator, particle_system,
                                       neighbor_system, particle, neighbor,
                                       pos_diff, distance, vdiff, m_b, rho_a, rho_b, grad_kernel)

            # h = particle_system.cache.smoothing_length
            # alpha = particle_system.viscosity.alpha
            # epsilon = particle_system.viscosity.epsilon

            # vr = dot(vdiff, pos_diff)
            # if vr < 0
            #     # mu = h * vr / (distance2 + epsilon)
            #     mu = Base.FastMath.div_fast(h * vr, distance^2 + epsilon)
            #     rho_mean = (rho_a + rho_b) / 2
            #     # @fastmath pi_ab = (alpha * sound_speed * mu) / rho_mean * grad_kernel
            #     pi_ab = Base.FastMath.div_fast(alpha * sound_speed * mu, rho_mean) * grad_kernel
            #     dv_particle[] += m_b * pi_ab
            # end
        end

        for i in eachindex(dv_particle[])
            @inbounds dv[i, particle] += dv_particle[][i]
        end
        @inbounds dv[end, particle] += drho_particle[]
    end
end

@propagate_inbounds function velocity_and_density(v, system, particle)
    v_particle = current_velocity(v, system, particle)
    rho_particle = current_density(v, system, particle)

    return v_particle, rho_particle
end

@inline function velocity_and_density(v, ::WeaklyCompressibleSPHSystem{3}, particle)
    vrho_a = vloada(Vec{4, eltype(v)}, pointer(v, 4 * (particle - 1) + 1))
    a, b, c, d = Tuple(vrho_a)
    v_particle = SVector(a, b, c)
    rho_particle = d

    return v_particle, rho_particle
end

@propagate_inbounds function particle_neighbor_pressure(v_particle_system,
                                                        v_neighbor_system,
                                                        particle_system, neighbor_system,
                                                        particle, neighbor)
    p_a = current_pressure(v_particle_system, particle_system, particle)
    p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)

    return p_a, p_b
end

@inline function particle_neighbor_pressure(v_particle_system, v_neighbor_system,
                                            particle_system,
                                            neighbor_system::WallBoundarySystem{<:BoundaryModelDummyParticles{PressureMirroring}},
                                            particle, neighbor)
    p_a = current_pressure(v_particle_system, particle_system, particle)

    return p_a, p_a
end
