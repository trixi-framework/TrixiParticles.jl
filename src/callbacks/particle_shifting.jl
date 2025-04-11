@doc raw"""
    ParticleShiftingCallback()

Callback to apply the Particle Shifting Technique by [Sun et al. (2017)](@cite Sun2017).
Following the original paper, the callback is applied in every time step and not
in every stage of a multi-stage time integration method to reduce the computational
cost and improve the stability of the scheme.

## References
[Sun2017](@cite)
"""
function ParticleShiftingCallback()
    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback((particle_shifting_condition), particle_shifting!,
                            save_positions=(false, false))
end

# `condition`
function particle_shifting_condition(u, t, integrator)
    return true
end

# `affect!`
function particle_shifting!(integrator)
    t = integrator.t
    semi = integrator.p
    v_ode, u_ode = integrator.u.x
    dt = integrator.dt
    # Internal cache vector, which is safe to use as temporary array
    u_cache = first(get_tmp_cache(integrator))

    # Update quantities that are stored in the systems. These quantities (e.g. pressure)
    # still have the values from the last stage of the previous step if not updated here.
    update_systems_and_nhs(v_ode, u_ode, semi, t; update_from_callback=true)

    @trixi_timeit timer() "particle shifting" foreach_system(semi) do system
        u = wrap_u(u_ode, system, semi)
        v = wrap_v(v_ode, system, semi)
        particle_shifting!(u, v, system, v_ode, u_ode, semi, u_cache, dt)
    end

    # Tell OrdinaryDiffEq that `u` has been modified
    u_modified!(integrator, true)

    return integrator
end

function particle_shifting!(u, v, system, v_ode, u_ode, semi, u_cache, dt)
    return u
end

function particle_shifting!(u, v, system::FluidSystem, v_ode, u_ode, semi, u_cache, dt)
    # Wrap the cache vector to an NDIMS x NPARTICLES matrix.
    # We need this buffer because we cannot safely update `u` while iterating over it.
    delta_r = wrap_u(u_cache, system, semi)
    set_zero!(delta_r)

    v_max = maximum(particle -> norm(current_velocity(v, system, particle)),
                    eachparticle(system))

    # TODO this needs to be adapted to multi-resolution.
    # Section 3.2 explains what else needs to be changed.
    Wdx = smoothing_kernel(system, particle_spacing(system, 1), 1)
    h = smoothing_length(system, 1)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_moving_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            rho_a = particle_density(v, system, particle)
            rho_b = particle_density(v_neighbor, neighbor_system, neighbor)

            kernel = smoothing_kernel(system, distance, particle)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            # According to p. 29 below Eq. 9
            R = 0.2
            n = 4

            # Eq. 7 in Sun et al. (2017).
            # CFL * Ma can be rewritten as Δt * v_max / h (see p. 29, right above Eq. 9).
            delta_r_ = -dt * v_max * 4 * h * (1 + R * (kernel / Wdx)^n) *
                       m_b / (rho_a + rho_b) * grad_kernel

            # Write into the buffer
            for i in eachindex(delta_r_)
                @inbounds delta_r[i, particle] += delta_r_[i]
            end
        end
    end

    # Add δ_r from the buffer to the current coordinates
    @threaded semi for particle in eachparticle(system)
        for i in axes(delta_r, 1)
            @inbounds u[i, particle] += delta_r[i, particle]
        end
    end

    return u
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, typeof(particle_shifting!)})
    @nospecialize cb # reduce precompilation time
    print(io, "ParticleShiftingCallback()")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, typeof(particle_shifting!)})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        summary_box(io, "ParticleShiftingCallback")
    end
end

# For `OpenBoundarySPHSystem` with an `InFlow`-zone, we need a high quality transition
# from the boundary zone to the fluid domain, as small perturbations can lead to instabilities.
function particle_shifting!(u, v,
                            system::OpenBoundarySPHSystem{<:Any, <:BoundaryZone{<:InFlow}},
                            v_ode, u_ode, semi, u_cache, dt)
    (; fluid_system, boundary_zone) = system
    (; zone_origin, spanning_set) = boundary_zone

    # Wrap the cache vector to an NDIMS x NPARTICLES matrix.
    # We need this buffer because we cannot safely update `u` while iterating over it.
    delta_r = wrap_u(u_cache, system, semi)
    set_zero!(delta_r)

    v_max = maximum(particle -> norm(current_velocity(v, system, particle)),
                    eachparticle(system))

    # TODO this needs to be adapted to multi-resolution.
    # Section 3.2 explains what else needs to be changed.
    Wdx = smoothing_kernel(system, particle_spacing(system, 1), 1)
    h = smoothing_length(system, 1)

    # We use the neighborhood search (NHS) of the fluid system.
    # This is appropriate because we only want to shift the buffer particles near the transition plane.
    # Therefore, we use half of the compact support as the maximum distance within which the buffer particles are shifted.
    max_dist = 0.5 * compact_support(system_smoothing_kernel(system),
                               smoothing_length(system, 1))

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        neighborhood_search = get_neighborhood_search(fluid_system, neighbor_system, semi)

        @threaded semi for particle in each_moving_particle(system)
            particle_coords = current_coords(u, system, particle)

            particle_position = particle_coords - zone_origin

            # Distance to transition plane
            dist = dot(particle_position, spanning_set[1])

            if dist <= max_dist
                for neighbor in PointNeighbors.eachneighbor(particle_coords,
                                                            neighborhood_search)
                    neighbor_coords = current_coords(u_neighbor, neighbor_system, neighbor)

                    pos_diff = particle_coords - neighbor_coords
                    distance2 = dot(pos_diff, pos_diff)

                    # Check if the neighbor is within the search radius
                    if distance2 <= neighborhood_search.search_radius^2
                        distance = sqrt(distance2)

                        m_b = hydrodynamic_mass(neighbor_system, neighbor)
                        rho_a = particle_density(v, system, particle)
                        rho_b = particle_density(v_neighbor, neighbor_system, neighbor)

                        kernel = smoothing_kernel(system, distance, particle)
                        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance,
                                                            particle)

                        # According to p. 29 below Eq. 9
                        R = 0.2
                        n = 4

                        # Eq. 7 in Sun et al. (2017).
                        # CFL * Ma can be rewritten as Δt * v_max / h (see p. 29, right above Eq. 9).
                        delta_r_ = -dt * v_max * 4 * h * (1 + R * (kernel / Wdx)^n) *
                                   m_b / (rho_a + rho_b) * grad_kernel

                        # Write into the buffer
                        for i in eachindex(delta_r_)
                            @inbounds delta_r[i, particle] += delta_r_[i]
                        end
                    end
                end
            end
        end
    end

    # Add δ_r from the buffer to the current coordinates
    @threaded semi for particle in eachparticle(system)
        for i in axes(delta_r, 1)
            @inbounds u[i, particle] += delta_r[i, particle]
        end
    end

    return u
end
