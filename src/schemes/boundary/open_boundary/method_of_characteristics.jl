@doc raw"""
    BoundaryModelLastiwka(; extrapolate_reference_values=nothing)

Boundary model for [`OpenBoundarySPHSystem`](@ref).
This model uses the characteristic variables to propagate the appropriate values
to the outlet or inlet and was proposed by Lastiwka et al. (2009).
It requires a specific flow direction to be passed to the [`BoundaryZone`](@ref).
For more information about the method see [description below](@ref method_of_characteristics).

# Keywords
- `extrapolate_reference_values=nothing`: If one of the following mirroring methods is selected,
  the reference values are extrapolated from the fluid domain to the boundary particles:
    - [`ZerothOrderMirroring`](@ref)
    - [`FirstOrderMirroring`](@ref)
    - [`SimpleMirroring`](@ref)

  **Note:** This feature is experimental and has not been fully validated yet.
  As of now, we are not aware of any published literature supporting its use.
  Note that even without this extrapolation feature,
  the reference values don't need to be prescribed - they're computed from the characteristics.
"""
struct BoundaryModelLastiwka{T}
    extrapolate_reference_values::T

    function BoundaryModelLastiwka(; extrapolate_reference_values=nothing)
        return new{typeof(extrapolate_reference_values)}(extrapolate_reference_values)
    end
end

# Called from update callback via `update_open_boundary_eachstep!`
@inline function update_boundary_quantities!(system, boundary_model::BoundaryModelLastiwka,
                                             v, u, v_ode, u_ode, semi, t)
    (; density, pressure, cache, boundary_zone,
     reference_velocity, reference_pressure, reference_density) = system
    (; flow_direction) = boundary_zone

    fluid_system = corresponding_fluid_system(system, semi)

    sound_speed = system_sound_speed(fluid_system)

    if !isnothing(boundary_model.extrapolate_reference_values)
        (; prescribed_pressure, prescribed_velocity, prescribed_density) = cache
        v_fluid = wrap_v(v_ode, fluid_system, semi)
        u_fluid = wrap_u(u_ode, fluid_system, semi)

        @trixi_timeit timer() "extrapolate and correct values" begin
            extrapolate_values!(system, boundary_model.extrapolate_reference_values,
                                v, v_fluid, u, u_fluid, semi, t;
                                prescribed_pressure, prescribed_velocity,
                                prescribed_density)
        end
    end

    # Update quantities based on the characteristic variables
    @threaded semi for particle in each_moving_particle(system)
        particle_position = current_coords(u, system, particle)

        J1 = cache.characteristics[1, particle]
        J2 = cache.characteristics[2, particle]
        J3 = cache.characteristics[3, particle]

        rho_ref = reference_value(reference_density, density[particle],
                                  particle_position, t)
        density[particle] = rho_ref + ((-J1 + (J2 + J3) / 2) / sound_speed^2)

        p_ref = reference_value(reference_pressure, pressure[particle],
                                particle_position, t)
        pressure[particle] = p_ref + (J2 + J3) / 2

        v_current = current_velocity(v, system, particle)
        v_ref = reference_value(reference_velocity, v_current,
                                particle_position, t)
        rho = density[particle]
        v_ = v_ref + ((J2 - J3) / (2 * sound_speed * rho)) * flow_direction

        for dim in 1:ndims(system)
            v[dim, particle] = v_[dim]
        end
    end

    if boundary_zone.average_inflow_velocity
        # Even if the velocity is prescribed, this boundary model computes the velocity for each particle individually.
        # Thus, turbulent flows near the inflow can lead to a non-uniform buffer particle distribution,
        # resulting in a potential numerical instability. Averaging mitigates these effects.
        average_velocity!(v, u, system, boundary_model, boundary_zone, semi)
    end

    return system
end

# Called from semidiscretization
function update_final!(system, ::BoundaryModelLastiwka, v, u, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "evaluate characteristics" begin
        evaluate_characteristics!(system, v, u, v_ode, u_ode, semi, t)
    end

    return system
end

# ==== Characteristics
# J1: Associated with convection and entropy and propagates at flow velocity.
# J2: Propagates downstream to the local flow
# J3: Propagates upstream to the local flow
function evaluate_characteristics!(system, v, u, v_ode, u_ode, semi, t)
    (; volume, cache, boundary_zone) = system
    (; characteristics, previous_characteristics) = cache
    fluid_system = corresponding_fluid_system(system, semi)

    @threaded semi for particle in eachparticle(system)
        previous_characteristics[1, particle] = characteristics[1, particle]
        previous_characteristics[2, particle] = characteristics[2, particle]
        previous_characteristics[3, particle] = characteristics[3, particle]
    end

    set_zero!(characteristics)
    set_zero!(volume)

    # Evaluate the characteristic variables with the fluid system
    evaluate_characteristics!(system, fluid_system, v, u, v_ode, u_ode, semi, t)

    # Only some of the in-/outlet particles are in the influence of the fluid particles.
    # Thus, we compute the characteristics for the particles that are outside the influence
    # of fluid particles by using the average of the values of the previous time step.
    # See eq. 27 in Negi (2020) https://doi.org/10.1016/j.cma.2020.113119
    @threaded semi for particle in each_moving_particle(system)

        # Particle is outside of the influence of fluid particles
        if isapprox(volume[particle], 0)

            # Using the average of the values at the previous time step for particles which
            # are outside of the influence of fluid particles.
            avg_J1 = zero(eltype(volume))
            avg_J2 = zero(eltype(volume))
            avg_J3 = zero(eltype(volume))
            counter = 0

            for neighbor in each_moving_particle(system)
                # Make sure that only neighbors in the influence of
                # the fluid particles are used.
                if volume[neighbor] > sqrt(eps())
                    avg_J1 += previous_characteristics[1, neighbor]
                    avg_J2 += previous_characteristics[2, neighbor]
                    avg_J3 += previous_characteristics[3, neighbor]
                    counter += 1
                end
            end

            # To prevent NANs here if the boundary has not been in contact before.
            if counter > 0
                characteristics[1, particle] = avg_J1 / counter
                characteristics[2, particle] = avg_J2 / counter
                characteristics[3, particle] = avg_J3 / counter
            else
                characteristics[1, particle] = 0
                characteristics[2, particle] = 0
                characteristics[3, particle] = 0
            end
        else
            characteristics[1, particle] /= volume[particle]
            characteristics[2, particle] /= volume[particle]
            characteristics[3, particle] /= volume[particle]
        end
        prescribe_conditions!(characteristics, particle, boundary_zone)
    end

    return system
end

function evaluate_characteristics!(system, neighbor_system::FluidSystem,
                                   v, u, v_ode, u_ode, semi, t)
    (; volume, cache, boundary_zone, density, pressure,
     reference_velocity, reference_pressure, reference_density) = system
    (; flow_direction) = boundary_zone
    (; characteristics) = cache

    v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    sound_speed = system_sound_speed(neighbor_system)

    # Loop over all fluid neighbors within the kernel cutoff
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_moving_particle(system)) do particle, neighbor,
                                                                   pos_diff, distance
        neighbor_position = current_coords(u_neighbor_system, neighbor_system, neighbor)

        # Determine current and prescribed quantities
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)
        rho_ref = reference_value(reference_density, density[particle],
                                  neighbor_position, t)

        p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)
        p_ref = reference_value(reference_pressure, pressure[particle],
                                neighbor_position, t)

        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
        v_particle = current_velocity(v, system, particle)
        v_neighbor_ref = reference_value(reference_velocity, v_particle,
                                         neighbor_position, t)

        # Determine characteristic variables
        density_term = -sound_speed^2 * (rho_b - rho_ref)
        pressure_term = p_b - p_ref
        velocity_term = rho_b * sound_speed * (dot(v_b - v_neighbor_ref, flow_direction))

        kernel_ = smoothing_kernel(neighbor_system, distance, particle)

        characteristics[1, particle] += (density_term + pressure_term) * kernel_
        characteristics[2, particle] += (velocity_term + pressure_term) * kernel_
        characteristics[3, particle] += (-velocity_term + pressure_term) * kernel_

        volume[particle] += kernel_
    end

    return system
end

@inline function prescribe_conditions!(characteristics, particle, ::BoundaryZone{OutFlow})
    # J3 is prescribed (i.e. determined from the exterior of the domain).
    # J1 and J2 is transmitted from the domain interior.
    characteristics[3, particle] = zero(eltype(characteristics))

    return characteristics
end

@inline function prescribe_conditions!(characteristics, particle, ::BoundaryZone{InFlow})
    # Allow only J3 to propagate upstream to the boundary
    characteristics[1, particle] = zero(eltype(characteristics))
    characteristics[2, particle] = zero(eltype(characteristics))

    return characteristics
end

function average_velocity!(v, u, system, ::BoundaryModelLastiwka, boundary_zone, semi)
    # Only apply averaging at the inflow
    return v
end

function average_velocity!(v, u, system, ::BoundaryModelLastiwka, ::BoundaryZone{InFlow},
                           semi)

    # Division inside the `sum` closure to maintain GPU compatibility
    avg_velocity = sum(each_moving_particle(system)) do particle
        return current_velocity(v, system, particle) / system.buffer.active_particle_count[]
    end

    @threaded semi for particle in each_moving_particle(system)
        # Set the velocity of the ghost node to the average velocity of the fluid domain
        for dim in eachindex(avg_velocity)
            @inbounds v[dim, particle] = avg_velocity[dim]
        end
    end

    return v
end
