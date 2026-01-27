@doc raw"""
    BoundaryModelDynamicalPressureZhang()

Boundary model for the [`OpenBoundarySystem`](@ref).
This model implements the method of [Zhang et al. (2025)](@cite Zhang2025) for imposing dynamical pressure conditions at open boundaries.
In this model, the momentum equation is solved for particles within the [`BoundaryZone`](@ref).
The prescribed boundary pressure is directly incorporated into the SPH approximation of the pressure gradient for particles near the boundary.
This model is highly robust for handling bidirectional flow,
allowing particles to enter or leave the domain through a single boundary surface.
For more information about the method see [the documentation](@ref dynamical_pressure).
"""
struct BoundaryModelDynamicalPressureZhang end

@inline function v_nvariables(system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang})
    v_nvariables(system, system.fluid_system)
end

@inline function v_nvariables(system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                              ::WeaklyCompressibleSPHSystem)
    # Velocity and density is integrated
    return ndims(system) + 1
end

@inline function v_nvariables(system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                              ::EntropicallyDampedSPHSystem)
    # Velocity, density and pressure is integrated
    return ndims(system) + 2
end

@inline function current_density(v,
                                 system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang})
    return view(v, size(v, 1), :)
end

@inline function current_pressure(v,
                                  system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang})
    current_pressure(v, system, system.fluid_system)
end

@inline function current_pressure(v,
                                  system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                                  ::WeaklyCompressibleSPHSystem)
    return system.cache.pressure
end

@inline function current_pressure(v,
                                  system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                                  ::EntropicallyDampedSPHSystem)
    return view(v, ndims(system) + 1, :)
end

@inline function density_diffusion(system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang})
    return system.cache.density_diffusion
end

@inline function density_calculator(system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang})
    return system.cache.density_calculator
end

@inline impose_rest_density!(v, system, particle, boundary_model) = v

@inline function impose_rest_density!(v, system, particle,
                                      boundary_model::BoundaryModelDynamicalPressureZhang)
    (; density_rest, pressure_boundary) = system.cache

    state_equation = system_state_equation(system.fluid_system)
    density = current_density(v, system)

    # Density of recycled buffer particles is obtained following the EoS (Zhang et al. 2025)
    inverse_state_equation!(density, density_rest, state_equation, pressure_boundary,
                            particle)
end

@inline impose_rest_pressure!(v, system, particle, boundary_model) = v

@inline function impose_rest_pressure!(v, system, particle,
                                       boundary_model::BoundaryModelDynamicalPressureZhang)
    boundary_zone = current_boundary_zone(system, particle)
    set_particle_pressure!(v, system, particle, boundary_zone.rest_pressure)
end

function write_v0!(v0, system::OpenBoundarySystem, ::BoundaryModelDynamicalPressureZhang)
    write_v0!(v0, system, system.boundary_model, system.fluid_system)
end

function write_v0!(v0, system::OpenBoundarySystem, ::BoundaryModelDynamicalPressureZhang,
                   ::EntropicallyDampedSPHSystem)
    v0[size(v0, 1), :] = system.initial_condition.density
    v0[size(v0, 1) - 1, :] = system.initial_condition.pressure

    return v0
end

function write_v0!(v0, system::OpenBoundarySystem, ::BoundaryModelDynamicalPressureZhang,
                   ::WeaklyCompressibleSPHSystem)
    v0[size(v0, 1), :] = system.initial_condition.density

    return v0
end

function reference_pressure(boundary_zone, v,
                            system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                            particle, pos, t)
    (; prescribed_pressure, rest_pressure) = boundary_zone
    (; pressure_reference_values) = system.cache

    # From Zhang et al. (2025):
    #   "When the bidirectional in-/outlet buffer works with the velocity
    #   in-/outflow boundary condition, such as in PIVO (Pressurized Inlet,
    #   Velocity Outlet) and VIPO (Velocity Inlet, Pressure Outlet) flows, the
    #   pressure boundary condition should also be imposed at the velocity
    #   in-/outlet to eliminate the truncated error in approximating pressure gradient,
    #   but the corresponding p_b in Eq. (13) is given as p_i.
    #   Meanwhile, both the density and pressure of newly populated particles remain unchanged."
    #
    # In simple words: If no pressure is prescribed, we still prescribe the rest pressure.
    if prescribed_pressure
        zone_id = system.boundary_zone_indices[particle]

        # `pressure_reference_values[zone_id](pos, t)`, but in a type-stable way
        return apply_ith_function(pressure_reference_values, zone_id, pos, t)
    else
        return rest_pressure
    end
end

function update_boundary_model!(system, boundary_model::BoundaryModelDynamicalPressureZhang,
                                v, u, v_ode, u_ode, semi, t)
    (; pressure_boundary) = system.cache

    compute_pressure!(system, system.fluid_system, v, semi)

    @threaded semi for particle in each_integrated_particle(system)
        boundary_zone = current_boundary_zone(system, particle)
        particle_coords = current_coords(u, system, particle)

        pressure_boundary[particle] = reference_pressure(boundary_zone, v, system,
                                                         particle, particle_coords, t)
    end

    return system
end

function compute_pressure!(system::OpenBoundarySystem,
                           fluid_system::EntropicallyDampedSPHSystem, v, semi)
    return system
end

function compute_pressure!(system::OpenBoundarySystem,
                           fluid_system::WeaklyCompressibleSPHSystem, v, semi)
    @threaded semi for particle in eachparticle(system)
        density = current_density(v, system, particle)
        system.cache.pressure[particle] = fluid_system.state_equation(density)
    end

    return system
end

# Called from update callback via `update_open_boundary_eachstep!`
function update_boundary_quantities!(system,
                                     boundary_model::BoundaryModelDynamicalPressureZhang,
                                     v, u, v_ode, u_ode, semi, t)
    (; pressure_boundary) = system.cache

    @threaded semi for particle in each_integrated_particle(system)
        boundary_zone = current_boundary_zone(system, particle)
        (; prescribed_density, prescribed_velocity) = boundary_zone

        particle_coords = current_coords(u, system, particle)

        # Pressure is always prescribed with `BoundaryModelDynamicalPressureZhang`,
        # as the term in the momentum equation vanishes for full kernel support.
        pressure_boundary[particle] = reference_pressure(boundary_zone, v, system,
                                                         particle, particle_coords, t)

        if prescribed_density
            rho_ref = reference_density(boundary_zone, v, system, particle,
                                        particle_coords, t)
            set_particle_density!(v, system, particle, rho_ref)
        end

        if prescribed_velocity
            v_ref = reference_velocity(boundary_zone, v, system, particle,
                                       particle_coords, t)

            for dim in eachindex(v_ref)
                @inbounds v[dim, particle] = v_ref[dim]
            end
        end

        # Project the velocity on the normal direction of the boundary zone
        # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
        # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
        # only this component of interpolated velocity is kept [...]"
        project_velocity_on_face_normal!(v, system, particle, boundary_zone,
                                         boundary_model)
    end

    return system
end

function project_velocity_on_face_normal!(v, system, particle, boundary_zone,
                                          boundary_model::BoundaryModelDynamicalPressureZhang)
    # Project the velocity on the normal direction of the boundary zone
    # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
    # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
    # only this component of interpolated velocity is kept [...]"
    v_particle = current_velocity(v, system, particle)
    v_particle_projected = dot(v_particle, boundary_zone.face_normal) *
                           boundary_zone.face_normal

    for dim in eachindex(v_particle)
        @inbounds v[dim, particle] = v_particle_projected[dim]
    end

    return v
end

function inverse_state_equation!(density, density_rest, state_equation::Nothing,
                                 pressure, particle)
    # If no equation of state is provided (e.g. for an `EntropicallyDampedSPHSystem`),
    # set the particle's density to the rest density.
    @inbounds density[particle] = density_rest
    return density
end

function inverse_state_equation!(density, density_rest, state_equation, pressure, particle)
    return inverse_state_equation!(density, state_equation, pressure, particle)
end
