struct RCRBoundaryModel{ELTYPE, PAI, AI, DT}
    resistance_1             :: ELTYPE
    resistance_2             :: ELTYPE
    capacitance              :: ELTYPE
    cross_section_area       :: ELTYPE
    previous_averaged_inflow :: PAI
    averaged_inflow          :: AI
    dt                       :: DT
end

function RCRBoundaryModel(; resistance_1, resistance_2, capacitance, pipe_radius)
    ELTYPE = eltype(resistance_1)
    cross_section_area = pi * pipe_radius^2

    previous_averaged_inflow = Ref(zero(ELTYPE))
    averaged_inflow = Ref(zero(ELTYPE))
    dt = Ref(zero(ELTYPE))

    return RCRBoundaryModel{ELTYPE, typeof(previous_averaged_inflow),
                            typeof(averaged_inflow),
                            typeof(dt)}(resistance_1, resistance_2, capacitance,
                                        cross_section_area,
                                        previous_averaged_inflow, averaged_inflow, dt)
end

@inline function current_pressure_open_boundary(v, system, pressure_model::RCRBoundaryModel)
    return view(v, size(v, 1), :)
end

# This is called each time step from `update_open_boundary_eachstep!`
update_pressure_model!(system, ::Nothing, v, dt) = system

# This is called each time step from `update_open_boundary_eachstep!`
function update_pressure_model!(system, model::RCRBoundaryModel, v, dt)
    (; boundary_zone) = system

    vel_normal = sum(each_moving_particle(system)) do particle
        # TODO: Lu et al. (2024), p. 9:
        # Is it the normal dircetion pointing into the fluid domain or the flow direction?
        v_n = dot(current_velocity(v, system, particle), boundary_zone.flow_direction)
        return v_n / system.buffer.active_particle_count[]
    end

    model.previous_averaged_inflow[] = model.averaged_inflow[]
    model.averaged_inflow[] = vel_normal * pi * model.cross_section_area
    model.dt[] = dt

    return system
end

pressure_evolution!(dv, system, ::Nothing, u, v, semi) = dv

function pressure_evolution!(dv, system, pressure_model::RCRBoundaryModel, v, u, semi)
    (; resistance_1, resistance_2, capacitance,
    averaged_inflow, previous_averaged_inflow, dt) = pressure_model

    factor = -1 / (capacitance[] * resistance_2[])
    second_term = (resistance_1[] + resistance_2[]) * averaged_inflow[] /
                  (capacitance[] * resistance_2[])

    # TODO: Check this.
    # We calculate Q^n and Q^(n-1) for each time step, sow we use current `dt` from the callback here.
    # According to Lu et al. (2024), Eq. (41):
    # (dp/dt)^n = ... + ... + R_1 * (Q^n - Q^(n-1)) / Δt
    # Does the time integration cancel out  Δt in this context?
    # Is it valid to omit 1/Δt?
    # I think only if we compute Q^n and Q^(n-1) for each stage, no?
    third_term = dt[] > 0 ?
                 resistance_1[] * (averaged_inflow[] - previous_averaged_inflow[]) / dt[] :
                 zero(eltype(v))


    @threaded semi for particle in each_moving_particle(system)
        dv[end, particle] = factor * current_pressure(v, system, particle) + second_term +
                            third_term
    end

    return dv
end
