struct RestartCondition{V, U}
    v_restart::V
    u_restart::U
    t_restart::Real
end

function RestartCondition(system::AbstractSystem, restart_file; precondition_values=nothing)
    restart_data = vtk2trixi(restart_file)
    v_restart = restart_v(system, restart_data)
    u_restart = restart_u(system, restart_data)

    # TODO
    t_restart = 0.0

    precondition_system!(system, precondition_values)

    return RestartCondition(v_restart, u_restart, t_restart)
end

function set_intial_conditions!(v0_ode, u0_ode, semi, restart_conditions)
    if length(semi.systems) != length(restart_conditions)
        throw(ArgumentError("Number of systems in `semi` does not match number of `restart_conditions`"))
    end

    foreach_noalloc(semi.systems, restart_conditions) do (system, restart_condition)
        v0_system = wrap_v(v0_ode, system, semi)
        u0_system = wrap_u(u0_ode, system, semi)

        u0_system .= restart_condition.u_restart
        v0_system .= restart_condition.v_restart
    end
end

function time_span(tspan, restart_conditions::RestartCondition)
    return (first(restart_conditions).t_restart, tspan[2])
end

function write_density_and_pressure!(v_restart, system, density_calculator,
                                     pressure, density)
    return v_restart
end

function write_density_and_pressure!(v_restart, system,
                                     density_calculator::ContinuityDensity,
                                     pressure, density)
    v_restart[size(v_restart, 1), :] = density

    return v_restart
end

function write_density_and_pressure!(v_restart, system::EntropicallyDampedSPHSystem,
                                     density_calculator::ContinuityDensity,
                                     pressure, density)
    v_restart[size(v_restart, 1), :] = density
    v_restart[size(v_restart, 1) - 1, :] = pressure

    return v_restart
end

precondition_system!(system, values::Nothing) = system

function precondition_system!(system::OpenBoundarySystem, values)
    (; pressure_reference_values, boundary_zones_flow_rate) = system.cache

    if !haskey(values, :pressure_reference_values, :boundary_zones_flow_rate)
        throw(ArgumentError("Missing required fields in `values` for `OpenBoundarySystem`"))
    end

    foreach_noalloc(pressure_reference_values,
                    values.pressure_reference_values) do (pressure_model,
                                                          previous_pressure)
        if isa(pressure_model, AbstractPressureModel)
            pressure_model.pressure[] = previous_pressure
        else
            if previous_pressure != Inf
                throw(ArgumentError("Expected `Inf` for non-pressure-model boundary zones"))
            end
        end
    end

    foreach_noalloc(pressure_reference_values,
                    values.boundary_zones_flow_rate) do (pressure_model,
                                                         previous_flow_rate)
        if isa(pressure_model, AbstractPressureModel)
            pressure_model.flow_rate[] = previous_flow_rate
        else
            if previous_flow_rate != Inf
                throw(ArgumentError("Expected `Inf` for non-pressure-model boundary zones"))
            end
        end
    end

    return system
end
