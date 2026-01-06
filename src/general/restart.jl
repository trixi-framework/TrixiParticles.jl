struct RestartCondition{V, U}
    v_restart::V
    u_restart::U
    t_restart::Real
end

function RestartCondition(system::AbstractSystem, filename; output_directory="out")
    if !occursin(vtkname(system), basename(splitext(filename)[1]))
        throw(ArgumentError("Filename '$filename' does not seem to correspond to system of type $(nameof(typeof(system)))."))
    end

    restart_file = joinpath(output_directory, filename)
    restart_data = vtk2trixi(restart_file)
    v_restart = restart_v(system, restart_data)
    u_restart = restart_u(system, restart_data)

    precondition_system!(system, restart_file)

    return RestartCondition(v_restart, u_restart, restart_data.time)
end

function Base.show(io::IO, rc::RestartCondition)
    @nospecialize rc # reduce precompilation time

    print(io, "RestartCondition{}()")
end

function Base.show(io::IO, ::MIME"text/plain", rc::RestartCondition)
    @nospecialize rc # reduce precompilation time

    if get(io, :compact, false)
        show(io, rc)
    else
        summary_header(io, "RestartCondition")
        summary_line(io, "#particles u", "$(size(rc.u_restart, 2))")
        summary_line(io, "#particles v", "$(size(rc.v_restart, 2))")
        summary_line(io, "eltype", "$(eltype(rc.v_restart))")
        summary_footer(io)
    end
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

function time_span(tspan, restart_conditions)
    t_restart = first(restart_conditions).t_restart

    if !isapprox(tspan[1], t_restart)
        @info "Adjusting initial time from $(tspan[1]) to restart time $t_restart"
    end

    return (t_restart, tspan[2])
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

precondition_system!(system, restart_file) = system
