"""
    RestartCondition(system::AbstractSystem, filename; output_directory="out")

Create a `RestartCondition` for the given `system` from the specified `filename`.

# Arguments
- `system`: The system to restart.
- `filename`: The name of the file from which to restart. This file should be in
              VTK format and correspond to the specified `system`.

# Keywords
- `output_directory`: The directory where the restart file is located (default: `"out"`).
"""
struct RestartCondition{V, U}
    system    :: AbstractSystem
    v_restart :: V
    u_restart :: U
    t_restart :: Real
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

    t_restart = convert(eltype(system), restart_data.time)

    return RestartCondition(system, v_restart, u_restart, t_restart)
end

function Base.show(io::IO, rc::RestartCondition)
    @nospecialize rc # reduce precompilation time

    print(io, "RestartCondition{$(nameof(typeof(rc.system)))}()")
end

function Base.show(io::IO, ::MIME"text/plain", rc::RestartCondition)
    @nospecialize rc # reduce precompilation time

    if get(io, :compact, false)
        show(io, rc)
    else
        summary_header(io, "RestartCondition")
        summary_line(io, "System", "$(nameof(typeof(rc.system)))")
        summary_line(io, "#particles u", "$(size(rc.u_restart, 2))")
        summary_line(io, "#particles v", "$(size(rc.v_restart, 2))")
        summary_line(io, "eltype u", "$(eltype(rc.u_restart))")
        summary_line(io, "eltype v", "$(eltype(rc.v_restart))")
        summary_footer(io)
    end
end

function set_intial_conditions!(v0_ode, u0_ode, semi, restart_conditions)
    # Check number of systems
    if length(semi.systems) != length(restart_conditions)
        throw(ArgumentError("Number of systems in `semi` does not match number of `restart_conditions`"))
    end

    # Check that systems match
    foreach_system(semi) do system
        system_index = system_indices(system, semi)
        if !(system == restart_conditions[system_index].system)
            throw(ArgumentError("System at index $system_index in `semi` does not match system in `restart_conditions`"))
        end
    end

    # Set initial conditions
    foreach_noalloc(semi.systems, restart_conditions) do (system, restart_condition)
        v0_system = wrap_v(v0_ode, system, semi)
        u0_system = wrap_u(u0_ode, system, semi)

        v0_system .= Adapt.adapt(semi.parallelization_backend, restart_condition.v_restart)
        u0_system .= Adapt.adapt(semi.parallelization_backend, restart_condition.u_restart)
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

function initialize_neighborhood_searches!(semi, u0_ode, restart_conditions)
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            # TODO Initialize after adapting to the GPU.
            # Currently, this cannot use `semi.parallelization_backend`
            # because data is still on the CPU.
            PointNeighbors.initialize!(get_neighborhood_search(system, neighbor, semi),
                                       initial_restart_coordinates(system, u0_ode, semi),
                                       initial_restart_coordinates(neighbor, u0_ode, semi),
                                       eachindex_y=each_active_particle(neighbor),
                                       parallelization_backend=PolyesterBackend())
        end
    end

    return semi
end

function initial_restart_coordinates(system, u0_ode, semi)
    # Transfer to CPU if data is on the GPU. Do nothing if already on CPU.
    return transfer2cpu(wrap_u(u0_ode, system, semi))
end

function initial_restart_coordinates(system::Union{WallBoundarySystem,
                                                   AbstractStructureSystem}, u0_ode, semi)
    return initial_coordinates(system)
end

function initialize!(semi::Semidiscretization, restart_conditions)
    foreach_system(semi) do system
        # Initialize this system
        initialize_restart!(system, semi)
    end

    return semi
end

initialize_restart!(system, semi) = initialize!(system, semi)
