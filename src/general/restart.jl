function set_initial_conditions!(v0_ode, u0_ode, semi, restart_with::Tuple{Vararg{String}})
    # Check number of systems
    if length(semi.systems) != length(restart_with)
        throw(ArgumentError("Number of systems in `semi` does not match number of restart files provided " *
                            "in `restart_with`"))
    end

    # Check that systems match
    expected_system_names = system_names(semi.systems)
    foreach_system(semi) do system
        system_index = system_indices(system, semi)
        filename = restart_with[system_index]
        expected_system_name = expected_system_names[system_index]
        if !occursin(expected_system_name, basename(splitext(filename)[1]))
            throw(ArgumentError("Filename '$filename' for system $system_index does not contain expected name '$expected_system_name'. " *
                                "Expected a VTK file for system of type $(nameof(typeof(system)))."))
        end
    end

    # Set initial conditions
    foreach_noalloc(semi.systems, restart_with) do (system, restart_file)
        v0_system = wrap_v(v0_ode, system, semi)
        u0_system = wrap_u(u0_ode, system, semi)

        restart_data = vtk2trixi(restart_file)
        v_restart = restart_v(system, restart_data)
        u_restart = restart_u(system, restart_data)

        v0_system .= Adapt.adapt(semi.parallelization_backend, v_restart)
        u0_system .= Adapt.adapt(semi.parallelization_backend, u_restart)

        precondition_system!(system, restart_file)
    end
end

function time_span(tspan, restart_with::Tuple{Vararg{String}})
    restart_data = vtk2trixi(first(restart_with))
    t_restart = convert(eltype(tspan), restart_data.time)

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

function initialize_neighborhood_searches!(semi, u0_ode,
                                           restart_with::Tuple{Vararg{String}})
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

function initialize!(semi::Semidiscretization, restart_with::Tuple{Vararg{String}})
    foreach_system(semi) do system
        # Initialize this system
        initialize_restart!(system, semi)
    end

    return semi
end

initialize_restart!(system, semi) = initialize!(system, semi)
