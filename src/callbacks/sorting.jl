struct SortingCallback{I}
    interval::I
end

"""
    SortingCallback(; interval::Integer=-1, dt=0.0)

# Keywords
- `interval=1`: Sort particles at the end of every `interval` time steps.
- `dt`: Sort particles in regular intervals of `dt` in terms of integration time
        by adding additional `tstops` (note that this may change the solution).
"""
function SortingCallback(; interval::Integer=-1, dt=0.0)
    if dt > 0 && interval !== -1
        throw(ArgumentError("Setting both interval and dt is not supported!"))
    end

    # Sort in intervals in terms of simulation time
    if dt > 0
        interval = Float64(dt)

        # Sort every time step (default)
    elseif interval == -1
        interval = 1
    end

    sorting_callback! = SortingCallback(interval)

    if dt > 0
        # Add a `tstop` every `dt`
        return PeriodicCallback(sorting_callback!, dt,
                                initialize=(initial_sort!),
                                save_positions=(false, false))
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(sorting_callback!, sorting_callback!,
                                initialize=(initial_sort!),
                                save_positions=(false, false))
    end
end

# `initialize`
function initial_sort!(cb, u, t, integrator)
    # The `SortingCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.

    initial_sort!(cb.affect!, u, t, integrator)
end

function initial_sort!(cb::SortingCallback, u, t, integrator)
    return cb(integrator)
end

# `condition`
function (sorting_callback!::SortingCallback)(u, t, integrator)
    (; interval) = sorting_callback!

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (sorting_callback!::SortingCallback)(integrator)
    semi = integrator.p
    v_ode, u_ode = integrator.u.x

    @trixi_timeit timer() "sorting callback" begin
        foreach_system(semi) do system
            v = wrap_v(v_ode, system, semi)
            u = wrap_u(u_ode, system, semi)

            sort_particles!(system, v, u, semi)
        end
    end

    # Tell OrdinaryDiffEq that `integrator.u` has been modified
    u_modified!(integrator, true)

    return integrator
end

sort_particles!(system, v, u, semi) = system

function sort_particles!(system::AbstractFluidSystem, v, u, semi)
    nhs = get_neighborhood_search(system, semi)
    cell_list = nhs.cell_list

    sort_particles!(system, v, u, nhs, cell_list, semi)
end

# TODO: Sort also masses and particle spacings for variable smoothing lengths.
function sort_particles!(system::AbstractFluidSystem, v, u, nhs,
                         cell_list::FullGridCellList, semi)
    cell_ids = zero(allocate(semi.parallelization_backend, Int, nparticles(system)))
    @threaded semi for particle in each_active_particle(system)
        point_coords = current_coords(u, system, particle)
        cell_ids[particle] = PointNeighbors.cell_index(cell_list,
                                                       PointNeighbors.cell_coords(point_coords,
                                                                                  nhs))
    end

    perm = sortperm(transfer2cpu(cell_ids))

    sort_system!(system, v, u, perm, system.buffer)

    return system
end

function sort_system!(system, v, u, perm, buffer::Nothing)
    # Note that the following contain also inactive particles
    system_coords = current_coordinates(u, system)
    system_velocity = current_velocity(v, system)
    system_density = current_density(v, system)
    system_pressure = current_pressure(v, system)

    system_coords .= system_coords[:, perm]
    system_velocity .= system_velocity[:, perm]
    system_pressure .= system_pressure[perm]
    system_density .= system_density[perm]

    return system
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SortingCallback})
    @nospecialize cb # reduce precompilation time
    print(io, "SortingCallback(interval=", cb.affect!.interval, ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SortingCallback}})
    @nospecialize cb # reduce precompilation time
    print(io, "SortingCallback(dt=", cb.affect!.affect!.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SortingCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        sorting_cb = cb.affect!
        setup = [
            "interval" => sorting_cb.interval
        ]
        summary_box(io, "SortingCallback", setup)
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SortingCallback}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        sorting_cb = cb.affect!.affect!
        setup = [
            "dt" => sorting_cb.interval
        ]
        summary_box(io, "SortingCallback", setup)
    end
end
