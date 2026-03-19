# These are the systems that require sorting.
# TODO: The `DEMSystem` should be added here in the future.
# Boundary particles always stay fixed relative to each other, TLSPH computes in the initial configuration.
const RequiresSortingSystem = AbstractFluidSystem

struct SortingCallback{I}
    interval::I
end

"""
    SortingCallback(; interval::Integer, initial_sort=true)

Reorders particles according to neighborhood-search cells for performance optimization.

When particles become very unordered throughout a long-running simulation, performance
decreases due to increased cache-misses (on CPUs) and lack of block structure (on GPUs).
On GPUs, a fully shuffled particle ordering causes a 3-4x slowdown compared to a sorted configuration.
On CPUs, there is no difference for small problems (<16k particles), but the performance penalty
grows linearly with the problem size up to 10x slowdown for very large problems (65M particles).
See [#1044](https://github.com/trixi-framework/TrixiParticles.jl/pull/1044) for more details.

# Keywords
- `interval::Integer`: Sort particles at the end of every `interval` time steps.
- `initial_sort=true`: When enabled, particles are sorted at the beginning of the simulation.
                       When the initial configuration is a perfect grid of particles,
                       sorting at the beginning is not necessary and might even slightly
                       slow down the first time steps, since a perfect grid is even better
                       than sorting by NHS cell index.
"""
function SortingCallback(; interval::Integer, initial_sort=true)
    sorting_callback! = SortingCallback(interval)

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(sorting_callback!, sorting_callback!,
                            initialize=initial_sort ? (initial_sort!) : nothing,
                            save_positions=(false, false))
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

function sort_particles!(system::RequiresSortingSystem, v, u, semi)
    nhs = get_neighborhood_search(system, semi)

    if !(nhs isa GridNeighborhoodSearch)
        throw(ArgumentError("`SortingCallback` can only be used with a `GridNeighborhoodSearch`"))
    end

    sort_particles!(system, v, u, nhs, nhs.cell_list, semi)
end

# TODO: Sort also masses and particle spacings for variable smoothing lengths.
function sort_particles!(system::RequiresSortingSystem, v, u, nhs,
                         cell_list::FullGridCellList, semi)
    cell_coords = allocate(semi.parallelization_backend, SVector{ndims(system), Int},
                           nparticles(system))
    @threaded semi for particle in each_active_particle(system)
        point_coords = current_coords(u, system, particle)
        cell_coords[particle] = PointNeighbors.cell_coords(point_coords, nhs)
    end

    perm = sortperm(transfer2cpu(cell_coords))

    sort_system!(system, v, u, perm, system.buffer)

    return system
end

function sort_system!(system, v, u, perm, buffer::Nothing)
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
