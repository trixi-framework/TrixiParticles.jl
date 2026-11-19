# Neighborhood Search

The neighborhood search is one of the most performance-critical components.
We provide several implementations in the package
[PointNeighbors.jl](https://github.com/trixi-framework/PointNeighbors.jl).
See the PointNeighbors.jl documentation for an overview and a comparison of the
different implementations.

!!! note "Usage"
    To run a simulation with a neighborhood search implementation, pass a neighborhood
    search template to the constructor of the
    [`Semidiscretization`](@ref TrixiParticles.Semidiscretization).
    A template is just an empty neighborhood search with search radius `0.0`.
    See [`copy_neighborhood_search`](@ref) and the examples below for more details.
    ```jldoctest semi_example; output=false, setup = :(using TrixiParticles; trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing); system1 = fluid_system; system2 = boundary_system)
    semi = Semidiscretization(system1, system2,
                              neighborhood_search=PrecomputedNeighborhoodSearch{2}())

    # output
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ Semidiscretization                                                                               │
    │ ══════════════════                                                                               │
    │ #spatial dimensions: ………………………… 2                                                                │
    │ #systems: ……………………………………………………… 2                                                                │
    │ neighborhood search: ………………………… PrecomputedNeighborhoodSearch                                    │
    │ total #particles: ………………………………… 636                                                              │
    │ eltype: …………………………………………………………… Float64                                                          │
    │ coordinates eltype: …………………………… Float64                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    ```

    The keyword argument `periodic_box` in the neighborhood search constructors can be used
    to define a periodic domain. See the PointNeighbors.jl docs for more details.
    ```jldoctest semi_example; output = false
    periodic_box = PeriodicBox(min_corner=[0.0, -0.25], max_corner=[1.0, 0.75])
    semi = Semidiscretization(system1, system2,
                              neighborhood_search=GridNeighborhoodSearch{2}(; periodic_box))

    # output
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ Semidiscretization                                                                               │
    │ ══════════════════                                                                               │
    │ #spatial dimensions: ………………………… 2                                                                │
    │ #systems: ……………………………………………………… 2                                                                │
    │ neighborhood search: ………………………… GridNeighborhoodSearch                                           │
    │ total #particles: ………………………………… 636                                                              │
    │ eltype: …………………………………………………………… Float64                                                          │
    │ coordinates eltype: …………………………… Float64                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    ```

## [Neighborhood Search Handlers](@id neighborhood_search_handlers)

Neighborhood search handlers control how neighborhood searches for the interacting
systems are stored and looked up. This only affects memory use and update cost
by avoiding redundant neighborhood searches, not the underlying search algorithm.

By default, the handler is chosen automatically (see also
[neighborhood search performance](@ref neighborhood_search_performance)).
To choose a handler explicitly, pass the handler type to `Semidiscretization`.

```jldoctest semi_example; output=false, setup = :(using TrixiParticles; trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing); system1 = fluid_system; system2 = boundary_system)
semi = Semidiscretization(system1, system2,
                          neighborhood_search_handler=PairsNHSHandler)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Semidiscretization                                                                               │
│ ══════════════════                                                                               │
│ #spatial dimensions: ………………………… 2                                                                │
│ #systems: ……………………………………………………… 2                                                                │
│ neighborhood search: ………………………… GridNeighborhoodSearch                                           │
│ total #particles: ………………………………… 636                                                              │
│ eltype: …………………………………………………………… Float64                                                          │
│ coordinates eltype: …………………………… Float64                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## [Sorting Particles](@id particle_sorting)

Sorting the particles by neighborhood search cell improves memory locality and makes
the neighbor loops faster on both CPUs and GPUs.
When a [`SortingGridNeighborhoodSearch`](@ref) is used, the particles are sorted by cell
in every update of the neighborhood search.
This requires that the particles of all systems are stored in the same grid
(see [neighborhood search performance](@ref neighborhood_search_performance)),
that is, all pairs of systems use the same search radius and no neighborhood search
handler other than the default is passed to the [`Semidiscretization`](@ref).
Otherwise, an error is thrown.

```jldoctest semi_example; output=false, filter = [r"u0: .*", r"\[ Info: .*"]
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)
neighborhood_search = SortingGridNeighborhoodSearch{2}(; min_corner, max_corner)
semi = Semidiscretization(system1, system2; neighborhood_search)
ode = semidiscretize(semi, (0.0, 1.0))

# output
ODEProblem with uType RecursiveArrayTools.ArrayPartition{Float64, Tuple{TrixiParticles.ThreadedBroadcastArray{Float64, 1, Vector{Float64}, PolyesterBackend}, TrixiParticles.ThreadedBroadcastArray{Float64, 1, Vector{Float64}, PolyesterBackend}}} and tType Float64. In-place: true
Non-trivial mass matrix: false
timespan: (0.0, 1.0)
u0: ([...], [...]) *this line is ignored by filter*
```

Currently, sorting particles is only supported for the
[`WeaklyCompressibleSPHSystem`](@ref) and the [`WallBoundarySystem`](@ref).
Fixed wall boundaries are only sorted once at the beginning of the simulation.

The time integration arrays `v_ode` and `u_ode` (`sol.u[i].x` or `integrator.u.x`)
are never reordered, so the particles in these arrays stay in their original order.
This is also the order in which particles are written to VTK files.
All arrays owned by the systems (like the particle masses) are stored in the sorted order.
Sorting the particles reorders these arrays in place. To avoid modifying the passed systems,
[`semidiscretize`](@ref) creates a copy of the [`Semidiscretization`](@ref),
which can be accessed with `ode.p.semi`.

!!! warning "Accessing the time integration arrays"
    When particles are sorted, the particle indices of the systems don't match
    the particle indices in the time integration arrays.
    Therefore, the time integration arrays must not be indexed with particle indices,
    and `TrixiParticles.wrap_v` and `TrixiParticles.wrap_u` throw an error
    for these arrays.
    Callbacks and custom quantities receive copies of the time integration arrays
    in the sorted order of the system arrays, for which `wrap_v` and `wrap_u`
    work as usual.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "neighborhood_search.jl"),
         joinpath("general", "particle_sorting.jl")]
```
