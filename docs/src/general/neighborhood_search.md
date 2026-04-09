# Neighborhood Search

The neighborhood search is the most essential component for performance.
We provide several implementations in the package
[PointNeighbors.jl](https://github.com/trixi-framework/PointNeighbors.jl).
See the docs of this package for an overview and a comparison of different implementations.

!!! note "Usage"
    To run a simulation with a neighborhood search implementation, pass a template of the
    neighborhood search to the constructor of the [`Semidiscretization`](@ref).
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
