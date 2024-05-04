# Neighborhood Search

The neighborhood search is the most essential component for performance.
We provide several implementations in the package
[TrixiNeighborhoodSearch.jl](https://github.com/trixi-framework/TrixiNeighborhoodSearch.jl).
See the docs of this package for an overview and a comparison of different implementations.

!!! note "Usage"
    To run a simulation with a neighborhood search implementation, just pass the type
    to the constructor of the [`Semidiscretization`](@ref):
    ```jldoctest semi_example; output=false, setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing); system1 = fluid_system; system2 = boundary_system)
    semi = Semidiscretization(system1, system2,
                              neighborhood_search=GridNeighborhoodSearch)

    # output
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ Semidiscretization                                                                               │
    │ ══════════════════                                                                               │
    │ #spatial dimensions: ………………………… 2                                                                │
    │ #systems: ……………………………………………………… 2                                                                │
    │ neighborhood search: ………………………… GridNeighborhoodSearch                                           │
    │ total #particles: ………………………………… 636                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    ```
    The keyword arguments `periodic_box_min_corner` and `periodic_box_max_corner` mentioned
    in the TrixiNeighborhoodSearch.jl docs can also be passed to the
    [`Semidiscretization`](@ref) and will internally be forwarded to the neighborhood search.
    See the docs of [`Semidiscretization`](@ref) for more details.
    ```jldoctest semi_example; output = false
    semi = Semidiscretization(system1, system2,
                              neighborhood_search=GridNeighborhoodSearch,
                              periodic_box_min_corner=[0.0, -0.25],
                              periodic_box_max_corner=[1.0, 0.75])

    # output
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ Semidiscretization                                                                               │
    │ ══════════════════                                                                               │
    │ #spatial dimensions: ………………………… 2                                                                │
    │ #systems: ……………………………………………………… 2                                                                │
    │ neighborhood search: ………………………… GridNeighborhoodSearch                                           │
    │ total #particles: ………………………………… 636                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    ```
