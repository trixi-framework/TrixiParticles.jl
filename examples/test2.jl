using Metal, TrixiParticles

# Import variables into scope
trixi_include_changeprecision(Float32, @__MODULE__,
joinpath(examples_dir(), "fluid",
         "dam_break_3d.jl"),
fluid_particle_spacing=0.1,
sol=nothing, ode=nothing)

# Neighborhood search with `FullGridCellList` for GPU compatibility
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)
cell_list = FullGridCellList(; min_corner, max_corner)
semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
     neighborhood_search=GridNeighborhoodSearch{3}(; cell_list))

trixi_include_changeprecision(Float32, @__MODULE__,
                 joinpath(examples_dir(), "fluid",
                          "dam_break_3d.jl"),
                 tspan=(0.0f0, 0.1f0),
                 fluid_particle_spacing=0.1,
                 semi=semi_fullgrid,
                 data_type=MtlArray, maxiters=0)
