using TrixiParticles
using CUDA

fluid_particle_spacing = 0.003

# Load setup from dam break example
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=fluid_particle_spacing,
              tank_size=(4.0, 3.0), W=1.0, H=2.0,
              spacing_ratio=1, boundary_layers=1,
              sol=nothing, ode=nothing)

tank.fluid.coordinates .+= 0.005
tank.boundary.coordinates .+= 0.005

# Define a GPU-compatible neighborhood search
min_corner = minimum(tank.boundary.coordinates, dims=2) .- 1
max_corner = maximum(tank.boundary.coordinates, dims=2) .+ 1
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; cell_list,
                                                update_strategy=ParallelUpdate())

# Run the dam break simulation with this neighborhood search
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              tank=tank,
              smoothing_length=1.414216 * fluid_particle_spacing,
              time_integration_scheme=SymplecticPositionVerlet(),
              boundary_density_calculator=ContinuityDensity(),
              fluid_particle_spacing=fluid_particle_spacing,
              tank_size=(4.0, 3.0), W=1.0, H=2.0,
              spacing_ratio=1, boundary_layers=1,
              tspan=(0.0, 2.0), cfl=0.2,
              neighborhood_search=neighborhood_search,
              viscosity_wall=viscosity_fluid,
              dt=1e-5, stepsize_callback=nothing, saving_callback=nothing,
              parallelization_backend=CUDABackend())
