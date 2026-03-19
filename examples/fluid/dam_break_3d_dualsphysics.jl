# Modify the 01_DamBreak example of DualSPHysics like this:
# <parameter key="StepAlgorithm" value="2" comment="Step Algorithm 1:Verlet, 2:Symplectic (default=1)" />
# <parameter key="DensityDT" value="1" comment="Density Diffusion Term 0:None, 1:Molteni, 2:Fourtakas, 3:Fourtakas(full) (default=0)" />
# <parameter key="TimeMax" value="1.0" comment="Time of simulation" units_comment="seconds" />
#
# When comparing with high resolution, change the resolution here:
# <definition dp="0.002" units_comment="metres (m)">
# With this resolution, use:
# <parameter key="DtFixed" value="1e-5" comment="Fixed Dt value. Use 0 to disable (default=disabled)" units_comment="seconds" />

using TrixiParticles, TrixiParticles.PointNeighbors, OrdinaryDiffEq

fluid_particle_spacing = 0.0085

smoothing_length = 1.414216 * fluid_particle_spacing
tank_size = (1.6, 0.665, 0.4)
initial_fluid_size = (0.4, 0.665, 0.3)
acceleration = (0.0, 0.0, -9.81)
spacing_ratio = 1
boundary_layers = 1
fluid_density = 1000.0
sound_speed = 20 * sqrt(9.81 * tank_size[2])
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       coordinates_eltype=Float64)

tank.fluid.coordinates .+= 0.005
tank.boundary.coordinates .+= 0.005

# Define a GPU-compatible neighborhood search
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)
cell_list = FullGridCellList(; min_corner, max_corner)#, backend=PointNeighbors.CompactVectorOfVectors{Int32})
neighborhood_search = GridNeighborhoodSearch{3}(; cell_list,
                                                update_strategy=ParallelUpdate())

search_radius = TrixiParticles.compact_support(WendlandC2Kernel{3}(), smoothing_length)
nhs = PointNeighbors.copy_neighborhood_search(neighborhood_search, search_radius, 0)
cell_coords(x) = PointNeighbors.cell_coords(x, nhs)
cell_index(x) = PointNeighbors. cell_index(nhs.cell_list, cell_coords(x))
coords = reinterpret(reshape, SVector{ndims(nhs), eltype(tank.fluid.coordinates)}, tank.fluid.coordinates)
sort!(coords, by=cell_index)

# Run the dam break simulation with this neighborhood search
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_3d.jl"),
              tank=tank,
              smoothing_length=1.7320508 * fluid_particle_spacing,
              time_integration_scheme=SymplecticPositionVerlet(),
              boundary_density_calculator=ContinuityDensity(),
              state_equation=state_equation,
              fluid_particle_spacing=fluid_particle_spacing,
              tank_size=tank_size, initial_fluid_size=initial_fluid_size,
              acceleration=acceleration,
              alpha=0.1,
              spacing_ratio=spacing_ratio, boundary_layers=boundary_layers,
              tspan=(0.0, 1.0), #cfl=0.2,
              neighborhood_search=neighborhood_search,
            #   viscosity_wall=viscosity_fluid, TODO
              # This is the same saving frequency as in DualSPHysics for easier comparison
            #   saving_callback=SolutionSavingCallback(dt=0.01),
              extra_callback=SortingCallback(interval=1),
              density_diffusion=nothing, # TODO only for benchmark
              # For benchmarks, use spacing 0.002, fix time steps, and disable VTK saving:
              dt=8e-5, #stepsize_callback=nothing, saving_callback=nothing,
              parallelization_backend=PolyesterBackend())
