# Modify the 01_DamBreak example of DualSPHysics like this:
# <parameter key="StepAlgorithm" value="2" comment="Step Algorithm 1:Verlet, 2:Symplectic (default=1)" />
# <parameter key="DensityDT" value="1" comment="Density Diffusion Term 0:None, 1:Molteni, 2:Fourtakas, 3:Fourtakas(full) (default=0)" />
# <parameter key="TimeMax" value="1.0" comment="Time of simulation" units_comment="seconds" />
#
# When comparing with high resolution, change the resolution here:
# <definition dp="0.002" units_comment="metres (m)">
# With this resolution, use:
# <parameter key="DtFixed" value="1e-5" comment="Fixed Dt value. Use 0 to disable (default=disabled)" units_comment="seconds" />

using TrixiParticles, TrixiParticles.PointNeighbors

fluid_particle_spacing = 0.002

# Load setup from dam break example
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=fluid_particle_spacing,
              smoothing_length=1.414216 * fluid_particle_spacing,
              tank_size=(4.0, 3.0), W=1.0, H=2.0,
              spacing_ratio=1, boundary_layers=1,
              sol=nothing, ode=nothing)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       coordinates_eltype=Float64)

tank.fluid.coordinates .+= 0.005
tank.boundary.coordinates .+= 0.005

# Define a GPU-compatible neighborhood search
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)
cell_list = FullGridCellList(; min_corner, max_corner, backend=PointNeighbors.CompactVectorOfVectors{Int32})
neighborhood_search = GridNeighborhoodSearch{2}(; cell_list,
                                                update_strategy=ParallelUpdate())

search_radius = TrixiParticles.get_neighborhood_search(fluid_system, fluid_system, semi).search_radius
nhs = PointNeighbors.copy_neighborhood_search(neighborhood_search, search_radius, 0)
cell_coords(x) = PointNeighbors.cell_coords(x, nhs)
cell_index(x) = PointNeighbors. cell_index(nhs.cell_list, cell_coords(x))
coords = reinterpret(reshape, SVector{ndims(fluid_system), eltype(tank.fluid.coordinates)}, tank.fluid.coordinates)
sort!(coords, by=cell_index)

# function cells(coordinates, system, semi)
#     nhs = TrixiParticles.get_neighborhood_search(fluid_system, fluid_system, semi)
#     coords = reinterpret(reshape, SVector{ndims(system), eltype(coordinates)}, coordinates)
#     return TrixiParticles.PointNeighbors.cell_coords.(coords, Ref(nhs))
# end

# For benchmarking, run:
# trixi_include_changeprecision(Float32, "../TrixiParticles.jl/examples/fluid/dam_break_2d_dualsphysics.jl", parallelization_backend=CUDABackend(), tspan=(0.0f0, 1.0f-10), fluid_particle_spacing=0.001, coordinates_eltype=Float32);
# dv_ode, du_ode = copy(sol.u[end]).x; v_ode, u_ode = copy(sol.u[end]).x; semi = ode.p; system = semi.systems[1]; dv = TrixiParticles.wrap_v(dv_ode, system, semi); v = TrixiParticles.wrap_v(v_ode, system, semi); u = TrixiParticles.wrap_u(u_ode, system, semi);
# @benchmark TrixiParticles.interact_reassembled!($dv, $v, $u, $v, $u, $system, $system, $semi)

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
              tspan=(0.0, 1.0), cfl=0.2,
              neighborhood_search=neighborhood_search,
              viscosity_wall=viscosity_fluid,
              # This is the same saving frequency as in DualSPHysics for easier comparison
              saving_callback=SolutionSavingCallback(dt=0.01),
            #   extra_callback=SortingCallback(interval=1),
              density_diffusion=nothing, # TODO only for benchmark
              # For benchmarks, use spacing 0.002, fix time steps, and disable VTK saving:
              dt=1e-5, stepsize_callback=nothing, #saving_callback=nothing,
              parallelization_backend=PolyesterBackend())
