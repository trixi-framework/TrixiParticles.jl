# Modify the 01_DamBreak example of DualSPHysics like this:
# <parameter key="StepAlgorithm" value="2" comment="Step Algorithm 1:Verlet, 2:Symplectic (default=1)" />
# <parameter key="Kernel" value="2" comment="Interaction Kernel 1:Cubic Spline, 2:Wendland (default=2)" />
# <parameter key="DensityDT" value="1" comment="Density Diffusion Term 0:None, 1:Molteni, 2:Fourtakas, 3:Fourtakas(full) (default=0)" />
# <parameter key="TimeMax" value="1.0" comment="Time of simulation" units_comment="seconds" />
# and remove the <setmkvoid /> through <shapeout file="Building"/> block
#
# When comparing with high resolution, change the resolution here:
# <definition dp="0.002" units_comment="metres (m)">
# With this resolution, use:
# <parameter key="DtFixed" value="1e-5" comment="Fixed Dt value. Use 0 to disable (default=disabled)" units_comment="seconds" />

using TrixiParticles, TrixiParticles.PointNeighbors, OrdinaryDiffEqSymplecticRK

fluid_particle_spacing = 0.0085

use_dualsphysics_nhs = true
if use_dualsphysics_nhs
    cell_list_backend = PointNeighbors.CompactVectorOfVectors{Int32}
    time_integration_scheme = SymplecticPositionVerletWithSorting()
else
    cell_list_backend = PointNeighbors.DynamicVectorOfVectors{Int32}
    time_integration_scheme = SymplecticPositionVerlet()
end

smoothing_length = 1.7320508 * fluid_particle_spacing
tank_size = (1.6 - fluid_particle_spacing, 0.67 - fluid_particle_spacing, 0.4)
tspan = (0.0, 0.1)
initial_fluid_size = (0.4, 0.67 - fluid_particle_spacing, 0.3)
acceleration = (0.0, 0.0, -9.81)
spacing_ratio = 1
boundary_layers = 1
fluid_density = 1000.0
sound_speed = 20 * sqrt(9.81 * (initial_fluid_size[3] - fluid_particle_spacing))
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density;
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       coordinates_eltype=Float64, acceleration, state_equation,
                       faces = (true, true, true, true, true, false))

# TrixiParticles initializes a hydrostatic density field combined with the corresponding
# particle masses, whereas DualSPHysics uses constant particle masses.
tank.fluid.mass .= fluid_density * fluid_particle_spacing^3
tank.boundary.mass .= fluid_density * fluid_particle_spacing^3

tank.fluid.coordinates .+= 0.005
tank.boundary.coordinates .+= 0.005

# Run the dam break simulation with this neighborhood search
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_3d.jl"),
              tank=tank,
              smoothing_length=1.7320508 * fluid_particle_spacing,
              boundary_density_calculator=ContinuityDensity(),
              state_equation=state_equation,
              fluid_particle_spacing=fluid_particle_spacing,
              tank_size=tank_size, initial_fluid_size=initial_fluid_size,
              coordinates_eltype=Float64,
              acceleration=acceleration,
              alpha=0.1,
              spacing_ratio=spacing_ratio, boundary_layers=boundary_layers,
              tspan=tspan,
              semi=nothing, ode=nothing, sol=nothing)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel, smoothing_length,
                                           density_calculator=fluid_density_calculator,
                                           state_equation, viscosity, density_diffusion,
                                           acceleration, buffer_size=0)

# Re-create boundary model to use no-slip BC.
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length;
                                             state_equation, viscosity,
                                             clip_negative_pressure=true)
boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

# Define a GPU-compatible neighborhood search
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)
cell_list = FullGridCellList(; min_corner, max_corner, backend=cell_list_backend)
neighborhood_search = GridNeighborhoodSearch{3}(; cell_list,
                                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, boundary_system; neighborhood_search,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.1, prefix="")
sorting_callback = SortingCallback(interval=1000)
callbacks = CallbackSet(info_callback, saving_callback, sorting_callback)

fluid_dt = 8e-5
sol = solve(ode, time_integration_scheme,
            dt=fluid_dt, save_everystep=false, callback=callbacks);
