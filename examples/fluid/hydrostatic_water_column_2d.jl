using TrixiParticles
using OrdinaryDiffEq
using P4estTypes, PointNeighbors
using MPI; MPI.Init()

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.3

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (1.0, 1.0)
tank_size = (1.0, 1.0)

fluid_density = 1000.0
sound_speed = 10.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, acceleration=(0.0, -gravity),
                       state_equation=state_equation)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

alpha = 0.02
viscosity = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)

fluid_density_calculator = ContinuityDensity()

# This is to set acceleration with `trixi_include`
system_acceleration = (0.0, -gravity)
fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                        #    acceleration=system_acceleration,
                                           source_terms=nothing)

fluid_system2 = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                        #    acceleration=system_acceleration,
                                           source_terms=nothing)
ghost_system = TrixiParticles.GhostSystem(fluid_system2)

# ==========================================================================================
# ==== Boundary

# This is to set another boundary density calculation with `trixi_include`
boundary_density_calculator = AdamiPressureExtrapolation()

# This is to set wall viscosity with `trixi_include`
# viscosity_wall = nothing
# boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
#                                              state_equation=state_equation,
#                                              boundary_density_calculator,
#                                              smoothing_kernel, smoothing_length,
#                                              viscosity=viscosity_wall)
k_solid = 0.2
beta_solid = 1
boundary_model = BoundaryModelMonaghanKajtar(k_solid, beta_solid,
                                                   fluid_particle_spacing,
                                                   tank.boundary.mass)
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model, movement=nothing)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(tank.fluid.coordinates, dims = 2) .- 2 * smoothing_length .- 0.2
max_corner = maximum(tank.fluid.coordinates, dims = 2) .+ 2 * smoothing_length .- 0.2

semi = Semidiscretization(fluid_system, ghost_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(cell_list=PointNeighbors.P4estCellList(; min_corner, max_corner)))
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="$(MPI.Comm_rank(MPI.COMM_WORLD))")

# This is to easily add a new callback with `trixi_include`
extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, extra_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks);
