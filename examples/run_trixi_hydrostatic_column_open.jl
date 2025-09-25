using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
# fluid_particle_spacing = 0.08
fluid_particle_spacing = 0.05

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
# tspan = (0.0, 0.2)
tspan = (0.0, 1.0)
# tspan = (0.0, 5.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (1.0, 0.9)
tank_size = (1.0, 1.0)

fluid_density = 1000.0
sound_speed = 40.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)
tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, acceleration=(0.0, -gravity),
                       state_equation=state_equation, faces=(true, true, false, true))
# ==========================================================================================
# ==== Fluid
smoothing_length = 3.5 * fluid_particle_spacing / 2
# smoothing_kernel = SchoenbergCubicSplineKernel{2}()
smoothing_kernel = TrixiParticles.WendlandC2Kernel{2}()

alpha = 0.02
viscosity = TrixiParticles.ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)
fluid_density_calculator = ContinuityDensity()
density_diffusion = TrixiParticles.DensityDiffusionMolteniColagrossi(delta=0.1)

# This is to set acceleration with `trixi_include`
system_acceleration = (0.0, -gravity)
# old_density = deepcopy(tank.fluid.density) # FIXME: Fluid density has to be constant in order for buffer to work. 
# tank.fluid.density .= tank.fluid.density[end]
fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=system_acceleration,
                                           buffer_size=10000,
                                           source_terms=nothing)
# tank.fluid.density .= old_density # FIXME: Bypassed the constant-density check? 
# ==========================================================================================
# ==== Boundary

# This is to set another boundary density calculation with `trixi_include`
boundary_density_calculator = AdamiPressureExtrapolation()

# This is to set wall viscosity with `trixi_include`
viscosity_wall = nothing
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             viscosity=viscosity_wall)
boundary_system = WallBoundarySystem(tank.boundary, boundary_model,
                                     prescribed_motion=nothing)

# Open boundary
open_boundary_model = BoundaryModelCharacteristicsLastiwka(;
                                                           extrapolate_reference_values=nothing)
boundary_face_width = floor(1/fluid_particle_spacing) * fluid_particle_spacing
boundary_y = boundary_layers * fluid_particle_spacing
boundary_face = ([0.0, boundary_y], [boundary_face_width, boundary_y])
face_normal = [0.0, 1.0]
boundary_zone = BoundaryZone(; boundary_face=boundary_face, face_normal, density=1000.0,
                             particle_spacing=fluid_particle_spacing,
                             initial_condition=nothing, extrude_geometry=nothing,
                             open_boundary_layers=4, average_inflow_velocity=true,
                             boundary_type=OutFlow(),
                             reference_density=1000.0, reference_pressure=nothing,
                             reference_velocity=nothing) # TODO: Figure out what/how to assign reference values? Hydrostatic?

open_boundary_system = OpenBoundarySystem(boundary_zone, fluid_system=fluid_system,
                                          buffer_size=10000,
                                          boundary_model=open_boundary_model)
# ==========================================================================================

# Neighborhood search
nhs = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate()) # TODO: Find out if serial update is the only way to do this with a buffer?

# ================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system, open_boundary_system,
                          neighborhood_search=nhs)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
update_callback = UpdateCallback(interval=1)
saving_callback = SolutionSavingCallback(dt=5e-3,
                                         output_directory=joinpath(@__DIR__, "out_trixi"),
                                         prefix="vis_")

# This is to easily add a new callback with `trixi_include`

callbacks = CallbackSet(update_callback, info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# sol = solve(ode, dt=5e-4, adaptive=false, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks)
sol = solve(ode, RDPK3SpFSAL35(), dt=3e-3, adaptive=false, save_everystep=false,
            callback=callbacks);
