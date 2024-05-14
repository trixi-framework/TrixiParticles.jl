# In this example two circles of water drop to the floor demonstrating the difference
# between the behavior with and without surface tension modelling.
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.001

boundary_layers = 5
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 1.5)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.0, 0.0)
tank_size = (2.0, 2.0)

fluid_density = 1000.0
sound_speed = 150
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

sphere_radius = 0.05

sphere1_center = (0.5, sphere_radius + 0.5 * fluid_particle_spacing)
sphere2_center = (1.5, sphere_radius - 0.5 * fluid_particle_spacing)
sphere1 = SphereShape(fluid_particle_spacing, sphere_radius, sphere1_center,
                      fluid_density, sphere_type=VoxelSphere())
sphere2 = SphereShape(fluid_particle_spacing, sphere_radius, sphere2_center,
                      fluid_density, sphere_type=VoxelSphere())

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 1.0 * fluid_particle_spacing
fluid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_smoothing_length_2 = 3.0 * fluid_particle_spacing
fluid_smoothing_kernel_2 = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

# water at 20C
nu = 0.00089

alpha = 8 * nu / (fluid_smoothing_length * sound_speed)
viscosity = ViscosityAdami(nu=nu)
density_diffusion = DensityDiffusionAntuono(sphere2, delta=0.1)

sphere_surface_tension = WeaklyCompressibleSPHSystem(sphere1, fluid_density_calculator,
                                                     state_equation, fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     viscosity=viscosity,
                                                     acceleration=(0.0, -gravity),
                                                     surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.05),
                                                     correction=AkinciFreeSurfaceCorrection(fluid_density))

sphere = WeaklyCompressibleSPHSystem(sphere2, fluid_density_calculator,
                                     state_equation, fluid_smoothing_kernel_2,
                                     fluid_smoothing_length_2, viscosity=viscosity,
                                     density_diffusion=density_diffusion,
                                     acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel_2,
                                             fluid_smoothing_length_2,
                                             viscosity=ViscosityAdami(nu=nu))

# adhesion_coefficient = 1.0 and surface_tension_coefficient=0.01 for perfect wetting
# adhesion_coefficient = 0.001 and surface_tension_coefficient=2.0 for no wetting
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model, adhesion_coefficient=1.5)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(boundary_system, sphere_surface_tension)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.01, output_directory="out", prefix="",
                                         write_meta_data=true)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6
            reltol=1e-4, # Default reltol is 1e-3
            dt=1e-5,
            save_everystep=false, callback=callbacks);
