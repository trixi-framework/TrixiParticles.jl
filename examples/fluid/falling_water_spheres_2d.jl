# ==========================================================================================
# 2D Falling Water Spheres Simulation (With and Without Surface Tension)
#
# This example simulates two circular water "spheres" falling under gravity.
# One sphere includes a surface tension model (Akinci et al.), while the other does not.
# This demonstrates the effect of surface tension on fluid behavior.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.005

boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 0.3)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.0, 0.0)
tank_size = (2.0, 0.5)

fluid_density = 1000.0
sound_speed = 100
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

sphere_radius = 0.05

sphere1_center = (0.5, 0.2)
sphere2_center = (1.5, 0.2)
sphere1 = SphereShape(fluid_particle_spacing, sphere_radius, sphere1_center,
                      fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, -3.0))
sphere2 = SphereShape(fluid_particle_spacing, sphere_radius, sphere2_center,
                      fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, -3.0))

# ==========================================================================================
# ==== Fluid

# Using a smoothing_length of exactly 1.0 * fluid_particle is necessary for this model to be accurate.
# This yields some numerical issues though which can be circumvented by subtracting eps().
fluid_smoothing_length = 1.0 * fluid_particle_spacing - eps()
fluid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()

nu = 0.005
alpha = 8 * nu / (fluid_smoothing_length * sound_speed)
viscosity = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)
density_diffusion = DensityDiffusionAntuono(sphere2, delta=0.1)

sphere_surface_tension = EntropicallyDampedSPHSystem(sphere1, fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     sound_speed, viscosity=viscosity,
                                                     density_calculator=ContinuityDensity(),
                                                     acceleration=(0.0, -gravity),
                                                     surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.05),
                                                     reference_particle_spacing=fluid_particle_spacing)

sphere = WeaklyCompressibleSPHSystem(sphere2, fluid_density_calculator,
                                     state_equation, fluid_smoothing_kernel,
                                     fluid_smoothing_length, viscosity=viscosity,
                                     density_diffusion=density_diffusion,
                                     acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
wall_viscosity = nu
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length,
                                             viscosity=ViscosityAdami(nu=wall_viscosity),
                                             reference_particle_spacing=fluid_particle_spacing)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model,
                                    adhesion_coefficient=1.0)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(sphere_surface_tension, sphere, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=1000)
saving_callback = SolutionSavingCallback(dt=0.01, output_directory="out",
                                         prefix="", write_meta_data=true)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-7, # Default abstol is 1e-6
            reltol=1e-4, # Default reltol is 1e-3
            save_everystep=false, callback=callbacks);
