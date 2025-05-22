# ==========================================================================================
# 2D Falling Water Spheres Simulation (With and Without Surface Tension)
#
# This example simulates two circular water "spheres" falling under gravity.
# One sphere includes a surface tension model (Akinci et al.), while the other does not.
# This demonstrates the effect of surface tension on fluid behavior.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Resolution Parameters
# ------------------------------------------------------------------------------

# Particle spacing, determines the resolution of the simulation
particle_spacing = 0.005

# Boundary particle layers and spacing ratio for the tank
boundary_layers = 3
spacing_ratio = 1

# ------------------------------------------------------------------------------
# Experiment Setup
# ------------------------------------------------------------------------------
# Gravitational acceleration
gravity = -9.81
gravity_vec = (0.0, gravity)

# Simulation time span
tspan = (0.0, 0.3)

# Tank dimensions (acts as a floor and side walls, open top)
# Initial fluid size is (0,0) as fluid is defined by spheres, not a bulk volume.
tank_initial_fluid_size = (0.0, 0.0)
tank_dims = (2.0, 0.5) # width, height

# Fluid properties
fluid_density = 1000.0 # Reference density of water (kg/m^3)
sound_speed = 100.0    # Speed of sound (m/s)

# Equation of state for the fluid (weakly compressible)
# Exponent 1 is often used with surface tension models or for less "bouncy" behavior.
state_equation = StateEquationCole(sound_speed=sound_speed,
                                   reference_density=fluid_density,
                                   exponent=1)

# Setup for the rectangular tank (floor and walls)
tank = RectangularTank(particle_spacing, tank_initial_fluid_size, tank_dims,
                       fluid_density,
                       n_layers=boundary_layers,
                       spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false), # (left, right, bottom, top)
                       acceleration=gravity_vec,
                       state_equation=state_equation)

# Define the two water spheres
sphere_radius = 0.05
initial_velocity_spheres = (0.0, -3.0) # Falling downwards

sphere1_center = (0.5, 0.2) # Center coordinates of the first sphere
sphere2_center = (1.5, 0.2) # Center coordinates of the second sphere

# Create sphere shapes. `VoxelSphere` creates a discretized sphere.
sphere1_particles = SphereShape(particle_spacing, sphere_radius, sphere1_center,
                                fluid_density, sphere_type=VoxelSphere(),
                                velocity=initial_velocity_spheres)
sphere2_particles = SphereShape(particle_spacing, sphere_radius, sphere2_center,
                                fluid_density, sphere_type=VoxelSphere(),
                                velocity=initial_velocity_spheres)

# ------------------------------------------------------------------------------
# Fluid Systems Setup
# ------------------------------------------------------------------------------

# Using a smoothing_length of exactly 1.0 * fluid_particle is necessary for this model to be accurate.
# This yields some numerical issues though which can be circumvented by subtracting eps().
smoothing_length = 1.0 * particle_spacing - eps()
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()

# Physical kinematic viscosity (nu()
physical_nu = 0.005

# way to calculate alpha from physical viscosity as provided by Monaghan
alpha = 8 * physical_nu / (smoothing_length * sound_speed)
viscosity_model = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)

# Density diffusion model used for the second sphere
density_diffusion_model = DensityDiffusionAntuono(sphere2_particles, delta=0.1)

# System 1: Water sphere WITH surface tension (using Entropically Damped SPH)
surface_tension_model_akinci = SurfaceTensionAkinci(surface_tension_coefficient=0.05)
sphere1_system_surftens = EntropicallyDampedSPHSystem(sphere1_particles,
                                                      smoothing_kernel,
                                                      smoothing_length,
                                                      sound_speed,
                                                      viscosity=viscosity_model,
                                                      density_calculator=ContinuityDensity(),
                                                      acceleration=gravity_vec,
                                                      surface_tension=surface_tension_model_akinci,
                                                      reference_particle_spacing=particle_spacing)

# System 2: Water sphere WITHOUT surface tension (using standard Weakly Compressible SPH)
sphere2_system_nosurftens = WeaklyCompressibleSPHSystem(sphere2_particles,
                                                        fluid_density_calculator,
                                                        state_equation,
                                                        smoothing_kernel,
                                                        smoothing_length,
                                                        viscosity=viscosity_model,
                                                        density_diffusion=density_diffusion_model,
                                                        acceleration=gravity_vec,
                                                        reference_particle_spacing=particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup
# ------------------------------------------------------------------------------
boundary_density_calculator = AdamiPressureExtrapolation()

# Viscosity for wall-fluid interaction (Adami boundary viscosity)
wall_kinematic_viscosity = physical_nu
boundary_viscosity_model = ViscosityAdami(nu=wall_kinematic_viscosity)

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel,
                                             smoothing_length,
                                             viscosity=boundary_viscosity_model,
                                             reference_particle_spacing=particle_spacing)

# Adhesion coefficient can model how much fluid "sticks" to the boundary.
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model,
                                    adhesion_coefficient=1.0)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------

semi = Semidiscretization(sphere1_system_surftens, sphere2_system_nosurftens,
                          boundary_system,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01, output_directory="out",
                                         prefix="", write_meta_data=true)

callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-7, # Absolute tolerance (default: 1e-6)
            reltol=1e-4, # Relative tolerance (default: 1e-3)
            save_everystep=false,
            callback=callbacks)
