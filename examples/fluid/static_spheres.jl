# In this example two circles of water drop to the floor demonstrating the difference
# between the behavior with and without surface tension modelling.
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.005

boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 0.0
tspan = (0.0, 0.3)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.0, 0.0)
tank_size = (2.5, 0.5)

fluid_density = 1000.0
sound_speed = 100
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

sphere_radius = 0.05

sphere1_center = (0.5, 0.5)
sphere2_center = (1.5, 0.5)
sphere1 = SphereShape(fluid_particle_spacing, sphere_radius, sphere1_center,
                      fluid_density, sphere_type=RoundSphere(), velocity=(0.0, 0.0))
sphere2 = SphereShape(fluid_particle_spacing, sphere_radius, sphere2_center,
                      fluid_density, sphere_type=RoundSphere(), velocity=(0.0, 0.0))

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 3.5 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

nu = 0.005
alpha = 8 * nu / (fluid_smoothing_length * sound_speed)
viscosity = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)
# density_diffusion = DensityDiffusionAntuono(sphere2, delta=0.1)

sphere_surface_tension = EntropicallyDampedSPHSystem(sphere1, fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     sound_speed, viscosity=viscosity,
                                                     density_calculator=ContinuityDensity(),
                                                     reference_particle_spacing=fluid_particle_spacing,
                                                     acceleration=(0.0, -gravity),
                                                     surface_tension=SurfaceTensionMorris(surface_tension_coefficient=0.0728))

sphere = EntropicallyDampedSPHSystem(sphere2, fluid_smoothing_kernel,
                                     fluid_smoothing_length,
                                     sound_speed, viscosity=viscosity,
                                     density_calculator=ContinuityDensity(),
                                     reference_particle_spacing=fluid_particle_spacing,
                                     acceleration=(0.0, -gravity),
                                     surface_normal_method=ColorfieldSurfaceNormal(fluid_smoothing_kernel,
                                                                                   fluid_smoothing_length))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
wall_viscosity = nu
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length,
                                             viscosity=ViscosityAdami(nu=wall_viscosity))

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(sphere_surface_tension, sphere, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.01, output_directory="out",
                                         prefix="static", write_meta_data=true)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-7, # Default abstol is 1e-6
            reltol=1e-4, # Default reltol is 1e-3
            dt=1e-6,
            save_everystep=false, callback=callbacks);
