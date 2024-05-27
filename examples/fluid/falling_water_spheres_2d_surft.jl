# In this example two circles of water drop to the floor demonstrating the difference
# between the behavior with and without surface tension modelling.
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.0001

boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.0, 0.0)
tank_size = (1.0, 0.5)

fluid_density = 1000.0
sound_speed = 100
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)


box = RectangularShape(fluid_particle_spacing, (300, 125), (0.485, 0.0), density=fluid_density)

# box = SphereShape(fluid_particle_spacing, 1.0, (0.5, 0.0),
#                       fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, 0.0))

sphere_radius = 0.0025

sphere1_center = (0.5, 0.05)
sphere2_center = (0.5, 0.1)
sphere3_center = (0.5, 0.15)
sphere4_center = (0.5, 0.2)
sphere1 = SphereShape(fluid_particle_spacing, sphere_radius, sphere1_center,
                      fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, -3.0))
sphere2 = SphereShape(fluid_particle_spacing, sphere_radius, sphere2_center,
                      fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, -3.0))
sphere3 = SphereShape(fluid_particle_spacing, sphere_radius, sphere3_center,
                      fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, -3.0))
sphere4 = SphereShape(fluid_particle_spacing, sphere_radius, sphere4_center,
                      fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, -3.0))

water = union(sphere1, sphere2, sphere3, sphere4)

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 2.5 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

nu = 0.00089
alpha = 0.75*8 * nu / (fluid_smoothing_length * sound_speed)
# viscosity = ViscosityAdami(nu=nu)
viscosity = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)

# density_diffusion = DensityDiffusionAntuono(sphere2, delta=0.1)

sphere_surface_tension = WeaklyCompressibleSPHSystem(water, fluid_density_calculator,
                                                     state_equation, fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     viscosity=viscosity,
                                                     acceleration=(0.0, -gravity),
                                                     surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.1*0.011),
                                                     correction=AkinciFreeSurfaceCorrection(fluid_density))

# sphere_surface_tension = WeaklyCompressibleSPHSystem(water, fluid_density_calculator,
#                                                      state_equation, fluid_smoothing_kernel,
#                                                      fluid_smoothing_length,
#                                                      viscosity=viscosity,
#                                                      acceleration=(0.0, -gravity))

# sphere_surface_tension2 = WeaklyCompressibleSPHSystem(sphere2, fluid_density_calculator,
#                                                      state_equation, fluid_smoothing_kernel,
#                                                      fluid_smoothing_length,
#                                                      viscosity=viscosity,
#                                                      acceleration=(0.0, -gravity),
#                                                      surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.011),
#                                                      correction=AkinciFreeSurfaceCorrection(fluid_density))

# sphere_surface_tension = WeaklyCompressibleSPHSystem(sphere1, fluid_density_calculator,
#                                                      state_equation, fluid_smoothing_kernel,
#                                                      fluid_smoothing_length,
#                                                      viscosity=viscosity,
#                                                      acceleration=(0.0, -gravity),
#                                                      surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.01),
#                                                      correction=AkinciFreeSurfaceCorrection(fluid_density))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length,
                                             viscosity=ViscosityAdami(nu=0.5*nu))

boundary_model_box = BoundaryModelDummyParticles(box.density, box.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length,
                                             viscosity=ViscosityAdami(nu=0.5*nu))

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model,
                                    adhesion_coefficient=0.01)

boundary_system2 = BoundarySPHSystem(box, boundary_model_box,
                                    adhesion_coefficient=0.01)

# boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# boundary_system2 = BoundarySPHSystem(box, boundary_model_box)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(boundary_system, boundary_system2, sphere_surface_tension)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.0025, output_directory="out", prefix="four_drop_terminal_m20_90d_01surft_05wallnu_075viscmon_001adh",
                                         write_meta_data=true)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6
            reltol=1e-3, # Default reltol is 1e-3
            save_everystep=false, callback=callbacks);
