# In this example two circles of water drop to the floor demonstrating the difference
# between the behavior with and without surface tension modelling.
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.003

boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 2.0)

fluid_density = 1000.0
sound_speed = 150
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

screw_diameter = 0.03
screw_height = 0.015

tank_height = 5 * screw_height

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (5 * screw_diameter, 5 * screw_diameter, tank_height)
tank_size = (10 * screw_diameter, 5 * screw_diameter, tank_height)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, false, true, true, true, false),
                       acceleration=(0.0, 0.0, -gravity), state_equation=state_equation)

box = RectangularShape(fluid_particle_spacing,
                       (round(Int, screw_diameter / fluid_particle_spacing),
                        round(Int, screw_diameter / fluid_particle_spacing),
                        round(Int, screw_height / fluid_particle_spacing)),
                       (7 * screw_diameter, 2 * screw_diameter, 0.0),
                       density=fluid_density)

# box = SphereShape(fluid_particle_spacing, 1.0, (0.5, 0.0),
#                       fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, 0.0))

# sphere_radius = 0.0025

# sphere1_center = (0.5, 0.25, 0.05)
# sphere2_center = (0.5, 0.25, 0.1)
# sphere3_center = (0.5, 0.25, 0.15)
# sphere4_center = (0.5, 0.25, 0.2)
# sphere1 = SphereShape(fluid_particle_spacing, sphere_radius, sphere1_center,
#                       fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, 0.0, -3.0))
# sphere2 = SphereShape(fluid_particle_spacing, sphere_radius, sphere2_center,
#                       fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, 0.0, -3.0))
# sphere3 = SphereShape(fluid_particle_spacing, sphere_radius, sphere3_center,
#                       fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, 0.0, -3.0))
# sphere4 = SphereShape(fluid_particle_spacing, sphere_radius, sphere4_center,
#                       fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, 0.0, -3.0))

# water = union(sphere1, sphere2, sphere3, sphere4)

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 1.0 * fluid_particle_spacing
fluid_smoothing_kernel = SchoenbergCubicSplineKernel{3}()

fluid_density_calculator = ContinuityDensity()

nu = 0.00089
alpha = 10 * nu / (fluid_smoothing_length * sound_speed)
# viscosity = ViscosityAdami(nu=nu)
viscosity = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)

# density_diffusion = DensityDiffusionAntuono(sphere2, delta=0.1)

# sphere_surface_tension = WeaklyCompressibleSPHSystem(water, fluid_density_calculator,
#                                                      state_equation, fluid_smoothing_kernel,
#                                                      fluid_smoothing_length,
#                                                      viscosity=viscosity,
#                                                      acceleration=(0.0, -gravity),
#                                                      surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.1 *
#                                                                                                                       0.011),
#                                                      correction=AkinciFreeSurfaceCorrection(fluid_density))

# sphere_surface_tension = EntropicallyDampedSPHSystem(tank.fluid, fluid_smoothing_kernel,
#                                                      fluid_smoothing_length,
#                                                      sound_speed, viscosity=viscosity,
#                                                      density_calculator=ContinuityDensity(),
#                                                      acceleration=(0.0, 0.0, -gravity),
#                                                      surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.5),
#                                                      surface_normal_method=AkinciSurfaceNormal(smoothing_kernel=WendlandC6Kernel{3}(),
#                                                                                                smoothing_length=4 *
#                                                                                                                 fluid_particle_spacing))
sphere_surface_tension = EntropicallyDampedSPHSystem(tank.fluid, fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     sound_speed, viscosity=viscosity,
                                                     density_calculator=ContinuityDensity(),
                                                     acceleration=(0.0, 0.0, -gravity),
                                                     surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.5))

# sphere_surface_tension = EntropicallyDampedSPHSystem(tank.fluid, fluid_smoothing_kernel,
#                                                      fluid_smoothing_length,
#                                                      sound_speed, viscosity=viscosity,
#                                                      density_calculator=ContinuityDensity(),
#                                                      acceleration=(0.0, 0.0, -gravity))

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
                                             viscosity=ViscosityAdami(nu=nu))

boundary_model_box = BoundaryModelDummyParticles(box.density, box.mass,
                                                 state_equation=state_equation,
                                                 boundary_density_calculator,
                                                 fluid_smoothing_kernel,
                                                 fluid_smoothing_length,
                                                 viscosity=ViscosityAdami(nu=nu))

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model,
                                    adhesion_coefficient=0.2)

boundary_system2 = BoundarySPHSystem(box, boundary_model_box,
                                     adhesion_coefficient=0.2)

# boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# boundary_system2 = BoundarySPHSystem(box, boundary_model_box)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(boundary_system, boundary_system2, sphere_surface_tension)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.005, output_directory="out",
                                         prefix="test_03surft_cubic_c150",
                                         write_meta_data=true)

stepsize_callback = StepsizeCallback(cfl=2.5)


callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, CKLLSRK54_3M_3R(),
            abstol=1e-5, # Default abstol is 1e-6
            reltol=1e-3, # Default reltol is 1e-3
            save_everystep=false, callback=callbacks);
# 900-950s RDPK3SpFSAL49
# 660-700s RDPK3SpFSAL35
# 510-530s CKLLSRK43_2
# 505-530s CKLLSRK54_3C
# 820-850s Tsit5
# 760-800s BS5
# 510-515s BS3
# 590-600s OwrenZen3
# 1000+s Ralston
# 1000+s Alshina3
# 530-540s CKLLSRK54_3M_3R
# CKLLSRK54_3C_3R
# CKLLSRK54_3N_3R
# CKLLSRK54_3N_4R
# CKLLSRK54_3M_4R

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt=1.0, # This is overwritten by the stepsize callback
#             save_everystep=false, callback=callbacks);
            #~157-179s to 1%
            # cfl 2.0 diverged
            # cfl 1.5 19700s to 100% 750-780s to 5%


# sol = solve(ode, Tsit5(),
#             dt=1.0, # This is overwritten by the stepsize callback
#             save_everystep=false, callback=callbacks);
            #~219-246s to 1%
            # cfl 2.0 842-850s to 5%

# sol = solve(ode, ParsaniKetchesonDeconinck3S32(),
#             dt=1.0, # This is overwritten by the stepsize callback
#             save_everystep=false, callback=callbacks);
# diverged
