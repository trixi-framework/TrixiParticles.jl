using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.00001 # x75 visc
# fluid_particle_spacing = 0.0001 # no improvement
boundary_particle_spacing = 0.00002

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 0.02)
fluid_density = 1000.0
sound_speed = 80

# Boundary geometry and initial fluid particle positions
plate_length = 0.01
plate_width = 0.01

sphere_radius = 0.0005
#terminal_velocity = 75 * 5.8 # match Re
terminal_velocity = 5.8 # real
#terminal_velocity = 1.0

boundary_thickness = 4 * boundary_particle_spacing

# ground floor
plate_size = (plate_length, plate_width, boundary_thickness)

plate = RectangularShape(boundary_particle_spacing,
                        round.(Int, plate_size ./ boundary_particle_spacing),
                        (0.0, 0.0, 0.0), density=fluid_density)

sphere1_center = (0.005, 0.005, sphere_radius + boundary_thickness)
sphere1 = SphereShape(fluid_particle_spacing, sphere_radius, sphere1_center,
                        fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, 0.0, -terminal_velocity))

# box = RectangularTank(fluid_particle_spacing, (0.3, 0.125, 0.0), (0.35, 0.075, 0.2), fluid_density,
# n_layers=8, spacing_ratio=spacing_ratio, acceleration=(0.0, -gravity, 0.0), state_equation=state_equation,
# faces=(false, false, true, false, true, true),
# min_coordinates=(tank_size[1], 0.0, 0.0))

# # move to the end
# for i in axes(end_tank.boundary.coordinates, 2)
#     end_tank.boundary.coordinates[:, i] .+= [tank_size[1], 0.0]
# end

#tank = union(tank, tank_end, tank_side_mz, tank_side_pz, tank_side_mz_end, tank_side_pz_end)

# ==========================================================================================
# ==== Fluid
smoothing_length = 3.5 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{3}()

fluid_density_calculator = ContinuityDensity()
# viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
# kinematic viscosity of water at 20C
nu_water = 8.9E-7

# Morris has a higher velocity with same viscosity.
# viscosity = ViscosityMorris(nu=75*nu_water)
viscosity = ViscosityAdami(nu=nu_water)

density_diffusion = DensityDiffusionAntuono(sphere1, delta=0.1)

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

fluid_system = WeaklyCompressibleSPHSystem(sphere1, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, 0.0, -gravity),
                                           surface_tension=SurfaceTensionMorris(surface_tension_coefficient=0.0728,
                                           free_surface_threshold=0.6), # 0.55 too many # 0.8 even more
                                           reference_particle_spacing=fluid_particle_spacing,
                                           surface_normal_method=ColorfieldSurfaceNormal(smoothing_kernel,
                                           smoothing_length, boundary_contact_threshold=1.0))


# fluid_system = WeaklyCompressibleSPHSystem(sphere1, fluid_density_calculator,
#                                            state_equation, smoothing_kernel,
#                                            smoothing_length, viscosity=viscosity,
#                                            density_diffusion=density_diffusion,
#                                            acceleration=(0.0, 0.0, -gravity))

# fluid_system = EntropicallyDampedSPHSystem(sphere1, smoothing_kernel,
#                                            smoothing_length,
#                                            sound_speed,
#                                            viscosity=viscosity,
#                                            density_calculator=fluid_density_calculator,
#                                            reference_particle_spacing=fluid_particle_spacing,
#                                            acceleration=(0.0, 0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(plate.density, plate.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             viscosity=viscosity)

boundary_system = BoundarySPHSystem(plate, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.00005, prefix="rainfall_morris_alpha001_h10micro")
callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-4, # Limit stepsize to prevent crashing
            dt=1e-8,
            save_everystep=false, callback=callbacks);
