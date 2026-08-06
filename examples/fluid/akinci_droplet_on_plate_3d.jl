# ==========================================================================================
# 3D Akinci Droplet Impact on a Plate
#
# Reproduces Figure 6 of Akinci, Akinci, and Teschner (2013): a large drop impacts a
# finite hydrophilic plate, spreads into a sheet, and can drip from its sides. The reference
# scene is scaled down while retaining its reported adhesion coefficient beta = 0.6.
# https://doi.org/10.1145/2508363.2508395
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

particle_spacing = 0.0125
fluid_density = 1000.0
sound_speed = 40.0
gravity = 9.81
tspan = (0.0, 0.45)
solution_saveat = ()

drop_radius = 0.075
drop_center = (0.0, 0.0, 0.22)
impact_velocity = 1.0
plate_size = (0.4, 0.4)
boundary_layers = 3

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)

drop = SphereShape(particle_spacing, drop_radius, drop_center, fluid_density;
                   sphere_type=VoxelSphere(), velocity=(0.0, 0.0, -impact_velocity))

n_plate = round.(Int, plate_size ./ particle_spacing)
plate = RectangularShape(particle_spacing, (n_plate..., boundary_layers),
                         (-plate_size[1] / 2, -plate_size[2] / 2,
                          -boundary_layers * particle_spacing);
                         density=fluid_density)

smoothing_length = particle_spacing - eps()
smoothing_kernel = SchoenbergCubicSplineKernel{3}()
viscosity = ArtificialViscosityMonaghan(alpha=0.01, beta=0.0)
surface_tension_coefficient = 1.0
surface_tension = SurfaceTensionAkinci(; surface_tension_coefficient)

fluid_system = WeaklyCompressibleSPHSystem(drop; smoothing_kernel, smoothing_length,
                                           density_calculator=ContinuityDensity(),
                                           state_equation, viscosity, surface_tension,
                                           correction=AkinciFreeSurfaceCorrection(fluid_density),
                                           reference_particle_spacing=particle_spacing,
                                           acceleration=(0.0, 0.0, -gravity))

boundary_model = BoundaryModelDummyParticles(plate; fluid_system,
                                             boundary_density_calculator=AdamiPressureExtrapolation(),
                                             viscosity=ViscosityAdami(nu=0.01),
                                             clip_negative_pressure=true)
boundary_system = WallBoundarySystem(plate, boundary_model; adhesion_coefficient=0.6)

semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01)
callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL35(); abstol=1e-7, reltol=1e-4, dtmax=1e-3,
            save_everystep=false, saveat=solution_saveat, callback=callbacks)
