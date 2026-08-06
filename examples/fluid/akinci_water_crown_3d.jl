# ==========================================================================================
# 3D Akinci Water-Crown Experiment
#
# Reproduces the experiment in Figures 1 and 5 of Akinci, Akinci, and Teschner (2013):
# a fast drop impacts a shallow pool and produces a crown and secondary droplets. The
# dimensions and drop volume match the paper; the particle count is reduced from one million.
# https://doi.org/10.1145/2508363.2508395
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

particle_spacing = 0.0025
fluid_density = 1000.0
sound_speed = 40.0
gravity = 9.81
tspan = (0.0, 0.12)
solution_saveat = ()

# The paper reports a 6.5 cm^3 drop and a 15 x 4 x 15 cm^3 filled container.
pool_size = (0.15, 0.15, 0.04)
tank_size = (0.15, 0.15, 0.1)
drop_volume = 6.5e-6
drop_radius = cbrt(3 * drop_volume / (4pi))
# `VoxelSphere` otherwise moves the outer particle centers half a spacing inwards. Compensate
# for that inset so the discretized volume remains close to the reported volume.
drop_sampling_radius = drop_radius + particle_spacing / 2
drop_center = (tank_size[1] / 2, tank_size[2] / 2, 0.075)
# The impact speed is not reported. This value reproduces the crown at reduced resolution.
impact_velocity = 2.0
boundary_layers = 3

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)

tank = RectangularTank(particle_spacing, pool_size, tank_size, fluid_density;
                       n_layers=boundary_layers,
                       faces=(true, true, true, true, true, false),
                       acceleration=(0.0, 0.0, -gravity), state_equation)
drop = SphereShape(particle_spacing, drop_sampling_radius, drop_center, fluid_density;
                   sphere_type=VoxelSphere(), velocity=(0.0, 0.0, -impact_velocity))
fluid = union(tank.fluid, drop)

smoothing_length = particle_spacing - eps()
smoothing_kernel = SchoenbergCubicSplineKernel{3}()
density_calculator = SummationDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.001, beta=0.0)
surface_tension_coefficient = 1.0
surface_tension = SurfaceTensionAkinci(; surface_tension_coefficient)

fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           density_calculator,
                                           state_equation, viscosity, surface_tension,
                                           correction=AkinciFreeSurfaceCorrection(fluid_density),
                                           reference_particle_spacing=particle_spacing,
                                           acceleration=(0.0, 0.0, -gravity))

boundary_model = BoundaryModelDummyParticles(tank.boundary; fluid_system,
                                             boundary_density_calculator=AdamiPressureExtrapolation(),
                                             viscosity,
                                             clip_negative_pressure=true)
boundary_system = WallBoundarySystem(tank.boundary, boundary_model;
                                     adhesion_coefficient=1.0)

semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01)
callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL35(); abstol=1e-7, reltol=1e-4, dtmax=1e-3,
            save_everystep=false, saveat=solution_saveat, callback=callbacks)
