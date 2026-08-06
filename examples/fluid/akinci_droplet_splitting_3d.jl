# ==========================================================================================
# 3D Akinci Adhesive-Box Droplet Splitting
#
# Reproduces Figure 9 of Akinci, Akinci, and Teschner (2013): a drop adheres strongly to a
# box while a descending blade with zero adhesion splits it. The contrast between the two
# boundary adhesion coefficients is the central feature of the experiment.
# https://doi.org/10.1145/2508363.2508395
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

particle_spacing = 0.01
fluid_density = 1000.0
sound_speed = 30.0
gravity = 9.81
tspan = (0.0, 0.35)
solution_saveat = ()

tank_size = (0.24, 0.18, 0.14)
drop_radius = 0.05
drop_center = (tank_size[1] / 2, tank_size[2] / 2,
               drop_radius + particle_spacing / 2)
boundary_layers = 3
blade_speed = 0.5
blade_motion_time = 0.25

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)
tank = RectangularTank(particle_spacing, (0.0, 0.0, 0.0), tank_size, fluid_density;
                       n_layers=boundary_layers,
                       faces=(true, true, true, true, true, false))
drop = SphereShape(particle_spacing, drop_radius, drop_center, fluid_density;
                   sphere_type=VoxelSphere())

blade_height = 0.08
blade_width = 0.12
n_blade = (3, round(Int, blade_width / particle_spacing),
           round(Int, blade_height / particle_spacing))
blade = RectangularShape(particle_spacing, n_blade,
                         (tank_size[1] / 2 - 1.5 * particle_spacing,
                          tank_size[2] / 2 - blade_width / 2,
                          drop_center[3] + drop_radius + 0.03);
                         density=fluid_density)

blade_motion = let blade_speed=blade_speed, blade_motion_time=blade_motion_time
    (x, t) -> x + SVector(0.0, 0.0, -blade_speed * min(t, blade_motion_time))
end
blade_is_moving = let blade_motion_time=blade_motion_time
    t -> t < blade_motion_time
end
prescribed_blade_motion = PrescribedMotion(blade_motion, blade_is_moving)

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

tank_boundary_model = BoundaryModelDummyParticles(tank.boundary; fluid_system,
                                                  boundary_density_calculator=AdamiPressureExtrapolation(),
                                                  viscosity=ViscosityAdami(nu=0.01),
                                                  clip_negative_pressure=true)
blade_boundary_model = BoundaryModelDummyParticles(blade; fluid_system,
                                                   boundary_density_calculator=AdamiPressureExtrapolation(),
                                                   viscosity=ViscosityAdami(nu=0.01),
                                                   clip_negative_pressure=true)

tank_boundary_system = WallBoundarySystem(tank.boundary, tank_boundary_model;
                                          adhesion_coefficient=2.0)
blade_boundary_system = WallBoundarySystem(blade, blade_boundary_model;
                                           prescribed_motion=prescribed_blade_motion,
                                           adhesion_coefficient=0.0)

semi = Semidiscretization(fluid_system, tank_boundary_system, blade_boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01)
callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL35(); abstol=1e-7, reltol=1e-4, dtmax=1e-3,
            save_everystep=false, saveat=solution_saveat, callback=callbacks)
