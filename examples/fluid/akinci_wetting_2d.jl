# ==========================================================================================
# 2D Akinci Surface Tension and Wall Adhesion
#
# A circular drop rests on the bottom wall. Set `wetting=true` to compare stronger wall
# adhesion with a surface-tension-dominated non-wetting setup. The full Akinci model uses
# surface normals, while its wall interaction uses the Akinci adhesion kernel.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

particle_spacing = 0.005
fluid_density = 1000.0
sound_speed = 100.0
gravity = 9.81
tspan = (0.0, 0.3)
wetting = false

if wetting
    surface_tension_coefficient = 0.01
    adhesion_coefficient = 1.0
    nu = 0.0005
else
    surface_tension_coefficient = 2.0
    adhesion_coefficient = 0.001
    nu = 0.001
end

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(particle_spacing, (0.0, 0.0), (0.5, 0.1), fluid_density;
                       n_layers=3, faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation)
drop = SphereShape(particle_spacing, 0.05, (0.25, 0.05), fluid_density;
                   sphere_type=VoxelSphere())

smoothing_length = particle_spacing - eps()
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
alpha = 8 * nu / (smoothing_length * sound_speed)
viscosity = ArtificialViscosityMonaghan(; alpha, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(drop; smoothing_kernel, smoothing_length,
                                           density_calculator=ContinuityDensity(),
                                           state_equation, viscosity,
                                           acceleration=(0.0, -gravity),
                                           surface_tension=SurfaceTensionAkinci(;
                                                                                surface_tension_coefficient),
                                           correction=AkinciFreeSurfaceCorrection(fluid_density),
                                           reference_particle_spacing=particle_spacing)

boundary_model = BoundaryModelDummyParticles(tank.boundary; fluid_system,
                                             boundary_density_calculator=AdamiPressureExtrapolation(),
                                             viscosity=ViscosityAdami(; nu=4 * nu),
                                             clip_negative_pressure=true)
boundary_system = WallBoundarySystem(tank.boundary, boundary_model;
                                     adhesion_coefficient)

semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01)
callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL35(); abstol=1e-7, reltol=1e-4,
            save_everystep=false, callback=callbacks)
