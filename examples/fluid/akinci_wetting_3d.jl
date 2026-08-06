# ==========================================================================================
# 3D Akinci Wetting-Regime Experiment
#
# Reproduces Figure 8 of Akinci, Akinci, and Teschner (2013). The companion video reports a
# 1 cm^3 drop represented by about 750 particles and provides the coefficient pairs below.
# https://doi.org/10.1145/2508363.2508395
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

drop_volume = 1.0e-6
target_particle_count = 750
drop_radius = cbrt(3 * drop_volume / (4pi))
particle_spacing = cbrt(drop_volume / target_particle_count)
# Compensate for the half-spacing inset of the outer `VoxelSphere` particle centers.
drop_sampling_radius = drop_radius + particle_spacing / 2
fluid_density = 1000.0
sound_speed = 30.0
gravity = 9.81
tspan = (0.0, 0.2)
solution_saveat = ()
wetting_case = "intermediate_wetting"

if wetting_case == "no_wetting"
    surface_tension_coefficient = 1.0
    adhesion_coefficient = 0.0
elseif wetting_case == "weak_wetting"
    surface_tension_coefficient = 1.0
    adhesion_coefficient = 0.05
elseif wetting_case == "moderate_wetting"
    surface_tension_coefficient = 1.0
    adhesion_coefficient = 0.1
elseif wetting_case == "intermediate_wetting"
    surface_tension_coefficient = 1.0
    adhesion_coefficient = 0.25
elseif wetting_case == "strong_wetting"
    surface_tension_coefficient = 0.1
    adhesion_coefficient = 0.01
elseif wetting_case == "near_perfect_wetting"
    surface_tension_coefficient = 0.01
    adhesion_coefficient = 0.001
elseif wetting_case == "perfect_wetting"
    surface_tension_coefficient = 0.001
    adhesion_coefficient = 0.0
else
    throw(ArgumentError("unknown `wetting_case`: $wetting_case"))
end

drop_center = (0.0, 0.0, drop_radius + particle_spacing / 2)
plate_size = (0.03, 0.03)
boundary_layers = 3
boundary_density_calculator = AdamiPressureExtrapolation()

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)
drop = SphereShape(particle_spacing, drop_sampling_radius, drop_center, fluid_density;
                   sphere_type=VoxelSphere())
initial_fluid_density = nothing
if !isnothing(initial_fluid_density)
    drop = InitialCondition(; particle_spacing, coordinates=drop.coordinates,
                            velocity=drop.velocity, mass=drop.mass,
                            density=initial_fluid_density, pressure=drop.pressure)
end

n_plate = round.(Int, plate_size ./ particle_spacing)
plate = RectangularShape(particle_spacing, (n_plate..., boundary_layers),
                         (-plate_size[1] / 2, -plate_size[2] / 2,
                          -boundary_layers * particle_spacing);
                         density=fluid_density)

smoothing_length = particle_spacing - eps()
smoothing_kernel = SchoenbergCubicSplineKernel{3}()
viscosity = ArtificialViscosityMonaghan(alpha=0.01, beta=0.0)
surface_tension = SurfaceTensionAkinci(; surface_tension_coefficient)
# Equation 2 sums fluid neighbors only; wall adhesion is modeled separately by Equation 6.
surface_normal_method = ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf)
fluid_density_calculator = SummationDensity()
fluid_density_diffusion = nothing
pressure_acceleration = nothing

fluid_system = WeaklyCompressibleSPHSystem(drop; smoothing_kernel, smoothing_length,
                                           density_calculator=fluid_density_calculator,
                                           density_diffusion=fluid_density_diffusion,
                                           state_equation, viscosity, surface_tension,
                                           surface_normal_method, pressure_acceleration,
                                           correction=AkinciFreeSurfaceCorrection(fluid_density),
                                           reference_particle_spacing=particle_spacing,
                                           acceleration=(0.0, 0.0, -gravity))

boundary_hydrodynamic_mass = plate.mass
boundary_model = BoundaryModelDummyParticles(plate; fluid_system,
                                             hydrodynamic_mass=boundary_hydrodynamic_mass,
                                             boundary_density_calculator,
                                             viscosity,
                                             clip_negative_pressure=true)
boundary_system = WallBoundarySystem(plate, boundary_model; adhesion_coefficient)

parallelization_backend = PolyesterBackend()
semi = Semidiscretization(fluid_system, boundary_system; parallelization_backend)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01)
callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL35(); abstol=1e-7, reltol=1e-4, dtmax=1e-3,
            save_everystep=false, saveat=solution_saveat, callback=callbacks)
