# ==========================================================================================
# 3D Akinci Cube-to-Sphere Experiment
#
# Reproduces the setup of Figure 2 in Akinci, Akinci, and Teschner (2013): an initially
# cubical drop first minimizes its surface area and is then released onto the ground.
# The companion video reports a 1 cm^3 drop represented by about 7,000 particles.
# https://doi.org/10.1145/2508363.2508395
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

drop_volume = 1.0e-6
target_particle_count = 7_000
cube_side_length = cbrt(drop_volume)
particles_per_dimension = round(Int, cbrt(target_particle_count))
particle_spacing = cube_side_length / particles_per_dimension
fluid_density = 1000.0
sound_speed = 40.0
fluid_clip_negative_pressure = true
gravity = 9.81
tspan = (0.0, 0.1)
solution_saveat = ()
drop_initial_condition = nothing
provide_boundary_surface_geometry = false

release_time = 0.05
cube_bottom_height = 0.0025
floor_size = 0.03
boundary_layers = 3

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7,
                                   clip_negative_pressure=fluid_clip_negative_pressure)

n_cube = ntuple(_ -> round(Int, cube_side_length / particle_spacing), 3)
cube_min = (-cube_side_length / 2, -cube_side_length / 2, cube_bottom_height)
drop = isnothing(drop_initial_condition) ?
       RectangularShape(particle_spacing, n_cube, cube_min; density=fluid_density) :
       drop_initial_condition

n_floor = round(Int, floor_size / particle_spacing)
floor_raw = RectangularShape(particle_spacing, (n_floor, n_floor, boundary_layers),
                             (-floor_size / 2, -floor_size / 2,
                              -boundary_layers * particle_spacing);
                             density=fluid_density)
floor_surface_measure = nothing
floor = floor_raw
if provide_boundary_surface_geometry
    exposed_height = maximum(floor_raw.coordinates[3, :])
    exposed = isapprox.(floor_raw.coordinates[3, :], exposed_height;
                        atol=eps(eltype(floor_raw)))
    normals = zeros(eltype(floor_raw), size(floor_raw.coordinates))
    normals[3, exposed] .= -particle_spacing / 2
    floor_surface_measure = zeros(eltype(floor_raw), nparticles(floor_raw))
    floor_surface_measure[exposed] .= particle_spacing^2
    floor = InitialCondition(; coordinates=floor_raw.coordinates,
                             velocity=floor_raw.velocity,
                             mass=floor_raw.mass, density=floor_raw.density,
                             pressure=floor_raw.pressure, particle_spacing,
                             normals)
end

smoothing_length = particle_spacing - eps()
smoothing_kernel = SchoenbergCubicSplineKernel{3}()
viscosity = ArtificialViscosityMonaghan(alpha=0.01, beta=0.0)
surface_tension_coefficient = 1.0
surface_tension = SurfaceTensionAkinci(; surface_tension_coefficient)
# Equation 2 sums fluid neighbors only; wall adhesion is modeled separately by Equation 6.
surface_normal_method = ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf)
density_calculator = SummationDensity()
density_diffusion = nothing
correction = AkinciFreeSurfaceCorrection(fluid_density)
pressure_acceleration = nothing
shifting_technique = nothing
adhesion_coefficient = surface_tension_coefficient

gravity_after_release = let release_time=release_time, gravity=gravity
    (coords, velocity, density, pressure,
     t) -> t < release_time ? SVector(0.0, 0.0, 0.0) : SVector(0.0, 0.0, -gravity)
end

fluid_system = WeaklyCompressibleSPHSystem(drop; smoothing_kernel, smoothing_length,
                                           density_calculator, density_diffusion,
                                           state_equation, viscosity, pressure_acceleration,
                                           surface_tension,
                                           surface_normal_method,
                                           correction,
                                           shifting_technique,
                                           reference_particle_spacing=particle_spacing,
                                           source_terms=gravity_after_release)

boundary_model = BoundaryModelDummyParticles(floor; fluid_system,
                                             boundary_density_calculator=AdamiPressureExtrapolation(),
                                             viscosity,
                                             clip_negative_pressure=true,
                                             surface_measure=floor_surface_measure)
boundary_system = WallBoundarySystem(floor, boundary_model; adhesion_coefficient)

semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02)
timestep_diagnostic_callback = nothing
update_callback = nothing
callbacks = CallbackSet(info_callback, saving_callback, timestep_diagnostic_callback,
                        update_callback)

sol = solve(ode, RDPK3SpFSAL35(); abstol=1e-7, reltol=1e-4, dtmax=2e-3,
            save_everystep=false, saveat=solution_saveat, callback=callbacks)
