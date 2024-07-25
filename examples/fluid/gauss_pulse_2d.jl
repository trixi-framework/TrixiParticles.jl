using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.04

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 5.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (2.0, 2.0)
n_particles_per_dimension = round.(Int, initial_fluid_size ./ fluid_particle_spacing)

fluid_density = 1000.0
sound_speed = 10.0

fluid = RectangularShape(fluid_particle_spacing, n_particles_per_dimension, (-1.0, -1.0);
                         velocity=[1.0, 0.0], pressure=1000,
                         density=fluid_density)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

const sigma_x = 0.05
const sigma_y = 0.05
const amplitude = 5.0
const x_center = 0.0
const y_center = 0.0

function pulse_2d(coords, velocity, density, pressure, t)
    x = coords[1]
    y = coords[2]

    if 0.25 < t < 2.5
        u = amplitude *
            exp(-((x - (t - x_center))^2 / (2 * sigma_x^2) +
                  (y - y_center)^2 / (2 * sigma_y^2)))
        return SVector(u, 0.0)
    end

    return SVector(0.0, 0.0)
end

viscosity = ViscosityAdami(nu=0.2 * smoothing_length * sound_speed / 8)

fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                           sound_speed, viscosity=viscosity,
                                           source_terms=pulse_2d,
                                           density_calculator=ContinuityDensity())

# ==========================================================================================
# ==== Simulation
periodic_box = PeriodicBox(min_corner=[-1.0, -1.0], max_corner=[1.0, 1.0])

semi = Semidiscretization(fluid_system;
                          neighborhood_search=GridNeighborhoodSearch{2}(; periodic_box))
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

# This is to easily add a new callback with `trixi_include`
extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, extra_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks);
