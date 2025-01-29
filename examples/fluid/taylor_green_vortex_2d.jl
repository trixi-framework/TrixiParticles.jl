# Taylor Green vortex
#
# P. Ramachandran, K. Puri
# "Entropically damped artiﬁcial compressibility for SPH".
# In: Computers and Fluids, Volume 179 (2019), pages 579-594.
# https://doi.org/10.1016/j.compﬂuid.2018.11.023

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.02

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 5.0)
reynolds_number = 100.0

box_length = 1.0

U = 1.0 # m/s
fluid_density = 1.0
sound_speed = 10U

b = -8pi^2 / reynolds_number

# Taylor Green Vortex Pressure Function
function pressure_function(pos, t)
    x = pos[1]
    y = pos[2]

    return -U^2 * exp(2 * b * t) * (cos(4pi * x) + cos(4pi * y)) / 4
end

initial_pressure_function(pos) = pressure_function(pos, 0.0)

# Taylor Green Vortex Velocity Function
function velocity_function(pos, t)
    x = pos[1]
    y = pos[2]

    vel = U * exp(b * t) * [-cos(2pi * x) * sin(2pi * y), sin(2pi * x) * cos(2pi * y)]

    return SVector{2}(vel)
end

initial_velocity_function(pos) = velocity_function(pos, 0.0)

n_particles_xy = round(Int, box_length / particle_spacing)

# ==========================================================================================
# ==== Fluid
nu = U * box_length / reynolds_number

background_pressure = sound_speed^2 * fluid_density

smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

# To be set via `trixi_include`
perturb_coordinates = true
fluid = RectangularShape(particle_spacing, (n_particles_xy, n_particles_xy), (0.0, 0.0),
                         # Perturb particle coordinates to avoid stagnant streamlines without TVF
                         coordinates_perturbation=perturb_coordinates ? 0.2 : nothing, # To avoid stagnant streamlines when not using TVF.
                         density=fluid_density, pressure=initial_pressure_function,
                         velocity=initial_velocity_function)

density_calculator = SummationDensity()
fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                           sound_speed,
                                           density_calculator=density_calculator,
                                           transport_velocity=TransportVelocityAdami(background_pressure),
                                           viscosity=ViscosityAdami(; nu))

# ==========================================================================================
# ==== Simulation
periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[box_length, box_length])
semi = Semidiscretization(fluid_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(; periodic_box))

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

saving_callback = SolutionSavingCallback(dt=0.02)

pp_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, pp_callback, UpdateCallback())

dt_max = min(smoothing_length / 4 * (sound_speed + U), smoothing_length^2 / (8 * nu))

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=dt_max, save_everystep=false, callback=callbacks);
