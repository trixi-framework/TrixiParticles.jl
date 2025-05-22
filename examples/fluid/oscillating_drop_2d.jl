# ==========================================================================================
# 2D Oscillating Drop Simulation
#
# Based on:
#   J. J. Monaghan, Ashkan Rafiee.
#   "A Simple SPH Algorithm for Multi-Fluid Flow with High Density Ratios."
#   International Journal for Numerical Methods in Fluids, 71(5) (2013), pp. 537-561.
#   https://doi.org/10.1002/fld.3671.
#
# This example simulates a 2D elliptical drop of fluid oscillating under a central
# force field.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Particle spacing (resolution)
fluid_particle_spacing = 0.05

# Parameters from Monaghan & Rafiee (2013), Appendix A
initial_drop_radius = 1.0 # Initial radius of the circular drop
sigma_oscillation = 0.5   # Parameter related to initial velocity field
omega_force_field = 1.0   # Angular frequency of the central force field

# Define OMEGA as a const for performance when used in the source_terms closure.
const OMEGA = omega_force_field

# Source term: central force field F = -OMEGA^2 * r (per unit mass)
source_terms_central_force(coords, velocity, density, pressure, t) = -OMEGA^2 * coords

analytical_period = 4.567375
tspan = (0.0, 1.0 * analytical_period)

# ------------------------------------------------------------------------------
# Fluid Properties and Initial Conditions
# ------------------------------------------------------------------------------

fluid_density = 1000.0
sound_speed = 10.0

state_equation = StateEquationCole(sound_speed=sound_speed,
                                   reference_density=fluid_density,
                                   exponent=7)

# Initialization based on Monaghan & Rafiee (2013), Appendix A
# sigma^2 = Q / rho * (a^2 + b^2) / (a^2 b^2) - OMEGA^2.
initial_Q_factor = (sigma_oscillation^2 + OMEGA^2) * fluid_density *
                   initial_drop_radius^2 / 2

function initial_pressure_field(coords)
    initial_Q_factor *
    (1 - (coords[1]^2 + coords[2]^2) / initial_drop_radius^2)
end

function initial_density_field(coords)
    TrixiParticles.inverse_state_equation(state_equation,
                                          initial_pressure_field(coords))
end

initial_velocity_field(coords) = sigma_oscillation .* SVector(coords[1], -coords[2])

fluid_particles = SphereShape(fluid_particle_spacing, initial_drop_radius, (0.0, 0.0),
                              initial_density_field;
                              pressure=initial_pressure_field,
                              sphere_type=RoundSphere(),
                              velocity=initial_velocity_field)

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

smoothing_length = 1.5 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity_model = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

density_diffusion_model = DensityDiffusionAntuono(fluid_particles, delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(fluid_particles, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length,
                                           viscosity=viscosity_model,
                                           density_diffusion=density_diffusion_model,
                                           source_terms=source_terms_central_force)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------

semi = Semidiscretization(fluid_system,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.04, prefix="oscillating_drop")
callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-7,
            reltol=1e-4,
            save_everystep=false,
            callback=callbacks)

# ------------------------------------------------------------------------------
# Post-processing: Compare with Analytical Solution (as in original file)
# ------------------------------------------------------------------------------
@inline function exact_solution_rhs(u, p, t)
    sigma_oscillation, A, B = u

    dsigma = (sigma_oscillation^2 + OMEGA^2) * ((B^2 - A^2) / (B^2 + A^2))
    d_a = sigma_oscillation * A
    d_b = -sigma_oscillation * B

    return SVector(dsigma, d_a, d_b)
end

exact_u0 = SVector(sigma_oscillation, initial_drop_radius, initial_drop_radius)
exact_solution_ode = ODEProblem(exact_solution_rhs, exact_u0, tspan)

# Use the same time integrator to avoid compilation of another integrator in CI
sol_exact = solve(exact_solution_ode, RDPK3SpFSAL49(), save_everystep=false)

# Error in the semi-major axis of the elliptical drop
error_A = maximum(sol.u[end].x[2]) + 0.5 * fluid_particle_spacing -
          maximum(sol_exact.u[end][2:3])
println("Error in the semi-major axis of the elliptical drop: ", error_A)
