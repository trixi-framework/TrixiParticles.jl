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

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.05

# ==========================================================================================
# ==== Experiment Setup
radius = 1.0
sigma = 0.5
# Make this a constant because global variables in the source terms are slow
const OMEGA = 1.0

source_terms = (coords, velocity, density, pressure, t) -> -OMEGA^2 * coords

# 1 period in the exact solution as computed below (but integrated with a small timestep)
period = 4.567375
tspan = (0.0, 1period)

fluid_density = 1000.0
sound_speed = 10.0
state_equation = StateEquationCole(; sound_speed, exponent=7,
                                   reference_density=fluid_density)

# Equation A.19 in the paper rearranged.
# sigma^2 = Q / rho * (a^2 + b^2) / (a^2 b^2) - OMEGA^2.
Q = (sigma^2 + OMEGA^2) * fluid_density / 2
pressure = coords -> Q * (1 - coords[1]^2 - coords[2]^2)
density = coords -> TrixiParticles.inverse_state_equation(state_equation, pressure(coords))

fluid = SphereShape(fluid_particle_spacing, radius, (0.0, 0.0),
                    density, pressure=pressure,
                    sphere_type=RoundSphere(),
                    velocity=coords -> sigma .* (coords[1], -coords[2]))

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.5 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

density_diffusion = DensityDiffusionAntuono(fluid, delta=0.1)
fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           source_terms=source_terms)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.04, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-7, # Default abstol is 1e-6 (may need to be tuned to prevent intabilities)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent intabilities)
            save_everystep=false, callback=callbacks);

@inline function exact_solution_rhs(u, p, t)
    sigma, A, B = u

    dsigma = (sigma^2 + OMEGA^2) * ((B^2 - A^2) / (B^2 + A^2))
    dA = sigma * A
    dB = -sigma * B

    return SVector(dsigma, dA, dB)
end

exact_u0 = SVector(sigma, radius, radius)
exact_solution_ode = ODEProblem(exact_solution_rhs, exact_u0, tspan)

# Use the same time integrator to avoid compilation of another integrator in CI
sol_exact = solve(exact_solution_ode, RDPK3SpFSAL49(), save_everystep=false)

# Error in the semi-major axis of the elliptical drop
error_A = maximum(sol.u[end].x[2]) + 0.5 * fluid_particle_spacing -
          maximum(sol_exact.u[end][2:3])
