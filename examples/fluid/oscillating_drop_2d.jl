using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.02

# ==========================================================================================
# ==== Experiment Setup
radius = 1.0
sigma = 0.5
omega = 1.0

source_terms = (coords, velocity, density, pressure) -> -omega^2 * coords

period = 4.83
tspan = (0.0, 1period)

fluid_density = 1000.0
sound_speed = 10.0
state_equation = StateEquationCole(; sound_speed, exponent=7,
                                   reference_density=fluid_density)

# Equation A.19 in the paper rearranged.
# sigma^2 = Q / rho * (a^2 + b^2) / (a^2 b^2) - omega^2.
Q = (sigma^2 + omega^2) * fluid_density / 2
pressure = coords -> Q * (1 - coords[1]^2 - coords[2]^2)
density = coords -> TrixiParticles.inverse_state_equation(state_equation, pressure(coords))

fluid = SphereShape(fluid_particle_spacing, radius, (0.0, 0.0),
                    density, pressure=pressure,
                    sphere_type=RoundSphere(),
                    velocity=coords -> sigma .* (coords[1], -coords[2]))

# ==========================================================================================
# ==== Fluid
smoothing_length = 3.0 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
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

@inline function exact_solution_rhs(u, omega, t)
    sigma, A, B = u

    dsigma = (sigma^2 + omega^2) * ((B^2 - A^2) / (B^2 + A^2))
    dA = sigma * A
    dB = -sigma * B

    return SVector(dsigma, dA, dB)
end

exact_u0 = SVector(sigma, 1.0, 1.0)
exact_solution_ode = ODEProblem(exact_solution_rhs, exact_u0, tspan, omega)

@time sol_exact = solve(exact_solution_ode, Tsit5(), save_everystep=false)

error_AB = maximum(sol.u[end].x[2]) - maximum(sol_exact.u[end][2:3])
