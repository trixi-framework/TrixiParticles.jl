# ==========================================================================================
# 2D Taylor-Green Vortex Simulation
#
# Based on:
#   P. Ramachandran, K. Puri.
#   "Entropically damped artiﬁcial compressibility for SPH".
#   Computers and Fluids, Volume 179 (2019), pages 579-594.
#   https://doi.org/10.1016/j.compfluid.2018.11.023
#
# This example simulates the Taylor-Green vortex, a classic benchmark case for
# incompressible viscous flow, characterized by an array of decaying vortices.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Resolution
particle_spacing = 0.02

# Physical parameters
reynolds_number = 100.0
domain_box_length = 1.0   # Side length of the square periodic domain
characteristic_velocity_U = 1.0 # Characteristic velocity (m/s)
fluid_density_ref = 1.0     # Reference density of the fluid (kg/m^3)

# Simulation time span
tspan = (0.0, 5.0)

# Choose fluid formulation:
# `true` for WCSPH, `false` for EDSPH
use_wcsph_formulation = false

# Option to perturb initial particle coordinates.
# This can help break symmetries and prevent stagnant streamlines, especially if
# not using a transport velocity formulation or if initial density estimation is sensitive.
perturb_initial_coordinates = true

# ------------------------------------------------------------------------------
# Analytical Solution and Derived Parameters
# ------------------------------------------------------------------------------
# Speed of sound, typically ~10 times the characteristic velocity for low compressibility.
sound_speed = 10 * characteristic_velocity_U

# Decay rate for Taylor-Green vortex analytical solution
decay_rate_b = -8 * pi^2 / reynolds_number

# Kinematic viscosity (nu = U * L / Re)
kinematic_viscosity_nu = characteristic_velocity_U * domain_box_length / reynolds_number

# Background pressure for transport velocity formulation (rho_0 * c_0^2)
background_pressure_transport_velocity = fluid_density_ref * sound_speed^2

# Analytical pressure function p(x, y, t) for Taylor-Green vortex
function analytical_pressure(position, time)
    x, y = position
    term_exp = exp(2 * decay_rate_b * time)
    term_cos = cos(4 * pi * x) + cos(4 * pi * y)
    return -characteristic_velocity_U^2 * term_exp * term_cos / 4
end
initial_pressure_at_position(pos) = analytical_pressure(pos, 0.0)

# Analytical velocity function v(x, y, t) for Taylor-Green vortex
function analytical_velocity(position, time)
    x, y = position
    vel_factor = characteristic_velocity_U * exp(decay_rate_b * time)
    vx = -vel_factor * cos(2 * pi * x) * sin(2 * pi * y)
    vy =  vel_factor * sin(2 * pi * x) * cos(2 * pi * y)
    return SVector(vx, vy)
end
initial_velocity_at_position(pos) = analytical_velocity(pos, 0.0)

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------
# Number of particles along each dimension of the square domain
num_particles_per_dim = round(Int, domain_box_length / particle_spacing)

smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

# Create initial fluid particles in a square domain
coordinate_perturbation_factor = perturb_initial_coordinates ? 0.2 : nothing
fluid_particles = RectangularShape(particle_spacing,
                                   (num_particles_per_dim, num_particles_per_dim),
                                   SVector(0.0, 0.0),
                                   density=fluid_density_ref,
                                   pressure=initial_pressure_at_position,
                                   velocity=initial_velocity_at_position,
                                   coordinates_perturbation=coordinate_perturbation_factor)

# Viscosity and transport velocity models (common to both formulations)
viscosity_model_adami = ViscosityAdami(nu=kinematic_viscosity_nu)
transport_velocity_adami = TransportVelocityAdami(background_pressure_transport_velocity)

if use_wcsph_formulation
    # Note on WCSPH with perturbed coordinates:
    # If using `SummationDensity` with perturbed coordinates, initial density estimates might be noisy.
    # Adami et al. (2013) often use a particle relaxation step for initial positions.
    # `ContinuityDensity` is generally more robust to initial particle arrangements.
    density_calculator_wcsph = ContinuityDensity()
    state_equation_wcsph = StateEquationCole(sound_speed=sound_speed,
                                             reference_density=fluid_density_ref,
                                             exponent=1)
    # Inter-particle averaged pressure gradient (Adami et al. 2013, Eq. 17)
    pressure_acceleration_wcsph = TrixiParticles.inter_particle_averaged_pressure

    fluid_system = WeaklyCompressibleSPHSystem(fluid_particles, density_calculator_wcsph,
                                               state_equation_wcsph, smoothing_kernel,
                                               smoothing_length,
                                               pressure_acceleration=pressure_acceleration_wcsph,
                                               viscosity=viscosity_model_adami,
                                               transport_velocity=transport_velocity_adami,
                                               reference_particle_spacing=particle_spacing)
else
    density_calculator_edsph = SummationDensity()
    fluid_system = EntropicallyDampedSPHSystem(fluid_particles, smoothing_kernel,
                                               smoothing_length,
                                               sound_speed,
                                               density_calculator=density_calculator_edsph,
                                               viscosity=viscosity_model_adami,
                                               transport_velocity=transport_velocity_adami,
                                               reference_particle_spacing=particle_spacing)
end

# ------------------------------------------------------------------------------
# Simulation Setup
# ------------------------------------------------------------------------------
# Periodic boundary conditions in both x and y directions.
periodic_box_domain = PeriodicBox(min_corner=SVector(0.0, 0.0),
                                  max_corner=SVector(domain_box_length, domain_box_length))

neighborhood_search = GridNeighborhoodSearch{2}(periodic_box=periodic_box_domain)

# No explicit boundary systems, only fluid system with periodic conditions.
semi = Semidiscretization(fluid_system,
                          neighborhood_search=neighborhood_search,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

# `UpdateCallback` is important for periodic boundaries.
update_callback = UpdateCallback()
extra_callback = nothing # For potential `trixi_include` overrides

callbacks = CallbackSet(info_callback, saving_callback, update_callback, extra_callback)

# Estimate for maximum stable time step (dt_max) based on CFL condition and viscous diffusion.
# This helps prevent the adaptive time stepper from choosing overly large steps.
cfl_dt_max = smoothing_length / (4 * (sound_speed + characteristic_velocity_U))
viscous_dt_max = smoothing_length^2 / (8 * kinematic_viscosity_nu)
dt_max_simulation = min(cfl_dt_max, viscous_dt_max)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=dt_max_simulation, save_everystep=false, callback=callbacks);
