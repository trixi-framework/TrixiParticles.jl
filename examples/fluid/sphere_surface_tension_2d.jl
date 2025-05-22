# ==========================================================================================
# 2D Sphere Formation via Surface Tension
#
# This example demonstrates how surface tension models in TrixiParticles.jl
# can cause an initially square patch of fluid to minimize its surface area,
# ideally forming a circular (2D sphere) shape.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

# Fluid properties
fluid_density_ref = 1000.0 # Reference density of the fluid (kg/m^3)
sound_speed = 20.0         # Speed of sound (m/s)

# Resolution and Geometry
# A higher resolution (smaller `particle_spacing`) yields better results
# but increases computation time.
particle_spacing = 0.05
# particle_spacing = 0.025

# Initial shape: A square patch of fluid.
# Only square shapes are expected to form a perfect circle under isotropic surface tension.
# The size of the square might require tuning of model coefficients for optimal results.
initial_fluid_size = (0.5, 0.5) # width, height

# SPH numerical parameters
# Specific choices for smoothing length and kernel can be important for surface tension models.
# Option 1: (Akinci model)
#   - smoothing_length = 1.0 * particle_spacing
#   - smoothing_kernel = WendlandC2Kernel{2}()
# Option 2: (Morris model)
smoothing_length = 1.75 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

# Viscosity or Damping
# Physical kinematic viscosity `nu` or parameters for artificial viscosity.
# For SurfaceTensionMorris with EDSPH:
nu_morris = 0.05
# For SurfaceTensionAkinci with WCSPH:
# nu_akinci_artificial = 0.01
# alpha_akinci = 8 * nu_akinci_artificial / (1.0 * particle_spacing * sound_speed)

# Surface tension coefficients
surface_tension_coeff_morris = 50 * 0.0728 # Scaled value (0.0728 N/m is water-air surf. tens.)
# surface_tension_coeff_akinci = 0.02

# Simulation time span
tspan = (0.0, 50.0) # Long enough to observe shape evolution

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

# Create initial square patch of fluid particles
fluid_particles = RectangularShape(particle_spacing,
                                   round.(Int, initial_fluid_size ./ particle_spacing),
                                   zeros(length(initial_fluid_size)),
                                   density=fluid_density_ref)

# Select one of the following fluid system setups:

density_calculator_edsph = ContinuityDensity()
viscosity_edsph = ViscosityMorris(nu=nu_morris)
surface_normal_method_colorfield = ColorfieldSurfaceNormal()
surface_tension_model_morris = SurfaceTensionMorris(surface_tension_coefficient=surface_tension_coeff_morris)

fluid_system = EntropicallyDampedSPHSystem(fluid_particles, smoothing_kernel,
                                           smoothing_length,
                                           sound_speed,
                                           density_calculator=density_calculator_edsph,
                                           viscosity=viscosity_edsph,
                                           surface_normal_method=surface_normal_method_colorfield,
                                           surface_tension=surface_tension_model_morris,
                                           acceleration=zeros(length(initial_fluid_size)),
                                           reference_particle_spacing=particle_spacing)

# == Alternative Setup: Weakly Compressible SPH with Akinci Surface Tension ==
# density_calculator_wcsph = SummationDensity() # Or ContinuityDensity
# viscosity_wcsph = ArtificialViscosityMonaghan(alpha=alpha_akinci, beta=0.0)
# surface_tension_model_akinci = SurfaceTensionAkinci(surface_tension_coefficient=surface_tension_coeff_akinci)
# free_surface_correction_akinci = AkinciFreeSurfaceCorrection(fluid_density_ref)
# # Damping can help stabilize the system, especially during large deformations.
# source_terms_damping = SourceTermDamping(damping_coefficient=0.5)
#
# state_equation = StateEquationCole(sound_speed=sound_speed,
# reference_density=fluid_density_ref,
# exponent=7,
# clip_negative_pressure=true)
#
# fluid_system_wcsph = WeaklyCompressibleSPHSystem(fluid_particles, density_calculator_wcsph,
#                                                  state_equation, smoothing_kernel,
#                                                  smoothing_length, # Use appropriate S.L. for Akinci
#                                                  viscosity=viscosity_wcsph,
#                                                  surface_tension=surface_tension_model_akinci,
#                                                  correction=free_surface_correction_akinci,
#                                                  source_terms=source_terms_damping,
#                                                  acceleration=zeros(length(initial_fluid_size)),
#                                                  reference_particle_spacing=particle_spacing)
# fluid_system = fluid_system_wcsph # Uncomment to activate this setup

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
# No explicit boundary systems in this example.
semi = Semidiscretization(fluid_system,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

# Callbacks
info_callback = InfoCallback(interval=100) # Print info every 100 steps

# `saving_callback` can be overridden by `trixi_include` if this file is used as a base.
saving_callback = SolutionSavingCallback(dt=1.0, prefix="") # Save every 1.0 time unit
stepsize_callback = StepsizeCallback(cfl=1.0) # CFL-based adaptive time stepping

callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

# `dt=1.0` is an initial guess; `StepsizeCallback` will override it.
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0,
            save_everystep=false,
            callback=callbacks)
