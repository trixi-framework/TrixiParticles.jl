# ==========================================================================================
# 2D Sphere Formation via Surface Tension
#
# This example demonstrates how surface tension models in TrixiParticles.jl
# can cause an initially square patch of fluid to minimize its surface area,
# ideally forming a circular (2D sphere) shape.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

fluid_density = 1000.0

particle_spacing = 0.05
# Use a higher resolution for a better result
# particle_spacing = 0.025

# Note: Only square shapes will result in a sphere.
# Furthermore, changes of the coefficients might be necessary for higher resolutions or larger squares.
fluid_size = (0.5, 0.5)

sound_speed = 20.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)

# For all surface tension simulations, we need a compact support of `2 * particle_spacing`
# smoothing_length = particle_spacing
# smoothing_kernel = WendlandC2Kernel{2}()
# nu = 0.01

smoothing_length = 1.75 * particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()
# nu = 0.001 # SurfaceTensionMomentumMorris
nu = 0.05 # SurfaceTensionMorris

fluid = RectangularShape(particle_spacing, round.(Int, fluid_size ./ particle_spacing),
                         zeros(length(fluid_size)), density=fluid_density)

alpha = 8 * nu / (smoothing_length * sound_speed)
source_terms = SourceTermDamping(; damping_coefficient=0.5)
# fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(),
#                                            state_equation, fluid_smoothing_kernel,
#                                            smoothing_length,
#                                            reference_particle_spacing=particle_spacing,
#                                            viscosity=ArtificialViscosityMonaghan(alpha=alpha,
#                                                                                  beta=0.0),
#                                            surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.02),
#                                            correction=AkinciFreeSurfaceCorrection(fluid_density),
#                                            source_terms=source_terms)

# Alternatively can also be used with surface_tension=SurfaceTensionMomentumMorris(surface_tension_coefficient=1.0)
fluid_system = EntropicallyDampedSPHSystem(fluid, fluid_smoothing_kernel,
                                           smoothing_length,
                                           sound_speed,
                                           viscosity=ViscosityMorris(nu=nu),
                                           density_calculator=ContinuityDensity(),
                                           reference_particle_spacing=particle_spacing,
                                           acceleration=zeros(length(fluid_size)),
                                           surface_normal_method=ColorfieldSurfaceNormal(),
                                           surface_tension=SurfaceTensionMorris(surface_tension_coefficient=50 *
                                                                                                            0.0728))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system)

tspan = (0.0, 50.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

# For overwriting via `trixi_include`
saving_callback = SolutionSavingCallback(dt=1.0)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);
