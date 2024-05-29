# In this example we can observe that the `SurfaceTensionAkinci` surface tension model correctly leads to a
# surface minimization of the water square and approaches a sphere.
using TrixiParticles
using OrdinaryDiffEq

fluid_density = 1000.0

particle_spacing = 0.1

# Note: Only square shapes will result in a sphere.
# Furthermore, changes of the coefficients might be necessary for higher resolutions or larger squares.
fluid_size = (0.5, 0.5)

sound_speed = 20.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)

# For all surface tension simulations, we need a compact support of `2 * particle_spacing`
# smoothing_length = 2.0 * particle_spacing
# smoothing_kernel = WendlandC2Kernel{2}()
# nu = 0.01

smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
nu = 0.025

fluid = RectangularShape(particle_spacing, round.(Int, fluid_size ./ particle_spacing),
                         zeros(length(fluid_size)), density=fluid_density)

alpha = 8 * nu / (smoothing_length * sound_speed)
source_terms = SourceTermDamping(; damping_coefficient=0.5)
fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(),
                                           state_equation, smoothing_kernel,
                                           smoothing_length,
                                           viscosity=ArtificialViscosityMonaghan(alpha=alpha,
                                                                                 beta=0.0),
                                           surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.02),
                                           correction=AkinciFreeSurfaceCorrection(fluid_density),
                                           source_terms=source_terms)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

# For overwriting via `trixi_include`
saving_callback = SolutionSavingCallback(dt=0.02)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);
