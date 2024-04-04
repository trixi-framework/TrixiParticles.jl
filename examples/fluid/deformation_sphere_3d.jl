# In this example we can observe that the `SurfaceTensionAkinci` surface tension model correctly leads to a
# surface minimization of the water cube and approaches a sphere.
using TrixiParticles
using OrdinaryDiffEq

fluid_density = 1000.0

particle_spacing = 0.1

sound_speed = 20
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)

# for all surface tension simulations needs to be smoothing_length = 4r
smoothing_length = 2.0 * particle_spacing
smoothing_kernel = WendlandC2Kernel{3}()

fluid = RectangularShape(particle_spacing, (9, 9, 9), (0.0, 0.0, 0.0),
                         density=fluid_density)

nu = 0.01
alpha = 10 * nu / (smoothing_length * sound_speed)

fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(),
                                           state_equation, smoothing_kernel,
                                           smoothing_length,
                                           viscosity=ArtificialViscosityMonaghan(alpha=alpha,
                                                                                 beta=0.0),
                                           acceleration=(0.0, 0.0, 0.0),
                                           surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.1),
                                           correction=AkinciFreeSurfaceCorrection(fluid_density))

semi = Semidiscretization(fluid_system)

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=1000)
saving_callback = SolutionSavingCallback(dt=0.02)

stepsize_callback = StepsizeCallback(cfl=1.2)

callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);
