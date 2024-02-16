using TrixiParticles
using OrdinaryDiffEq

fluid_density = 1000.0
atmospheric_pressure = 100000.0
sound_speed = 30
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

fluid_particle_spacing = 0.02
initial_condition = RectangularShape(fluid_particle_spacing, (10, 10), (0, 0),
                                     density=fluid_density, pressure=0.0)

fluid_system = WeaklyCompressibleSPHSystem(initial_condition, ContinuityDensity(),
                                           state_equation, SchoenbergCubicSplineKernel{2}(),
                                           1.2 * fluid_particle_spacing,
                                           viscosity=ArtificialViscosityMonaghan(alpha=0.02,
                                                                                 beta=0.0))

tspan = (0.0, 2.0)
semi = Semidiscretization(fluid_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)
sol = solve(ode, RDPK3SpFSAL49(), callback=callbacks);
