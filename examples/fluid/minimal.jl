using TrixiParticles
using OrdinaryDiffEq

fluid_density = 1000.0
atmospheric_pressure = 100000.0
state_equation = StateEquationCole(30, 7, fluid_density, atmospheric_pressure,
                                   background_pressure=atmospheric_pressure)

fluid_particle_spacing = 0.02
initial_condition = RectangularShape(fluid_particle_spacing, (10, 10), (0, 0),
                                     fluid_density,
                                     pressure=0.0)

fluid_system = WeaklyCompressibleSPHSystem(initial_condition, ContinuityDensity(),
                                           state_equation, SchoenbergCubicSplineKernel{2}(),
                                           1.2 * fluid_particle_spacing,
                                           viscosity=ArtificialViscosityMonaghan(alpha=0.02,
                                                                                 beta=0.0))

tspan = (0.0, 2.0)
ode = initialize_ode(tspan, fluid_system, neighborhood_search=GridNeighborhoodSearch)

sol = solve(ode, RDPK3SpFSAL49(), callback=default_callback());
