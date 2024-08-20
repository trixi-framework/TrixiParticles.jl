using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.008

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 7
spacing_ratio = 1

boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 5.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.2, 0.1, 0.2)
tank_size = (0.3, 0.15, 0.20)
end_size = (0.3, 0.05, 0.20)

fluid_density = 1000.0
sound_speed = 40
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, background_pressure=1000)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=4, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity, 0.0), state_equation=state_equation,
                       faces=(true, false, true, false, true, true))

end_tank = RectangularTank(fluid_particle_spacing, (0.0, 0.0, 0.0), end_size, fluid_density,
                           n_layers=8, spacing_ratio=spacing_ratio,
                           acceleration=(0.0, -gravity, 0.0), state_equation=state_equation,
                           faces=(false, false, true, false, true, true),
                           min_coordinates=(tank_size[1], 0.0, 0.0))

# # move to the end
# for i in axes(end_tank.boundary.coordinates, 2)
#     end_tank.boundary.coordinates[:, i] .+= [tank_size[1], 0.0]
# end

# tank = union(tank, end_tank)

# ==========================================================================================
# ==== Fluid
smoothing_length = 3.5 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{3}()

fluid_density_calculator = ContinuityDensity()
# viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
nu_water = 8.9E-7

# Morris has a higher velocity with same viscosity.
# viscosity = ViscosityMorris(nu=100*nu_water)
viscosity = ViscosityAdami(nu=100 * nu_water)

density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

# fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
#                                            state_equation, smoothing_kernel,
#                                            smoothing_length, viscosity=viscosity,
#                                            density_diffusion=density_diffusion,
#                                            acceleration=(0.0, -gravity, 0.0))

fluid_system = EntropicallyDampedSPHSystem(tank.fluid, smoothing_kernel,
                                           smoothing_length,
                                           sound_speed, viscosity=viscosity,
                                           density_calculator=ContinuityDensity(),
                                           acceleration=(0.0, -gravity, 0.0),
                                           #    surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.05),
                                           reference_particle_spacing=fluid_particle_spacing,
                                           buffer_size=1)

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             viscosity=viscosity)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

boundary_model2 = BoundaryModelDummyParticles(end_tank.boundary.density,
                                              end_tank.boundary.mass,
                                              state_equation=state_equation,
                                              boundary_density_calculator,
                                              smoothing_kernel, smoothing_length,
                                              viscosity=viscosity)

boundary_system2 = BoundarySPHSystem(end_tank.boundary, boundary_model2)

outflow = OutFlow(; plane=([0.4, -0.2, -0.1], [0.4, -0.2, 0.3], [0.8, -0.2, 0.3]),
                  flow_direction=[0.0, -1.0, 0.0], open_boundary_layers=1,
                  density=2*eps(), particle_spacing=fluid_particle_spacing)

open_boundary_out = OpenBoundarySPHSystem(outflow; fluid_system,
                                          boundary_model=BasicOutlet())

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system, boundary_system2,
                          open_boundary_out)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01, prefix="lower_visc")
callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            dt=1e-6,
            save_everystep=false, callback=callbacks);
