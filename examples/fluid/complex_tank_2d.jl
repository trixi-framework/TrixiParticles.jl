using TrixiParticles
using DelimitedFiles
using OrdinaryDiffEq

#polygon = TrixiParticles.Polygon([1.8 2.4 6.2 8.5 5.1 3.4 1.8;
#                                   4.2 1.7 0.8 3.2 5.1 0.2 4.2])
factor = 1.0e-2
fluid_particle_spacing = 1.0 * factor

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 3.0)
fluid_density = 1000.0
sound_speed = 20.0

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

points_tank = readdlm(joinpath(examples_dir(), "preprocessing", "tank_ramp.asc"), ' ',
                      Float64,
                      '\n', header=true)[1][:, 1:2] .* factor

shape_tank = TrixiParticles.Polygon(points_tank')
tank = TrixiParticles.sample(; shape=shape_tank, particle_spacing=fluid_particle_spacing,
                             density=fluid_density,
                             point_in_poly=TrixiParticles.WindingNumberHorman(shape_tank))

points_fluid = readdlm(joinpath(examples_dir(), "preprocessing", "fluid.asc"), ' ',
                       Float64, '\n', header=true)[1][:, 1:2] .* factor

shape_fluid = TrixiParticles.Polygon(points_fluid')
fluid = TrixiParticles.sample(; shape=shape_fluid, particle_spacing=fluid_particle_spacing,
                              density=fluid_density,
                              point_in_poly=TrixiParticles.WindingNumberJacobson())

fluid = setdiff(fluid, tank)

# ==========================================================================================
# ==== Fluid
smoothing_length = 3.0 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.1, beta=0.0)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
# density_diffusion = DensityDiffusionAntuono(tank.fluid, delta=0.1)
fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity), correction=nothing)

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.density, tank.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             correction=nothing)

boundary_system = BoundarySPHSystem(tank, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=250)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

use_reinit = false
density_reinit_cb = use_reinit ? DensityReinitializationCallback(semi.systems[1], dt=0.01) :
                    nothing

callbacks = CallbackSet(info_callback, saving_callback, density_reinit_cb)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-5, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
