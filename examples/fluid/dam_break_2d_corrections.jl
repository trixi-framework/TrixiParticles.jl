# 2D dam break simulation based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

using TrixiParticles
using OrdinaryDiffEq

gravity = 9.81

# ==========================================================================================
# ==== Fluid

particle_spacing = 0.05

# Spacing ratio between fluid and boundary particles
beta = 1
boundary_layers = 3

water_width = 2.0
water_height = 1.0
water_density = 1000.0

tank_width = floor(5.366 / particle_spacing * beta) * particle_spacing / beta
tank_height = 4

sound_speed = 20 * sqrt(gravity * water_height)

smoothing_length = 1.15 * particle_spacing
smoothing_kernel = SchoenbergQuarticSplineKernel{2}()

state_equation = StateEquationCole(sound_speed, 7, water_density, 100000.0,
                                   background_pressure=100000.0)

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

tank = RectangularTank(particle_spacing, (water_width, water_height),
                       (tank_width, tank_height), water_density,
                       n_layers=boundary_layers, spacing_ratio=beta)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             SummationDensity(), smoothing_kernel,
                                             smoothing_length)

correction_dict = Dict(
    "no_correction" => Nothing(),
    "shepard_kernel_correction" => ShepardKernelCorrection(),
    "akinci_free_surf_correction" => AkinciFreeSurfaceCorrection(water_density),
    "kernel_gradient_summation_correction" => KernelGradientCorrection(),
    "kernel_gradient_continuity_correction" => KernelGradientCorrection(),
)

density_calculator_dict = Dict(
    "no_correction" => SummationDensity(),
    "shepard_kernel_correction" => SummationDensity(),
    "akinci_free_surf_correction" => SummationDensity(),
    "kernel_gradient_summation_correction" => SummationDensity(),
    "kernel_gradient_continuity_correction" => ContinuityDensity(),
)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
sol = nothing
for correction_name in keys(correction_dict)
    particle_system = WeaklyCompressibleSPHSystem(tank.fluid,
                                                  density_calculator_dict[correction_name],
                                                  state_equation,
                                                  smoothing_kernel, smoothing_length,
                                                  viscosity=viscosity,
                                                  acceleration=(0.0, -gravity),
                                                  correction=correction_dict[correction_name])

    # Move right boundary.
    # Use the new fluid size, since it might have been rounded in `RectangularTank`.
    reset_faces = (false, true, false, false)
    positions = (0, tank.fluid_size[1], 0, 0)

    reset_wall!(tank, reset_faces, positions)

    semi = Semidiscretization(particle_system, boundary_system,
                              neighborhood_search=GridNeighborhoodSearch,
                              damping_coefficient=1e-5)

    tspan = (0.0, 3.0)
    ode = semidiscretize(semi, tspan)

    info_callback = InfoCallback(interval=100)
    saving_callback_relaxation = SolutionSavingCallback(interval=100,
                                                        prefix="$(correction_name)_relaxation")
    callbacks_relaxation = CallbackSet(info_callback, saving_callback_relaxation)

    # Use a Runge-Kutta method with automatic (error based) time step size control.
    # Enable threading of the RK method for better performance on multiple threads.
    # Limiting of the maximum stepsize is necessary to prevent crashing.
    # When particles are approaching a wall in a uniform way, they can be advanced
    # with large time steps. Close to the wall, the stepsize has to be reduced drastically.
    # Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
    # become extremely large when fluid particles are very close to boundary particles,
    # and the time integration method interprets this as an instability.
    global sol = solve(ode, RDPK3SpFSAL49(),
                       abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
                       reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
                       dtmax=1e-2, # Limit stepsize to prevent crashing
                       save_everystep=false, callback=callbacks_relaxation)

    # Move right boundary
    positions = (0, tank.tank_size[1], 0, 0)
    reset_wall!(tank, reset_faces, positions)

    # Run full simulation
    tspan = (0.0, 5.7 / sqrt(9.81))

    # Use solution of the relaxing step as initial coordinates
    restart_with!(semi, sol)

    semi = Semidiscretization(particle_system, boundary_system,
                              neighborhood_search=GridNeighborhoodSearch)
    ode = semidiscretize(semi, tspan)

    saving_callback = SolutionSavingCallback(dt=0.02,
                                             prefix="$(correction_name)")
    callbacks = CallbackSet(info_callback, saving_callback)

    # See above for an explanation of the parameter choice
    global sol = solve(ode, RDPK3SpFSAL49(),
                       abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
                       reltol=1e-5, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
                       dtmax=1e-2, # Limit stepsize to prevent crashing
                       save_everystep=false, callback=callbacks)
end
