# 2D dam break simulation based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

using TrixiParticles
using OrdinaryDiffEq

const GRAVITY = 9.81

# Fluid properties
const WATER_DENSITY = 1000.0
const WATER_WIDTH = 2.0
const WATER_HEIGHT = 1.0
const SOUND_SPEED = 20 * sqrt(GRAVITY * WATER_HEIGHT)

# Numerical settings
const PARTICLE_SPACING = 0.05
const SMOOTHING_LENGTH = 1.15 * PARTICLE_SPACING
const BOUNDARY_LAYERS = 3

# Tank properties
const TANK_WIDTH = floor(5.366 / PARTICLE_SPACING) * PARTICLE_SPACING
const TANK_HEIGHT = 4

"""
Set up the fluid properties and tank configuration.
"""
function setup_simulation(density_calculator, correction_method)
    smoothing_kernel = SchoenbergQuarticSplineKernel{2}()

    state_equation = StateEquationCole(SOUND_SPEED, 7, WATER_DENSITY, 100000.0,
                                       background_pressure=100000.0)
    viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

    # Tank setup
    tank = RectangularTank(PARTICLE_SPACING, (WATER_WIDTH, WATER_HEIGHT),
                           (TANK_WIDTH, TANK_HEIGHT), WATER_DENSITY,
                           n_layers=BOUNDARY_LAYERS)

    # Move right boundary
    # Recompute the new water column width since the width has been rounded in `RectangularTank`.
    new_wall_position = (tank.n_particles_per_dimension[1] + 1) * PARTICLE_SPACING
    move_wall(tank, new_wall_position)

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                 tank.boundary.mass, state_equation,
                                                 SummationDensity(), smoothing_kernel,
                                                 SMOOTHING_LENGTH)

    fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, density_calculator,
                                               state_equation,
                                               smoothing_kernel, SMOOTHING_LENGTH,
                                               viscosity=viscosity,
                                               acceleration=(0.0, -GRAVITY),
                                               correction=correction_method)

    bnd_system = BoundarySPHSystem(tank.boundary, boundary_model)

    return fluid_system, bnd_system, tank
end

function move_wall(tank, new_wall_position)
    reset_faces = (false, true, false, false)
    positions = (0, new_wall_position, 0, 0)
    reset_wall!(tank, reset_faces, positions)
end

"""
Run simulation.
"""
function run(semi, tspan::Tuple{Real, Real}, prefix; density_reinit=false)
    ode = semidiscretize(semi, tspan)

    info_callback = InfoCallback(interval=100)
    saving_callback_correction = SolutionSavingCallback(dt=0.02, prefix=prefix)
    density_reinit_cb = density_reinit ?
                        DensityReinitializationCallback(semi.systems[1], dt=0.05) : nothing
    callbacks_correction = density_reinit ?
                           CallbackSet(info_callback, saving_callback_correction,
                                       density_reinit_cb) :
                           CallbackSet(info_callback, saving_callback_correction)

    sol = solve(ode, RDPK3SpFSAL49(),
                abstol=1e-6, reltol=1e-5, dtmax=1e-2,
                save_everystep=false, callback=callbacks_correction)

    return sol
end

correction_dict = Dict("no_correction" => Nothing(),
                       "shepard_kernel_correction" => ShepardKernelCorrection(),
                       "akinci_free_surf_correction" => AkinciFreeSurfaceCorrection(WATER_DENSITY),
                       "kernel_gradient_summation_correction" => KernelGradientCorrection(),
                       "kernel_gradient_continuity_correction" => KernelGradientCorrection(),
                       "density_reinit" => Nothing())

density_calculator_dict = Dict("no_correction" => SummationDensity(),
                               "shepard_kernel_correction" => SummationDensity(),
                               "akinci_free_surf_correction" => SummationDensity(),
                               "kernel_gradient_summation_correction" => SummationDensity(),
                               "kernel_gradient_continuity_correction" => ContinuityDensity(),
                               "density_reinit" => ContinuityDensity())

reinit_dict = Dict("no_correction" => false,
                   "shepard_kernel_correction" => false,
                   "akinci_free_surf_correction" => false,
                   "kernel_gradient_summation_correction" => false,
                   "kernel_gradient_continuity_correction" => false,
                   "density_reinit" => true)

for correction_name in keys(correction_dict)
    density_calculator = density_calculator_dict[correction_name]
    correction_method = correction_dict[correction_name]

    fluid_system, bnd_system, tank = setup_simulation(density_calculator, correction_method)

    semi = Semidiscretization(fluid_system, bnd_system,
                              neighborhood_search=SpatialHashingSearch,
                              damping_coefficient=1e-5)

    # Run relaxation step
    tspan_relaxation = (0.0, 3.0)
    sol_relaxation = run(semi, tspan_relaxation, "$(correction_name)_relaxation",
                         density_reinit=reinit_dict[correction_name])

    # Move right boundary
    move_wall(tank, tank.tank_size[1])
    # set relaxation result within the systems
    restart_with!(semi, sol_relaxation)
    # reinitialize the neighborhood search
    semi = Semidiscretization(fluid_system, bnd_system,
                              neighborhood_search=SpatialHashingSearch,
                              damping_coefficient=1e-5)

    # Run full simulation
    tspan = (0.0, 5.7 / sqrt(GRAVITY))
    global sol = run(semi, tspan, "$(correction_name)_simulation",
                     density_reinit=reinit_dict[correction_name])
end
