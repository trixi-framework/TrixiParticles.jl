"""
2D dam break simulation based on S. Marrone et al. (2011).
"δ-SPH model for simulating violent impact flows".
In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
https://doi.org/10.1016/J.CMA.2010.12.016
"""

using TrixiParticles
using OrdinaryDiffEq

const GRAVITY = 9.81

"""
Set up the fluid properties and tank configuration.
"""
function setup_simulation()
    # Fluid properties
    PARTICLE_SPACING = 0.05
    BETA = 1
    BOUNDARY_LAYERS = 3
    WATER_WIDTH = 2.0
    WATER_HEIGHT = 1.0
    WATER_DENSITY = 1000.0

    # Tank properties
    TANK_WIDTH = floor(5.366 / PARTICLE_SPACING * BETA) * PARTICLE_SPACING / BETA
    TANK_HEIGHT = 4

    SOUND_SPEED = 20 * sqrt(GRAVITY * WATER_HEIGHT)
    SMOOTHING_LENGTH = 1.15 * PARTICLE_SPACING

    smoothing_kernel = SchoenbergQuarticSplineKernel{2}()

    state_equation = StateEquationCole(SOUND_SPEED, 7, WATER_DENSITY, 100000.0,
                                       background_pressure=100000.0)
    viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

    # Tank setup
    tank = RectangularTank(PARTICLE_SPACING, (WATER_WIDTH, WATER_HEIGHT),
                           (TANK_WIDTH, TANK_HEIGHT), WATER_DENSITY,
                           n_layers=BOUNDARY_LAYERS, spacing_ratio=BETA)

    # Move right boundary
    # Recompute the new water column width since the width has been rounded in `RectangularTank`.
    new_wall_position = (tank.n_particles_per_dimension[1] + 1) * PARTICLE_SPACING
    move_wall(tank, new_wall_position)

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                 tank.boundary.mass, state_equation,
                                                 SummationDensity(), smoothing_kernel,
                                                 SMOOTHING_LENGTH)

    fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, SummationDensity(),
                                               state_equation,
                                               smoothing_kernel, SMOOTHING_LENGTH,
                                               viscosity=viscosity,
                                               acceleration=(0.0, -GRAVITY),
                                               correction=ShepardKernelCorrection())

    bnd_system = BoundarySPHSystem(tank.boundary.coordinates, boundary_model)

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
function run(semi, tspan::Tuple{Real, Real}, prefix)
    ode = semidiscretize(semi, tspan)

    info_callback = InfoCallback(interval=100)
    saving_callback_correction = SolutionSavingCallback(dt=0.02, prefix=prefix)
    callbacks_correction = CallbackSet(info_callback, saving_callback_correction)

    sol = solve(ode, RDPK3SpFSAL49(),
                abstol=1e-6, reltol=1e-5, dtmax=1e-2,
                save_everystep=false, callback=callbacks_correction)

    return sol
end

# Setup simulation
fluid_system, bnd_system, tank = setup_simulation()

# Run relaxation step
tspan_relaxation = (0.0, 3.0)
semi = Semidiscretization(fluid_system, bnd_system,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-5)

sol_relaxation = run(semi, tspan_relaxation, "relaxation")

# Move right boundary
move_wall(tank, tank.tank_size[1])

restart_with!(semi, sol_relaxation)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(GRAVITY))
sol = run(semi, tspan, "simulation")
