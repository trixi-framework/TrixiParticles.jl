using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

particle_spacing = 0.005

# Ratio of fluid particle spacing to boundary particle spacing
beta = 1
boundary_layers = 3

water_density = 1000.0

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

tank_width = 0.96
tank_height = 0.48

water_width = tank_width
water_height = tank_height

sound_speed = 5.0
state_equation = StateEquationCole(sound_speed, 7, water_density, 1_000.0)
                                #    background_pressure=1000.0)

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

tank = RectangularTank(particle_spacing, (water_width, water_height),
                       (tank_width, tank_height), water_density,
                       n_layers=boundary_layers, spacing_ratio=beta,
                       faces=(false, false, true, true))

tank.fluid.coordinates[1, :] .-= 0.5 * particle_spacing
tank.boundary.coordinates[1, :] .-= 0.5 * particle_spacing
tank.fluid.velocity[1, :] .= 1.0

sphere = CircularShape(particle_spacing, 0.06, (0.2, 0.24), water_density,
                       shape_type=DrawCircle(n_layers=3, layer_inwards=true))


sphere2 = CircularShape(particle_spacing, 0.06, (0.2, 0.24), water_density,
                       shape_type=DrawCircle(n_layers=floor(Int, 0.06 / particle_spacing),
                       layer_inwards=true))

boundary = InitialCondition(tank.boundary, sphere)

function find_too_close_particles(coords1, coords2, max_distance)
    result = Int[]

    nhs = SpatialHashingSearch{2}(max_distance, size(coords2, 2))
    TrixiParticles.initialize!(nhs, coords2)

    TrixiParticles.for_particle_neighbor_serial(coords1, coords2, nhs) do i, j, _, _
        if !(i in result)
            append!(result, i)
        end
    end

    return result
end

too_close_particles = find_too_close_particles(tank.fluid.coordinates, sphere2.coordinates,
                                               0.8particle_spacing)
valid_particles = filter(i -> !(i in too_close_particles), axes(tank.fluid.coordinates, 2))

fluid_coordinates = tank.fluid.coordinates[:, valid_particles]
fluid_velocity = tank.fluid.velocity[:, valid_particles]
fluid_mass = tank.fluid.mass[valid_particles]
fluid_density = tank.fluid.density[valid_particles]

fluid = InitialCondition(fluid_coordinates, fluid_velocity, fluid_mass, fluid_density)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(boundary.density,
                                             boundary.mass, state_equation,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel,
                                             smoothing_length)

# K = 9.81 * water_height
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta,
#                                              boundary.mass)

# ==========================================================================================
# ==== Systems

fluid_system = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(), state_equation,
                                           smoothing_kernel, smoothing_length,
                                           viscosity=viscosity)

boundary_system = BoundarySPHSystem(boundary.coordinates, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
