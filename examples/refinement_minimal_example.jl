using TrixiParticles
using OrdinaryDiffEq

tspan = (0.0, 1.0)
# ==========================================================================================
# ==== Resolution
particle_spacing = 0.1

boundary_layers = 3
spacing_ratio = 2.0

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (1.5, 1.5)
tank_size = (1.0, 1.0)

fluid_density = 1000.0

tank = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, false, false, false))

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
pressure = 1000.0
particle_refinement = ParticleRefinement(;
                                         refinement_pattern=TrixiParticles.HexagonalSplitting(),
                                         max_spacing_ratio=1.2)
fluid_system = EntropicallyDampedSPHSystem(tank.fluid, smoothing_kernel, smoothing_length,
                                           10.0; particle_refinement=particle_refinement)

# ==========================================================================================
# ==== Boundary
smoothing_length_boundary = 1.2 * particle_spacing / tank.spacing_ratio
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length_boundary)
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system, neighborhood_search=nothing)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

refinement_callback = ParticleRefinementCallback(interval=1)

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(),
                        refinement_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks);
