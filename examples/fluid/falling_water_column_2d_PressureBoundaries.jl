using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.02

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.5, 1.0)
tank_size = (4.0, 4.0)

fluid_density = 1000.0
sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])
state_equation = nothing

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio)


for i in axes(tank.fluid.coordinates, 2)
    tank.fluid.coordinates[:, i] .+= [0.5 * tank_size[1] - 0.5 * initial_fluid_size[1], 0.2]
end

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Calculate kinematic viscosity for the viscosity model
nu = 0.02 * smoothing_length * sound_speed / 8
viscosity = ViscosityAdami(; nu)

# Use IISPH as fluid system
time_step = 1e-3
IISPH_system = ImplicitIncompressibleSPHSystem(tank.fluid, smoothing_kernel,
                                               smoothing_length, fluid_density,
                                               viscosity=ViscosityAdami(nu=nu),
                                               acceleration=(0.0, -gravity),
                                               min_iterations=10,
                                               max_iterations=30,
                                               time_step=time_step)


# ==========================================================================================
# ==== Boundary
boundary_density_calculator = PressureBoundaries(time_step)
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(IISPH_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, SymplecticEuler(),
        dt=time_step,
        save_everystep=false, callback=callbacks);
