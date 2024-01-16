using TrixiParticles
using OrdinaryDiffEq
# using PyPlot

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.02

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (2.0, 0.9)
tank_size = (2.0, 1.0)

fluid_density = 1000.0
sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary

# to set a no-slip boundary condition a wall visosity needs to be defined
alpha = 0.2
viscosity_wall = ViscosityAdami(nu=alpha * smoothing_length * sound_speed / 8)

boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             viscosity=viscosity_wall)

# Uncomment to use repulsive boundary particles by Monaghan & Kajtar.
# Also change spacing ratio and boundary layers (see comment above).
# K = gravity * initial_fluid_size[2]
# boundary_model = BoundaryModelMonaghanKajtar(K, spacing_ratio, boundary_particle_spacing,
#                                              tank.boundary.mass)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# # Example for using interpolation
# #######################################################################################

# # interpolate_point can be used to interpolate the properties of the 'fluid_system' with the original kernel and smoothing_length
# println(interpolate_point([1.0, 0.01], semi, fluid_system, sol))
# # or with an increased smoothing_length smoothing the result
# println(interpolate_point([1.0, 0.01], semi, fluid_system, sol,
#                           smoothing_length=2.0 * smoothing_length))

# # a point outside of the domain will result in properties with value 0
# # on the boundary a result can still be obtained
# println(interpolate_point([1.0, 0.0], semi, fluid_system, sol))
# # slightly befind the result is 0
# println(interpolate_point([1.0, -0.01], semi, fluid_system, sol))

# # multiple points can be interpolated by providing an array
# println(interpolate_point([
#                               [1.0, 0.01],
#                               [1.0, 0.1],
#                               [1.0, 0.0],
#                               [1.0, -0.01],
#                               [1.0, -0.05],
#                           ], semi, fluid_system, sol))

# # # it is also possible to interpolate along a line
# result = interpolate_line([0.5, -0.05], [0.5, 1.0], 20, semi, fluid_system, sol)
# result_endpoint = interpolate_line([0.5, -0.05], [0.5, 1.0], 20, semi, fluid_system, sol,
#                                    endpoint=false)
# # Extract densities and coordinates for plotting
# densities = [r.density for r in result]
# coords = [r.coord[2] for r in result]  # Assuming you want to plot against the y-coordinate

# densities_endpoint = [r.density for r in result_endpoint]
# coords_endpoint = [r.coord[2] for r in result_endpoint]

# Create the plot
# figure()
# plot(coords, densities, marker="o", linestyle="-", label="With Endpoint")
# plot(coords_endpoint, densities_endpoint, marker="x", linestyle="--",
#      label="Without Endpoint")

# # Add labels and legend
# xlabel("Y-Coordinate")
# ylabel("Density")
# title("Density Interpolation Along a Line")
# legend()

# # Display the plot
# show()

# v_x = [r.velocity[1] for r in result]
# v_x_endpoint = [r.velocity[1] for r in result_endpoint]

# figure()
# plot(v_x, coords, marker="o", linestyle="-", label="With Endpoint")
# plot(v_x_endpoint, coords_endpoint, marker="x", linestyle="--",
#      label="Without Endpoint")

# # Add labels and legend
# xlabel("Velocity-X")
# ylabel("Y-Coordinate")
# title("Velocity Interpolation Along a Line")
# legend()

# # Display the plot
# show()

# plane = interpolate_plane([0.0, 0.0], [1.0, 1.0], 0.005, semi, fluid_system, sol)

# # Extracting densities and coordinates for plotting
# densities = [p.density for p in plane]
# x_coords = [p.coord[1] for p in plane]
# y_coords = [p.coord[2] for p in plane]

# # Create the scatter plot
# scatter(x_coords, y_coords, c=densities, cmap="viridis", marker="o")

# # Add colorbar, labels, and title
# colorbar(label="Density")
# xlabel("X Coordinate")
# ylabel("Y Coordinate")
# title("Density Distribution in Plane")

# # Show the plot
# show()

# plane = interpolate_plane([0.0, 0.0], [1.0, 1.0], 0.005, semi, fluid_system, sol,
#                           smoothing_length=2.0 * smoothing_length)

# # Extracting densities and coordinates for plotting
# densities = [p.density for p in plane]
# x_coords = [p.coord[1] for p in plane]
# y_coords = [p.coord[2] for p in plane]

# # Create the scatter plot
# scatter(x_coords, y_coords, c=densities, cmap="viridis", marker="o")

# # Add colorbar, labels, and title
# colorbar(label="Density")
# xlabel("X Coordinate")
# ylabel("Y Coordinate")
# title("Density Distribution in Plane")

# # Show the plot
# show()

# # Extracting densities and coordinates for plotting
# velocity = [p.velocity[1] for p in plane]

# # Create the scatter plot
# scatter(x_coords, y_coords, c=velocity, cmap="viridis", marker="o")

# # Add colorbar, labels, and title
# colorbar(label="velocity_x")
# xlabel("X Coordinate")
# ylabel("Y Coordinate")
# title("Velocity Distribution in Plane")

# # Show the plot
# show()
