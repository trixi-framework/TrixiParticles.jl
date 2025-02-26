using TrixiParticles
using OrdinaryDiffEq

# Domain size
domain_width = 0.5
domain_height = 1.0
no_particles = 50

# Particle spacing
particle_spacing = domain_width / no_particles  # Adjust for resolution

# Numerical settings
smoothing_length = 1.2 * particle_spacing
sound_speed = 20.0

# Fluid properties
fluid_density = 1.0
# No gravity
gravity = (0.0, 0.0)

# Time span
tspan = (0.0, 1.0)

rect_size = (domain_width, domain_width)

color0 = RectangularShape(particle_spacing,
                          round.(Int, rect_size ./ particle_spacing),
                          zeros(length(rect_size)), density=fluid_density)

color1 = RectangularShape(particle_spacing,
                          round.(Int, rect_size ./ particle_spacing),
                          (0.0, domain_width), density=fluid_density)

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1,
                                   clip_negative_pressure=true)

wcsph_color0 = WeaklyCompressibleSPHSystem(color0, SummationDensity(),
                                           state_equation, SchoenbergCubicSplineKernel{2}(),
                                           smoothing_length,
                                           reference_particle_spacing=particle_spacing,
                                           surface_tension=SurfaceTensionMorris(surface_tension_coefficient=1.0),
                                           color_value=0)

wcsph_color1 = WeaklyCompressibleSPHSystem(color1, SummationDensity(),
                                           state_equation, SchoenbergCubicSplineKernel{2}(),
                                           smoothing_length,
                                           reference_particle_spacing=particle_spacing,
                                           surface_tension=SurfaceTensionMorris(surface_tension_coefficient=1.0),
                                           color_value=1)

periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[domain_width, domain_height])
semi = Semidiscretization(wcsph_color0, wcsph_color1,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing,
                                                                        periodic_box=periodic_box))
ode = semidiscretize(semi, tspan, data_type=nothing)

info_callback = InfoCallback(interval=100)

solution_prefix = ""
saving_callback = SolutionSavingCallback(dt=0.02, prefix=solution_prefix)

# This can be overwritten with `trixi_include`
extra_callback = nothing

use_reinit = false
stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback, extra_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);

# # Create particle coordinates
# nx = Int(domain_width / particle_spacing) + 1
# ny = Int(domain_height / particle_spacing) + 1
# x_coords = range(0.0, stop=domain_width, length=nx)
# y_coords = range(0.0, stop=domain_height, length=ny)

# coordinates = [ [x, y] for y in y_coords, x in x_coords ]
# coordinates = reduce(vcat, coordinates)'

# # Assign colors based on y-coordinate
# colors = [ coord[2] > 0.5 ? 1 : 0 for coord in eachcol(coordinates) ]

# # Create particle properties
# n_particles = size(coordinates, 2)
# particle_mass = fluid_density * particle_spacing^2

# particles = Particles(
#     coordinates = coordinates,
#     velocities = zeros(2, n_particles),
#     masses = fill(particle_mass, n_particles),
#     densities = fill(fluid_density, n_particles),
#     colors = colors
# )

# # Initialize random velocities
# # Internal energy per particle
# internal_energy = 0.5 * sound_speed^2

# # Desired kinetic energy per particle
# desired_ke_per_particle = internal_energy * 1e-6

# # Generate random velocities
# using Random
# Random.seed!(1234)  # For reproducibility

# velocities = zeros(2, n_particles)
# for i in 1:n_particles
#     # Random velocity direction
#     theta = 2 * π * rand()
#     # Random velocity magnitude
#     v_mag = sqrt(2 * desired_ke_per_particle)
#     velocities[:, i] = v_mag * [cos(theta), sin(theta)]
# end

# # Assign velocities to particles
# particles.velocities = velocities

# # Use appropriate density calculator
# fluid_density_calculator = ContinuityDensity()

# # Exclude viscosity
# viscosity = nothing

# # Create the fluid system
# fluid_system = WeaklyCompressibleSPHSystem(
#     particles,
#     fluid_density_calculator,
#     StateEquationCole(
#         sound_speed = sound_speed,
#         reference_density = fluid_density,
#         exponent = 7.0,
#         clip_negative_pressure = false
#     ),
#     SmoothingKernelCubicSpline(),
#     smoothing_length,
#     viscosity = viscosity,
#     acceleration = gravity,
#     surface_tension = SurfaceTensionMomentumMorris()
# )

# # Define the periodic box matching the simulation domain
# periodic_box = PeriodicBox(
#     min_corner = [0.0, 0.0],
#     max_corner = [domain_width, domain_height]
# )

# # Configure the neighborhood search with the periodic box
# neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box)

# # Set up the semidiscretization and ODE problem
# semi = Semidiscretization(
#     fluid_system,
#     neighborhood_search = neighborhood_search
# )

# ode = semidiscretize(semi, tspan)

# # Define callbacks to record kinetic and internal energy
# kinetic_energies = Float64[]
# internal_energies = Float64[]
# times = Float64[]

# function record_energies!(integrator)
#     u = integrator.u
#     v = integrator.cache

#     # Compute kinetic energy
#     velocities = v.velocities
#     masses = particles.masses
#     ke = 0.5 * sum(masses .* sum(velocities .^ 2, dims=1))
#     push!(kinetic_energies, ke)

#     # Compute internal energy
#     densities = v.densities
#     pressures = v.pressures
#     ie = sum(pressures .* masses / (fluid_density * sound_speed^2))
#     push!(internal_energies, ie)

#     push!(times, integrator.t)
#     return
# end

# cb = CallbackSet(SavingCallback(record_energies!))

# # Time integration settings
# dtmax = 0.25 * particle_spacing / (sound_speed + maximum(abs.(velocities)))

# # Solve the ODE problem
# sol = solve(
#     ode,
#     RDPK3SpFSAL35(),
#     callback = cb,
#     save_everystep = false,
#     abstol = 1e-6,
#     reltol = 1e-6,
#     dtmax = dtmax
# )

# # Compute the ratio of kinetic energy to internal energy
# energy_ratios = kinetic_energies ./ internal_energies

# # Plot the ratio over time
# plot(
#     times, energy_ratios,
#     xlabel = "Time",
#     ylabel = "KE / Internal Energy",
#     title = "Parasitic Currents Test",
#     legend = false
# )
