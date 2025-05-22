# ==========================================================================================
# 2D Oscillating Elastic Beam (Cantilever) Simulation
#
# This example simulates the oscillation of a 2D elastic beam (cantilever)
# fixed at one end and subjected to gravity. It uses the Total Lagrangian SPH (TLSPH)
# method for solid mechanics.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters (Defaults, can be overridden by `trixi_include`)
# ------------------------------------------------------------------------------
# Resolution: Number of particles across the beam's thickness (y-direction).
# `particle_spacing` will be derived from this and `elastic_beam_thickness`.
num_particles_beam_thickness_y = 5 # Default if not overridden

# Physical Parameters
gravity_magnitude = 2.0 # m/s^2
# `system_acceleration_vec` can be overridden for different gravity directions.
system_acceleration_vec = (0.0, -gravity_magnitude)
simulation_tspan = (0.0, 5.0) # seconds

# Beam Geometry and Material Properties
# Using a NamedTuple for cleaner grouping, accessible via `elastic_beam.length`, etc.
elastic_beam_properties = (length=0.35, # meters
                           thickness=0.02 # meters
                           )
material_properties = (density=1000.0, # kg/m^3
                       E=1.4e6,      # Young's modulus (Pa)
                       nu=0.4        # Poisson's ratio (dimensionless)
                       )

# Clamp: Radius of the circular region where particles are fixed.
# This creates the cantilever boundary condition.
clamp_fixture_radius = 0.05 # meters

# ------------------------------------------------------------------------------
# Particle Setup: Beam and Clamped Region
# ------------------------------------------------------------------------------
# Particle spacing is determined by the beam's thickness and the number of particles across it.
# This `particle_spacing` will be used for both the beam and the clamped particles.
particle_spacing = elastic_beam_properties.thickness / (num_particles_beam_thickness_y - 1)

# Create fixed particles for the clamp.
# These particles form a quarter-circle shape at the beam's root (x=0).
# `SphereShape` with `cutout_min/max` can create such shapes.
# `tlsph=true` ensures particles are placed precisely on the boundary for solids.
# Add `particle_spacing / 2` to `clamp_radius` to ensure particles fill the clamp area.
fixed_particles_clamp = SphereShape(particle_spacing,
                                    clamp_fixture_radius + particle_spacing / 2,
                                    (0.0, elastic_beam_properties.thickness / 2), # Center of the sphere for cutout
                                    material_properties.density,
                                    cutout_min=(0.0, 0.0), # Keep only the positive x, positive y quadrant
                                    cutout_max=(clamp_fixture_radius + particle_spacing, # Extend cutout slightly
                                                elastic_beam_properties.thickness + particle_spacing),
                                    tlsph=true)

# Number of particles along the x-direction for the clamped region.
num_particles_in_clamp_x = round(Int, clamp_fixture_radius / particle_spacing)

# Create particles for the main elastic beam.
# The beam extends from x=0 to x=`elastic_beam.length`.
# The total number of particles in x includes those overlapping with the clamp region
# and one extra to ensure the full length is covered.
num_particles_beam_length_x = round(Int, elastic_beam_properties.length / particle_spacing) +
                              num_particles_in_clamp_x + 1
total_beam_particles_dims = (num_particles_beam_length_x, num_particles_beam_thickness_y)

# Create the rectangular beam shape. `tlsph=true` is important for solid mechanics.
beam_shape_particles = RectangularShape(particle_spacing,
                                        total_beam_particles_dims,
                                        (0.0, 0.0), # Origin at (0,0)
                                        density=material_properties.density,
                                        tlsph=true)

# Combine the beam shape particles and the explicitly defined fixed clamp particles.
# `union` ensures unique particles if there's overlap, using properties from the first argument.
# The `TotalLagrangianSPHSystem` will later identify which of these total `solid_particles`
# are fixed based on `n_fixed_particles` or by comparing with `fixed_particles_clamp`.
solid_particles = union(beam_shape_particles, fixed_particles_clamp)

# Identify the number of fixed particles (those belonging to the clamp).
# This is crucial for the `TotalLagrangianSPHSystem`.
num_total_fixed_particles = nparticles(fixed_particles_clamp)

# ------------------------------------------------------------------------------
# Solid System Setup (Total Lagrangian SPH)
# ------------------------------------------------------------------------------
# Smoothing length for the solid domain.
# `sqrt(2) * particle_spacing` is a common choice for TLSPH with 2D Wendland kernels.
solid_smoothing_length = sqrt(2) * particle_spacing
solid_smoothing_kernel = WendlandC2Kernel{2}() # Common kernel for solids

# Define the TLSPH system for the elastic beam.
# `penalty_force=nothing` means no additional penalty for inter-particle contact within the solid,
# as TLSPH handles internal forces.
solid_beam_system = TotalLagrangianSPHSystem(solid_particles,
                                             solid_smoothing_kernel, solid_smoothing_length,
                                             material_properties.E, material_properties.nu,
                                             n_fixed_particles=num_total_fixed_particles,
                                             acceleration=system_acceleration_vec,
                                             penalty_force=nothing,
                                             reference_particle_spacing=particle_spacing)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
# Neighborhood search: `PrecomputedNeighborhoodSearch` is suitable for solids
# where particle topology doesn't change (Total Lagrangian formulation).
neighborhood_search_solid = PrecomputedNeighborhoodSearch{2}() # For 2D

semi = Semidiscretization(solid_beam_system,
                          neighborhood_search=neighborhood_search_solid,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, simulation_tspan)

# ------------------------------------------------------------------------------
# Callbacks for Monitoring and Output
# ------------------------------------------------------------------------------
info_callback = InfoCallback(interval=1000) # Print info every 1000 steps

# --- Tracking Tip Deflection ---
# Identify the ID of a particle at the tip of the beam to track its deflection.
# This calculation assumes a regular grid and might need adjustment if particle
# generation is more complex or if `union` reorders particles significantly.
# A robust way is to find the particle with max-x and mid-y coordinate initially.
# Original heuristic: middle particle in the last column.
# Particle ID for the middle particle at the beam's tip.
# (total_beam_particles_dims[1] is number of particles along length,
#  total_beam_particles_dims[2] is number of particles along thickness (y))
tip_particle_column_index = total_beam_particles_dims[1] # Last column of originally generated beam particles
tip_particle_row_index = round(Int, (total_beam_particles_dims[2] + 1) / 2) # Middle row

# Find the actual particle ID in the `solid_particles` set that corresponds to this conceptual tip particle.
# This requires finding the particle from `beam_shape_particles` that matches this logical position
# and then finding its ID within the combined `solid_particles` set.
# For simplicity, the original heuristic for `middle_particle_id` is used,
# but acknowledge it might not be robust if `union` significantly reorders or `fixed_particles_clamp`
# adds particles in a way that shifts indices non-trivially from `beam_shape_particles`.
# A safer approach would be to find it by coordinate query on `solid_particles.coordinates`
# based on the expected initial tip position.

# Heuristic from original file (assumes `beam_shape_particles` are first in `solid_particles`):
# Find initial coordinates of the conceptual tip particle from `beam_shape_particles`
initial_tip_x = (total_beam_particles_dims[1] - 0.5) * particle_spacing # x-coord of center of last column particle
initial_tip_y = (tip_particle_row_index - 0.5) * particle_spacing      # y-coord of center of middle row particle
initial_tip_position = SVector(initial_tip_x, initial_tip_y)

# Find the particle in `solid_particles` closest to this initial tip position.
min_dist_sq = Inf
tracked_tip_particle_id = -1
for i in 1:nparticles(solid_particles)
    dist_sq = sum((solid_particles.coordinates[:, i] .- initial_tip_position).^2)
    if dist_sq < min_dist_sq
        min_dist_sq = dist_sq
        tracked_tip_particle_id = i
    end
end
if tracked_tip_particle_id == -1
    error("Could not identify tip particle for deflection tracking.")
end
println("Tracking tip deflection of particle ID: $tracked_tip_particle_id")

# Initial position of the tracked tip particle. Use `const` for performance in callback functions.
const INITIAL_TIP_POS_X = solid_particles.coordinates[1, tracked_tip_particle_id]
const INITIAL_TIP_POS_Y = solid_particles.coordinates[2, tracked_tip_particle_id]

# Callback functions to compute deflection relative to initial position.
# Arguments: `system` (here, `solid_beam_system`),
#            `data_ode` (current state vector from OrdinaryDiffEq.jl solution object, containing `coordinates`),
#            `t` (current simulation time).
# Note: The original example used `data.coordinates`, assuming `data` is the system's particle data.
#       The `SolutionSavingCallback` actually passes `v_ode, u_ode, t, system`.
#       We need to reconstruct the current coordinates from `u_ode` or `v_ode`.
#       `current_coordinates = TrixiParticles.current_coordinates(u_ode, system)`
function deflection_x_tip(v_ode, u_ode, t, system) # Corrected signature
    current_coords = TrixiParticles.current_coordinates(u_ode, system)
    return current_coords[1, tracked_tip_particle_id] - INITIAL_TIP_POS_X
end

function deflection_y_tip(v_ode, u_ode, t, system) # Corrected signature
    current_coords = TrixiParticles.current_coordinates(u_ode, system)
    return current_coords[2, tracked_tip_particle_id] - INITIAL_TIP_POS_Y
end

# SolutionSavingCallback to save solution files and custom deflection data.
# `dt=0.02` saves output every 0.02 simulation time units.
# Custom quantities `deflection_x_tip` and `deflection_y_tip` will be computed and saved.
saving_callback = SolutionSavingCallback(dt=0.02, prefix="oscillating_beam_2d",
                                         deflection_x_tip=deflection_x_tip,
                                         deflection_y_tip=deflection_y_tip)

callbacks = CallbackSet(info_callback, saving_callback)

# ------------------------------------------------------------------------------
# Solve the ODE System
# ------------------------------------------------------------------------------
# Use a Runge-Kutta method with adaptive time stepping.
# `RDPK3SpFSAL49` is a common choice for SPH/TLSPH.
sol = solve(ode, RDPK3SpFSAL49(),
            save_everystep=false, # Save only at times specified by `saving_callback`
            callback=callbacks)

println("Oscillating beam 2D simulation finished. Check 'out/' directory for VTP and CSV files.")

# To plot deflection after the simulation:
# using CSV, DataFrames, Plots
# data = CSV.read("out/oscillating_beam_2d.csv", DataFrame)
# plot(data.time, data.deflection_y_tip_solid_particles_1, label="Tip Deflection (Y)", xlabel="Time (s)", ylabel="Deflection (m)")
