# ==========================================================================================
# 2D Oscillating Elastic Beam (Cantilever) Simulation
#
# This example simulates the oscillation of a 2D elastic beam (cantilever)
# clamped at one end and subjected to gravity. It uses the Total Lagrangian SPH (TLSPH)
# method for structure mechanics.
#
# Based on:
# J. O'Connor and B.D. Rogers
# "A fluid-structure interaction model for free-surface flows and
# flexible structures using smoothed particle hydrodynamics on a GPU",
# Journal of Fluids and Structures, Volume 104, 2021.
# DOI: 10.1016/j.jfluidstructs.2021.103312
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
n_particles_y = 5

# ==========================================================================================
# ==== Experiment Setup
gravity = 2.0
tspan = (0.0, 5.0)

elastic_beam = (length=0.35, thickness=0.02)
material = (density=1000.0, E=1.4e6, nu=0.4)
clamp_radius = 0.05

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = elastic_beam.thickness / (n_particles_y - 1)

# Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius
clamped_particles = SphereShape(particle_spacing, clamp_radius + particle_spacing / 2,
                                (0.0, elastic_beam.thickness / 2), material.density,
                                cutout_min=(0.0, 0.0),
                                cutout_max=(clamp_radius, elastic_beam.thickness),
                                place_on_shell=true)

n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)

# Beam and clamped particles
n_particles_per_dimension = (round(Int, elastic_beam.length / particle_spacing) +
                             n_particles_clamp_x + 1, n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for structures.
# We therefore need to pass `place_on_shell=true`.
beam = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (0.0, 0.0), density=material.density, place_on_shell=true)

structure = union(clamped_particles, beam)

# ==========================================================================================
# ==== Structure
smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

structure_system = TotalLagrangianSPHSystem(structure, smoothing_kernel, smoothing_length,
                                            material.E, material.nu,
                                            clamped_particles=1:nparticles(clamped_particles),
                                            acceleration=(0.0, -gravity),
                                            penalty_force=nothing, viscosity=nothing,
                                            clamped_particles_motion=nothing)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(structure_system,
                          neighborhood_search=PrecomputedNeighborhoodSearch{2}(),
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=1000)

# Track the position of the particle in the middle of the tip of the beam.
middle_particle_id = Int(n_particles_per_dimension[1] * (n_particles_per_dimension[2] + 1) /
                         2)

# Make these constants because global variables in the functions below are slow
const STARTPOSITION_X = beam.coordinates[1, middle_particle_id]
const STARTPOSITION_Y = beam.coordinates[2, middle_particle_id]

function deflection_x(system, data, t)
    return data.coordinates[1, middle_particle_id] - STARTPOSITION_X
end

function deflection_y(system, data, t)
    return data.coordinates[2, middle_particle_id] - STARTPOSITION_Y
end

saving_callback = SolutionSavingCallback(dt=0.02, prefix="",
                                         deflection_x=deflection_x,
                                         deflection_y=deflection_y)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks);
