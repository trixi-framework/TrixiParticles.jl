# ==========================================================================================
# 2D Oscillating Elastic Beam (Cantilever) Simulation
#
# This example simulates the oscillation of a 2D elastic beam (cantilever)
# glued at one end to a rigid cylinder and subjected to gravity.
# It uses the Total Lagrangian SPH (TLSPH) method for structure mechanics.
#
# This is the CSM3 benchmark by Turek and Hron (2006):
# S. Turek and J. Hron.
# "Proposal for numerical benchmarking of fluid-structure interaction between an elastic
# object and laminar incompressible flow".
# In: Fluid-Structure Interaction, Springer, 2006, pp. 371-385.
# https://doi.org/10.1007/3-540-34596-5_15
#
# See `validation/oscillating_beam_2d` for a validation of this setup against the
# reference solution and for the setup of O'Connor and Rogers (2021).
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqSymplecticRK

# ==========================================================================================
# ==== Resolution
# Number of particles across the beam thickness.
n_particles_y = 5

# ==========================================================================================
# ==== Experiment Setup
gravity = 2.0
tspan = (0.0, 5.0)

elastic_beam = (length=0.35, thickness=0.02)
material = (density=1000.0, E=1.4e6, nu=0.4)
cylinder_radius = 0.05

# In the setup, the elastic beam is glued to a rigid cylinder.
# Since it is rigid, the part of the cylinder extending past the beam vertically
# is negligible, so we just extend the beam into the imaginary cylinder
# and clamp the particles that lie inside the cylinder.
#
# Note that the beam is attached to the curved cylinder surface at
# `x = sqrt(cylinder_radius^2 - (thickness / 2)^2)`, so the free part of the beam
# is slightly longer than `elastic_beam.length`.
cylinder_center = (0.0, elastic_beam.thickness / 2)
tip_x = cylinder_radius + elastic_beam.length

particle_spacing = elastic_beam.thickness / n_particles_y

# The beam starts at the center of the cylinder and ends at the tip.
# Note that the `RectangularShape` places the particles cell-centered, i.e., the first and
# last particle are half a particle spacing away from the boundary. This way, the particle
# volumes fill exactly the beam with the given length and thickness.
n_particles_per_dimension = (round(Int, tip_x / particle_spacing), n_particles_y)
structure = RectangularShape(particle_spacing, n_particles_per_dimension, (0.0, 0.0),
                             density=material.density, coordinates_eltype=Float64)

# All particles inside the cylinder are clamped and represent the rigid cylinder.
clamped_particles = findall(particle -> (structure.coordinates[1, particle] -
                                         cylinder_center[1])^2 +
                                        (structure.coordinates[2, particle] -
                                         cylinder_center[2])^2 < cylinder_radius^2,
                            eachparticle(structure))

# ==========================================================================================
# ==== Structure
smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

structure_system = TotalLagrangianSPHSystem(structure; smoothing_kernel, smoothing_length,
                                            young_modulus=material.E,
                                            poisson_ratio=material.nu,
                                            clamped_particles,
                                            acceleration=(0.0, -gravity),
                                            penalty_force=PenaltyForceGanzenmueller(alpha=0.01),
                                            viscosity=nothing,
                                            clamped_particles_motion=nothing,
                                            self_interaction_nhs=:default)

# ==========================================================================================
# ==== Simulation
# Note that the `neighborhood_search` passed here is not used if the simulation
# consists of a single `TotalLagrangianSPHSystem`.
# Instead, the neighborhood search passed to the `TotalLagrangianSPHSystem` is used.
semi = Semidiscretization(structure_system,
                          neighborhood_search=nothing,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=1000)

saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Pure TLSPH simulations without a fluid can use `VelocityVerlet` with a time step based on
# the CFL condition for maximum performance. This is about 1.7x faster than `RDPK3SpFSAL49`
# with the same results. For fluid-structure interaction, use a Runge-Kutta method.
stepsize_callback = StepsizeCallback(cfl=0.7)

# `dt` is overwritten by the `StepsizeCallback`.
sol = solve(ode, VelocityVerlet(), dt=1.0, maxiters=10^8, save_everystep=false,
            callback=CallbackSet(callbacks, stepsize_callback));
