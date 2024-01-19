# Results are compared to the results in:
#
# P.N. Sun, D. Le Touzé, A.-M. Zhang.
# "Study of a complex fluid-structure dam-breaking benchmark problem using a multi-phase SPH method with APR".
# In: Engineering Analysis with Boundary Elements 104 (2019), pages 240-258.
# https://doi.org/10.1016/j.enganabound.2019.03.033

using TrixiParticles
using OrdinaryDiffEq
using PythonPlot
using JSON

# ==========================================================================================
# ==== Resolution
n_particles_y = 5

# ==========================================================================================
# ==== Experiment Setup
gravity = 2.0
tspan = (0.0, 5.0)

length_beam = 0.35
thickness = 0.02
clamp_radius = 0.05
material_density = 1000.0

# Young's modulus and Poisson ratio
E = 1.4e6
nu = 0.4

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = thickness / (n_particles_y - 1)

# Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius
fixed_particles = SphereShape(particle_spacing, clamp_radius + particle_spacing / 2,
                              (0.0, thickness / 2), material_density,
                              cutout_min=(0.0, 0.0), cutout_max=(clamp_radius, thickness),
                              tlsph=true)

n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)

# Beam and clamped particles
n_particles_per_dimension = (round(Int, length_beam / particle_spacing) +
                             n_particles_clamp_x + 1, n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for solids.
# We therefore need to pass `tlsph=true`.
beam = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (0.0, 0.0), density=material_density, tlsph=true)

solid = union(beam, fixed_particles)

# ==========================================================================================
# ==== Solid

smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

solid_system = TotalLagrangianSPHSystem(solid,
                                        smoothing_kernel, smoothing_length,
                                        E, nu, nothing,
                                        n_fixed_particles=nparticles(fixed_particles),
                                        acceleration=(0.0, -gravity))

# find points at the end of beam
beam_end_x = length_beam + clamp_radius
point_ids = []
for particle in TrixiParticles.eachparticle(solid_system)
    particle_coord = solid_system.current_coordinates[:, particle]

    if isapprox(particle_coord[1], beam_end_x, atol=1e-5)
        push!(point_ids, particle)
    end
end

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(solid_system, neighborhood_search=GridNeighborhoodSearch)
ode = semidiscretize(semi, tspan)

function particle_position(particle_id, pp, t, system, u, v, system_name)
    TrixiParticles.add_entry!(pp, "pos_x_$particle_id", t, system.current_coordinates[1, particle_id], system_name)
    TrixiParticles.add_entry!(pp, "pos_y_$particle_id", t, system.current_coordinates[2, particle_id], system_name)
end

point_pos_func = [(pp, t, sys, u, v, sys_name) -> particle_position(point_id, pp, t, sys, u, v, sys_name) for point_id in point_ids]
pp_callback = PostprocessCallback(point_pos_func, interval=500, filename="oscillating_beam_2d_positions")
info_callback = InfoCallback(interval=500)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback, pp_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks);
