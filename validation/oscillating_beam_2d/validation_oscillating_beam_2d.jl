# Results are compared to the results in:
#
# P.N. Sun, D. Le Touzé, A.-M. Zhang.
# "Study of a complex fluid-structure dam-breaking benchmark problem using a multi-phase SPH method with APR".
# In: Engineering Analysis with Boundary Elements 104 (2019), pages 240-258.
# https://doi.org/10.1016/j.enganabound.2019.03.033
# and
# Turek S , Hron J.
# "Proposal for numerical benchmarking of fluid-structure interaction between an elastic object and laminar incompressible flow."
# In: Fluid-structure interaction. Springer; 2006. p. 371–85 .
# https://doi.org/10.1007/3-540-34596-5_15

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Experiment Setup
gravity = 2.0
tspan = (0.0, 10.0)

elastic_plate_length = 0.35
elastic_plate_thickness = 0.02

cylinder_radius = 0.05
cylinder_diameter = 2 * cylinder_radius
material_density = 1000.0

E = 1.4e6
nu = 0.4

#resolution = [9, 21, 35]
resolution = [9]
for res in resolution
    n_particles_y = res
    particle_spacing = elastic_plate_thickness / (n_particles_y - 1)

    # Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius
    fixed_particles = SphereShape(particle_spacing, cylinder_radius + particle_spacing / 2,
                                  (0.0, elastic_plate_thickness / 2), material_density,
                                  cutout_min=(0.0, 0.0),
                                  cutout_max=(cylinder_radius, elastic_plate_thickness),
                                  tlsph=true)

    n_particles_clamp_x = round(Int, cylinder_radius / particle_spacing)

    # Beam and clamped particles
    n_particles_per_dimension = (round(Int, elastic_plate_length / particle_spacing) +
                                 n_particles_clamp_x + 1, n_particles_y)

    # Note that the `RectangularShape` puts the first particle half a particle spacing away
    # from the boundary, which is correct for fluids, but not for solids.
    # We therefore need to pass `tlsph=true`.
    elastic_plate = RectangularShape(particle_spacing, n_particles_per_dimension,
                                     (0.0, 0.0), density=material_density, tlsph=true)

    solid = union(elastic_plate, fixed_particles)

    # ==========================================================================================
    # ==== Solid

    smoothing_length = 2 * sqrt(2) * particle_spacing
    smoothing_kernel = WendlandC2Kernel{2}()

    solid_system = TotalLagrangianSPHSystem(solid,
                                            smoothing_kernel, smoothing_length,
                                            E, nu,
                                            n_fixed_particles=nparticles(fixed_particles),
                                            acceleration=(0.0, -gravity),
                                            penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

    # ==========================================================================================
    # ==== Postprocessing Setup

    # find points at the end of elastic plate
    plate_end_x = elastic_plate_length + cylinder_radius
    point_ids = []
    for particle in TrixiParticles.eachparticle(solid_system)
        particle_coord = solid_system.current_coordinates[:, particle]

        if isapprox(particle_coord[1], plate_end_x, atol=particle_spacing / 2)
            push!(point_ids, particle)
        end
    end

    # of those find the particle in the middle
    y_coords_at_plate_end = [solid_system.current_coordinates[2, particle]
                             for particle in point_ids]
    if isempty(y_coords_at_plate_end)
        error("No particles found at the specified beam_end_x coordinate.")
    end

    sorted_y_coords = sort(y_coords_at_plate_end)

    # Compute the median
    len = length(sorted_y_coords)
    if isodd(len)
        median_y = sorted_y_coords[ceil(Int, len / 2)]
    else
        half = round(Int, len / 2)
        median_y = (sorted_y_coords[half] + sorted_y_coords[half + 1]) / 2
    end
    closest_to_median_index = argmin(abs.(y_coords_at_plate_end .- median_y))
    middle_particle_id = point_ids[closest_to_median_index]

    function mid_point_x(t, v, u, system)
        return system.current_coordinates[1, middle_particle_id]
    end

    function mid_point_y(t, v, u, system)
        return system.current_coordinates[2, middle_particle_id]
    end

    pp_callback = PostprocessCallback(; mid_point_x, mid_point_y, dt=0.01,
                                      output_directory="out",
                                      filename="validation_run_oscillating_beam_2d_" *
                                               string(res), write_csv=false)
    info_callback = InfoCallback(interval=2500)
    saving_callback = SolutionSavingCallback(dt=0.5, prefix="validation_" * string(res))

    callbacks = CallbackSet(info_callback, saving_callback, pp_callback)

    # ==========================================================================================
    # ==== Simulation

    semi = Semidiscretization(solid_system, neighborhood_search=GridNeighborhoodSearch)
    ode = semidiscretize(semi, tspan)

    sol = solve(ode, RDPK3SpFSAL49(), abstol=1e-8, reltol=1e-6, dtmax=1e-3,
                save_everystep=false, callback=callbacks)
end
