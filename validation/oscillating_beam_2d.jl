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
using PythonPlot
using JSON
using Glob
using CSV
using DataFrames
using Interpolations
using Statistics
using Printf

# ==========================================================================================
# ==== Experiment Setup
gravity = 2.0
tspan = (0.0, 10.0)

elastic_plate_length = 0.35
elastic_plate_thickness = 0.02

cylinder_radius = 0.05
cylinder_diameter = 2 * cylinder_radius
material_density = 1000.0

# Young's modulus and Poisson ratio
E = 1.4e6
nu = 0.4

resolution = [20, 50, 100]
for res in resolution

    particle_spacing = cylinder_diameter/res


    # Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius
    fixed_particles = SphereShape(particle_spacing, cylinder_radius + particle_spacing / 2,
                                (0.0, elastic_plate_thickness / 2), material_density,
                                cutout_min=(0.0, 0.0), cutout_max=(cylinder_radius, elastic_plate_thickness),
                                tlsph=true)

    n_particles_clamp_x = round(Int, cylinder_radius / particle_spacing)

    # Beam and clamped particles
    n_particles_per_dimension = (round(Int, elastic_plate_length / particle_spacing) +
                                n_particles_clamp_x + 1, round(Int, elastic_plate_thickness/particle_spacing)+1)

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
                                            E, nu, nothing,
                                            n_fixed_particles=nparticles(fixed_particles),
                                            acceleration=(0.0, -gravity), penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

    # find points at the end of elastic plate
    plate_end_x = elastic_plate_length + cylinder_radius
    point_ids = []
    for particle in TrixiParticles.eachparticle(solid_system)
        particle_coord = solid_system.current_coordinates[:, particle]

        if isapprox(particle_coord[1], plate_end_x, atol=particle_spacing/2)
            push!(point_ids, particle)
        end
    end

    # of those find the particle in the middle
    y_coords_at_plate_end = [solid_system.current_coordinates[2, particle] for particle in point_ids]
    if isempty(y_coords_at_plate_end)
        error("No particles found at the specified beam_end_x coordinate.")
    end

    sorted_y_coords = sort(y_coords_at_plate_end)

    # Compute the median
    len = length(sorted_y_coords)
    if isodd(len)
        median_y = sorted_y_coords[ceil(Int, len / 2)]
    else
        half = round(Int, len/2)
        median_y = (sorted_y_coords[half] + sorted_y_coords[half + 1]) / 2
    end
    closest_to_median_index = argmin(abs.(y_coords_at_plate_end .- median_y))
    middle_particle_id = point_ids[closest_to_median_index]

    # ==========================================================================================
    # ==== Simulation
    semi = Semidiscretization(solid_system, neighborhood_search=GridNeighborhoodSearch)
    ode = semidiscretize(semi, tspan)

    function particle_position(particle_id, pp, t, system, u, v, system_name)
        TrixiParticles.add_entry!(pp, "pos_x_$particle_id", t, system.current_coordinates[1, particle_id], system_name)
        TrixiParticles.add_entry!(pp, "pos_y_$particle_id", t, system.current_coordinates[2, particle_id], system_name)
    end

    #point_pos_func = [(pp, t, sys, u, v, sys_name) -> particle_position(point_id, pp, t, sys, u, v, sys_name) for point_id in point_ids]
    point_pos_func = (pp, t, sys, u, v, sys_name) -> particle_position(middle_particle_id, pp, t, sys, u, v, sys_name)
    pp_callback = PostprocessCallback(point_pos_func, dt=0.025, filename="oscillating_beam_2d_positions_"*string(res))
    info_callback = InfoCallback(interval=500)
    saving_callback = SolutionSavingCallback(dt=0.05, prefix="")

    callbacks = CallbackSet(info_callback, saving_callback, pp_callback)

    sol = solve(ode, RDPK3SpFSAL49(), abstol=1e-8, reltol=1e-6, dtmax=1e-3,
    save_everystep=false, callback=callbacks);
end

# Load the reference simulation data
dx_data = CSV.read("validation/Turek_dx_T.csv", DataFrame)
dy_data = CSV.read("validation/Turek_dy_T.csv", DataFrame)

# fix slight misalignment
dx_data.time = dx_data.time .+ 0.015
dx_data.displacement = dx_data.displacement .+ 0.00005
dy_data.displacement = dy_data.displacement .- 0.001

# Get the list of JSON files
json_files = glob("oscillating_beam_2d_positions_*.json", ".")

# Create subplots
fig, (ax1, ax2) = subplots(1, 2, figsize=(12, 5))

# Regular expressions for matching keys
key_pattern_x = r"pos_x_\d+_solid_\d+"
key_pattern_y = r"pos_y_\d+_solid_\d+"

function calculate_mse(reference_data, simulation_data)
    # Interpolate simulation data
    interp_func = LinearInterpolation(simulation_data["time"], simulation_data["values"])

    # Align with reference data time points
    interpolated_values = interp_func(reference_data.time)

    # Calculate MSE
    mse = mean((interpolated_values .- reference_data.displacement).^2)
    return mse
end


for json_file in json_files
    json_data = JSON.parsefile(json_file)

    local resolution = parse(Int, split(split(json_file, "_")[end], ".")[1])

    # Find matching keys and plot data for each key
    matching_keys_x = sort(collect(filter(key -> occursin(key_pattern_x, key), keys(json_data))))
    matching_keys_y = sort(collect(filter(key -> occursin(key_pattern_y, key), keys(json_data))))

    particle_spacing = cylinder_diameter/resolution

    # calculate error compared to reference
    mse_results_x = 0
    mse_results_y = 0

    for key in matching_keys_x
        data = json_data[key]
        mse_results_x = calculate_mse(dx_data, data)
    end

    for key in matching_keys_y
        data = json_data[key]
        mse_results_y = calculate_mse(dy_data, data)
    end

    # Plot x-axis displacements
    for key in matching_keys_x
        data = json_data[key]
        times = data["time"]
        values = data["values"]
        initial_position = values[1]
        displacements = [value - initial_position for value in values]
        ax1.plot(times, displacements, label="dp = $(particle_spacing) mse=$(@sprintf("%.5f", mse_results_x))")
    end

    # Plot y-axis displacements
    for key in matching_keys_y
        data = json_data[key]
        times = data["time"]
        values = data["values"]
        initial_position = values[1]
        displacements = [value - initial_position for value in values]
        ax2.plot(times, displacements, label="dp = $(particle_spacing) mse=$(@sprintf("%.5f", mse_results_y))")
    end
end

ax1.plot(dx_data.time, dx_data.displacement, label="Turek and Hron 2006", color="black", linestyle="--")
ax2.plot(dy_data.time, dy_data.displacement, label="Turek and Hron 2006", color="black", linestyle="--")

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("X Displacement")
ax1.set_title("X-Axis Displacement")
ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Y Displacement")
ax2.set_title("Y-Axis Displacement")
ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

fig.subplots_adjust(right=0.7)
fig.tight_layout()

plotshow()
