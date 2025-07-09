
include("../validation_util.jl")

using TrixiParticles
using JSON

gravity = 9.81

H = 0.6
W = 2 * H
tspan = (0.0, 8.0 / sqrt(gravity / H))

# `particle_spacing` in this case is set relative to `H`, the initial height of the fluid.
# Use H / 80, H / 320 for validation.
# Note: H / 320 takes a few hours!
particle_spacing = H / 40
# For reference:
# H/320 is 0.001875m
# H/80 is 0.0075m
# H/40 is 0.015m
smoothing_length = 1.32 * particle_spacing
smoothing_kernel = GaussianKernel{2}()

fluid_density = 1000.0
sound_speed = 40 * sqrt(gravity * H)

boundary_layers = 4
spacing_ratio = 1
boundary_particle_spacing = particle_spacing / spacing_ratio

initial_fluid_size = (W, H)
tank_size = (floor(5.366 * H / boundary_particle_spacing) * boundary_particle_spacing, 4.0)

function max_x_coord(system, data, t)
    return maximum(j -> data.coordinates[1, j], axes(data.coordinates, 2))
end

# The top of the sensors is at the position where the middle of the sensors is in the experiment.
#
# > Note, the probe position in the numerical setup does not exactly match the position in the experiment, but
# > Greco [36] showed that this shift gives better agreement and reported several uncertainties in the
# > measurements motivating this adjustment.
# S. Adami, X. Y. Hu, N. A. Adams.
# "A generalized wall boundary condition for smoothed particle hydrodynamics".
# In: Journal of Computational Physics 231, 21 (2012), pages 7057--7075.
# https://doi.org/10.1016/J.JCP.2012.05.005
sensor_size = 0.09
P1_y_top = 160 / 600 * H
P1_y_bottom = P1_y_top - sensor_size
P2_y_top = (160 + 424) / 600 * H
P2_y_bottom = P2_y_top - sensor_size
P3_y_top = (160 + 424 + 416) / 600 * H
P3_y_bottom = P3_y_top - sensor_size

sensor_names = ["P1", "P2", "P3"]

tank_right_wall_x = floor(5.366 * H / particle_spacing) * particle_spacing -
                    0.5 * particle_spacing

interpolated_pressure_P1 = (system, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P1_y_top],
                                           [tank_right_wall_x, P1_y_bottom],
                                           v_ode, u_ode, t, system, semi)
interpolated_pressure_P2 = (system, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P2_y_top],
                                           [tank_right_wall_x, P2_y_bottom],
                                           v_ode, u_ode, t, system, semi)
interpolated_pressure_P3 = (system, v_ode, u_ode, semi,
               t) -> interpolated_pressure([tank_right_wall_x, P3_y_top],
                                           [tank_right_wall_x, P3_y_bottom],
                                           v_ode, u_ode, t, system, semi)

function interpolated_pressure(coord_top, coord_bottom, v_ode, u_ode, t, system, semi) end

function interpolated_pressure(coord_top, coord_bottom, v_ode, u_ode, t, system::TrixiParticles.FluidSystem, semi)
    # use at least 5 interpolation points for low resolution simulations
    # otherwise use at least the number of particles present
    n_interpolation_points = min(5, Int(ceil(sensor_size / particle_spacing)))
    interpolated_values = interpolate_line(coord_top, coord_bottom,
                                           n_interpolation_points, semi, system, v_ode,
                                           u_ode,
                                           smoothing_length=2.0 *
                                                            TrixiParticles.initial_smoothing_length(system),
                                           clip_negative_pressure=true)
    return sum(map(x -> isnan(x) ? 0.0 : x, interpolated_values.pressure)) /
           n_interpolation_points
end

particle_pressure_P1 = (system, v_ode, u_ode, semi,
               t) -> pressure_probe([tank_right_wall_x, P1_y_top],
                                           [tank_right_wall_x, P1_y_bottom],
                                           v_ode, u_ode, t, system, semi)
particle_pressure_P2 = (system, v_ode, u_ode, semi,
               t) -> pressure_probe([tank_right_wall_x, P2_y_top],
                                           [tank_right_wall_x, P2_y_bottom],
                                           v_ode, u_ode, t, system, semi)
particle_pressure_P3 = (system, v_ode, u_ode, semi,
               t) -> pressure_probe([tank_right_wall_x, P3_y_top],
                                           [tank_right_wall_x, P3_y_bottom],
                                           v_ode, u_ode, t, system, semi)

function pressure_probe(coord_top, coord_bottom, v_ode, u_ode, t, system, semi)
    # The sensor is at the right wall, so its x-coordinate is the same for top and bottom.
    x_sensor = coord_top[1]

    # Use the initial particle spacing as a reference for the thickness of the averaging region.
    # A thickness of one or two particle spacings is usually a good choice.
    region_thickness = 2.0 * particle_spacing

    # Define the rectangular region for averaging
    x_min = x_sensor - region_thickness
    x_max = x_sensor + region_thickness
    y_min = coord_bottom[2]
    y_max = coord_top[2]

    sum_of_pressures = 0.0
    num_particles_in_region = 0

    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)

    # Iterate over each particle in the specified fluid system
    for particle_idx in TrixiParticles.eachparticle(system)
        pc = TrixiParticles.current_coords(u, system, particle_idx)

        # Get coordinates for the current particle from the 1D vector
        px = pc[1] # x-coordinate
        py = pc[2] # y-coordinate

        # Check if the particle is inside the sensor's rectangular region
        if (x_min <= px <= x_max) && (y_min <= py <= y_max)
            # Add its pressure to the sum and increment the count
            sum_of_pressures += TrixiParticles.current_pressure(v, system, particle_idx)
            num_particles_in_region += 1
        end
    end

    # If no particles are in the region (e.g., before the water hits the wall),
    # the pressure is zero.
    if num_particles_in_region == 0
        return 0.0
    end

    # Return the calculated average pressure
    return sum_of_pressures / num_particles_in_region
end

formatted_string = replace(string(particle_spacing), "." => "")

# EDAC simulation
############################################################################################
method = "edac"
postprocessing_cb = PostprocessCallback(; dt=0.02, output_directory="out",
                                        filename="validation_result_dam_break_" *
                                                 method * "_" * formatted_string,
                                        write_csv=false, exclude_boundary=false, max_x_coord,
                                        interpolated_pressure_P1, interpolated_pressure_P2, interpolated_pressure_P3,
                                        particle_pressure_P1, particle_pressure_P2, particle_pressure_P3)

tank_edac = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density,
                            n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                            acceleration=(0.0, -gravity), state_equation=nothing)

alpha = 0.02
viscosity_edac = ViscosityAdami(nu=alpha * smoothing_length * sound_speed / 8)
fluid_system_edac = EntropicallyDampedSPHSystem(tank_edac.fluid, smoothing_kernel,
                                                smoothing_length,
                                                sound_speed, viscosity=viscosity_edac,
                                                density_calculator=ContinuityDensity(),
                                                pressure_acceleration=nothing,
                                                acceleration=(0.0, -gravity))

# Disable loop flipping to produce consistent results over different thread numbers
boundary_density_calculator = AdamiPressureExtrapolation(allow_loop_flipping=false)
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              smoothing_length=smoothing_length, smoothing_kernel=smoothing_kernel,
              boundary_density_calculator=boundary_density_calculator,
              boundary_layers=4, state_equation=nothing,
              solution_prefix="validation_" * method * "_" * formatted_string,
              extra_callback=postprocessing_cb, tspan=tspan,
              fluid_system=fluid_system_edac, tank=tank_edac,
              update_strategy=SerialUpdate()) # To get the same results with different thread numbers

reference_file_edac_name = joinpath(validation_dir(), "dam_break_2d",
                                    "validation_reference_edac_$formatted_string.json")
run_file_edac_name = joinpath("out",
                              "validation_result_dam_break_edac_$formatted_string.json")

reference_data = JSON.parsefile(reference_file_edac_name)
run_data = JSON.parsefile(run_file_edac_name)

error_edac_P1 = interpolated_mse(reference_data["interpolated_pressure_P1_fluid_1"]["time"],
                                 reference_data["interpolated_pressure_P1_fluid_1"]["values"],
                                 run_data["interpolated_pressure_P1_fluid_1"]["time"],
                                 run_data["interpolated_pressure_P1_fluid_1"]["values"])

error_edac_P2 = interpolated_mse(reference_data["interpolated_pressure_P2_fluid_1"]["time"],
                                 reference_data["interpolated_pressure_P2_fluid_1"]["values"],
                                 run_data["interpolated_pressure_P2_fluid_1"]["time"],
                                 run_data["interpolated_pressure_P2_fluid_1"]["values"])

# WCSPH simulation
############################################################################################
method = "wcsph"
postprocessing_cb = PostprocessCallback(; dt=0.02, output_directory="out",
                                        filename="validation_result_dam_break_" *
                                                 method * "_" * formatted_string,
                                        write_csv=false, exclude_boundary=false, max_x_coord,
                                        interpolated_pressure_P1, interpolated_pressure_P2, interpolated_pressure_P3,
                                        particle_pressure_P1, particle_pressure_P2, particle_pressure_P3)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              smoothing_length=smoothing_length, smoothing_kernel=smoothing_kernel,
              boundary_density_calculator=boundary_density_calculator,
              boundary_layers=4,
              solution_prefix="validation_" * method * "_" * formatted_string,
              extra_callback=postprocessing_cb, tspan=tspan,
              update_strategy=SerialUpdate()) # To get the same results with different thread numbers

reference_file_wcsph_name = joinpath(validation_dir(), "dam_break_2d",
                                     "validation_reference_wcsph_$formatted_string.json")
run_file_wcsph_name = joinpath("out",
                               "validation_result_dam_break_wcsph_$formatted_string.json")

reference_data = JSON.parsefile(reference_file_wcsph_name)
run_data = JSON.parsefile(run_file_wcsph_name)

error_wcsph_P1 = interpolated_mse(reference_data["interpolated_pressure_P1_fluid_1"]["time"],
                                  reference_data["interpolated_pressure_P1_fluid_1"]["values"],
                                  run_data["interpolated_pressure_P1_fluid_1"]["time"],
                                  run_data["interpolated_pressure_P1_fluid_1"]["values"])

error_wcsph_P2 = interpolated_mse(reference_data["interpolated_pressure_P2_fluid_1"]["time"],
                                  reference_data["interpolated_pressure_P2_fluid_1"]["values"],
                                  run_data["interpolated_pressure_P2_fluid_1"]["time"],
                                  run_data["interpolated_pressure_P2_fluid_1"]["values"])
