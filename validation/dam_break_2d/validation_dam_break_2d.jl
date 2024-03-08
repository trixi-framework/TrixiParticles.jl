
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
smoothing_length = 3.5 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density = 1000.0
sound_speed = 20 * sqrt(gravity * H)

boundary_layers = 4
spacing_ratio = 1
boundary_particle_spacing = particle_spacing / spacing_ratio

initial_fluid_size = (W, H)
tank_size = (floor(5.366 * H / boundary_particle_spacing) * boundary_particle_spacing, 4.0)

# The top of the sensors is at the position where the middle of the sensors is in the experiment.
#
# > Note, the probe position in the numerical setup does not exactly match the position in the experiment, but
# > Greco [36] showed that this shift gives better agreement and reported several uncertainties in the
# > measurements motivating this adjustment.
# S. Adami, X. Y. Hu, N. A. Adams.
# "A generalized wall boundary condition for smoothed particle hydrodynamics".
# In: Journal of Computational Physics 231, 21 (2012), pages 7057--7075.
# https://doi.org/10.1016/J.JCP.2012.05.005
sensor_size = 0.009
P1_y_top = 160 / 600 * H
P1_y_bottom = P1_y_top - sensor_size
P2_y_top = (160 + 424) / 600 * H
P2_y_bottom = P2_y_top - sensor_size
P3_y_top = (160 + 424 + 416) / 600 * H
P3_y_bottom = P3_y_top - sensor_size

sensor_names = ["P1", "P2", "P3"]

tank_right_wall_x = floor(5.366 * H / particle_spacing) * particle_spacing -
                    0.5 * particle_spacing

pressure_P1 = (v, u, t, sys) -> interpolated_pressure([tank_right_wall_x, P1_y_top],
                                                      [tank_right_wall_x, P1_y_bottom],
                                                      v, u, t, sys)
pressure_P2 = (v, u, t, sys) -> interpolated_pressure([tank_right_wall_x, P2_y_top],
                                                      [tank_right_wall_x, P2_y_bottom],
                                                      v, u, t, sys)
pressure_P3 = (v, u, t, sys) -> interpolated_pressure([tank_right_wall_x, P3_y_top],
                                                      [tank_right_wall_x, P3_y_bottom],
                                                      v, u, t, sys)

function max_x_coord(v, u, t, system)
    return maximum(particle -> TrixiParticles.current_coords(u, system, particle)[1],
                   TrixiParticles.eachparticle(system))
end

function interpolated_pressure(coord_top, coord_bottom, v, u, t, system)
    sol = (; u=[(; x=(v, u))])
    n_interpolation_points = 10
    interpolated_values = interpolate_line(coord_top, coord_bottom,
                                           n_interpolation_points, semi, system, sol,
                                           smoothing_length=2.0 * system.smoothing_length,
                                           clip_negative_pressure=true)
    return sum(map(x -> isnan(x) ? 0.0 : x, interpolated_values.pressure)) /
           n_interpolation_points
end

formatted_string = replace(string(particle_spacing), "." => "")

# EDAC simulation
############################################################################################
method = "edac"
postprocessing_cb = PostprocessCallback(; dt=0.02, output_directory="out",
                                        filename="validation_result_dam_break_" *
                                                 method * "_" * formatted_string,
                                        write_csv=false, max_x_coord, pressure_P1,
                                        pressure_P2, pressure_P3)

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

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              smoothing_length=smoothing_length, smoothing_kernel=smoothing_kernel,
              boundary_layers=4, state_equation=nothing,
              solution_prefix="validation_" * method * "_" * formatted_string,
              extra_callback=postprocessing_cb, tspan=tspan,
              fluid_system=fluid_system_edac, tank=tank_edac)

reference_file_edac_name = joinpath(validation_dir(), "dam_break_2d",
                                    "validation_reference_edac_$formatted_string.json")
run_file_edac_name = joinpath("out",
                              "validation_result_dam_break_edac_$formatted_string.json")

reference_data = JSON.parsefile(reference_file_edac_name)
run_data = JSON.parsefile(run_file_edac_name)

error_edac_P1 = interpolated_mse(reference_data["pressure_P1_fluid_1"]["time"],
                                 reference_data["pressure_P1_fluid_1"]["values"],
                                 run_data["pressure_P1_fluid_1"]["time"],
                                 run_data["pressure_P1_fluid_1"]["values"])

error_edac_P2 = interpolated_mse(reference_data["pressure_P2_fluid_1"]["time"],
                                 reference_data["pressure_P2_fluid_1"]["values"],
                                 run_data["pressure_P2_fluid_1"]["time"],
                                 run_data["pressure_P2_fluid_1"]["values"])

# WCSPH simulation
############################################################################################
method = "wcsph"
postprocessing_cb = PostprocessCallback(; dt=0.02, output_directory="out",
                                        filename="validation_result_dam_break_" *
                                                 method * "_" * formatted_string,
                                        write_csv=false, max_x_coord, pressure_P1,
                                        pressure_P2, pressure_P3)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              smoothing_length=smoothing_length, smoothing_kernel=smoothing_kernel,
              boundary_layers=4,
              solution_prefix="validation_" * method * "_" * formatted_string,
              extra_callback=postprocessing_cb, tspan=tspan)

reference_file_wcsph_name = joinpath(validation_dir(), "dam_break_2d",
                                     "validation_reference_wcsph_$formatted_string.json")
run_file_wcsph_name = joinpath("out",
                               "validation_result_dam_break_wcsph_$formatted_string.json")

reference_data = JSON.parsefile(reference_file_wcsph_name)
run_data = JSON.parsefile(run_file_wcsph_name)

error_wcsph_P1 = interpolated_mse(reference_data["pressure_P1_fluid_1"]["time"],
                                  reference_data["pressure_P1_fluid_1"]["values"],
                                  run_data["pressure_P1_fluid_1"]["time"],
                                  run_data["pressure_P1_fluid_1"]["values"])

error_wcsph_P2 = interpolated_mse(reference_data["pressure_P2_fluid_1"]["time"],
                                  reference_data["pressure_P2_fluid_1"]["values"],
                                  run_data["pressure_P2_fluid_1"]["time"],
                                  run_data["pressure_P2_fluid_1"]["values"])
