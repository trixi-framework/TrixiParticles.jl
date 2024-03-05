
include("../validation_util.jl")

using TrixiParticles
using JSON

gravity = 9.81

H = 0.6
W = 2 * H
tspan = (0.0, 8.0 / sqrt(gravity / H))

# particle_spacing in this case is set comparatively to `H` the initial height of the fluid.
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

sensor_size = 0.009
P1_y_top = 160 / 600 * H
P1_y_bottom = P1_y_top - sensor_size
P2_y_top = (160 + 424) / 600 * H
P2_y_bottom = P2_y_top - sensor_size
P3_y_top = (160 + 424 + 416) / 600 * H
P3_y_bottom = P3_y_top - sensor_size

sensor_names = ["P1", "P2", "P3"]

tank_size_x = floor(5.366 * H / particle_spacing) * particle_spacing -
              0.5 * particle_spacing
pressure_sensor_top = [
    [tank_size_x, P1_y_top],
    [tank_size_x, P2_y_top],
    [tank_size_x, P3_y_top],
]
pressure_sensor_bottom = [
    [tank_size_x, P1_y_bottom],
    [tank_size_x, P2_y_bottom],
    [tank_size_x, P3_y_bottom],
]

function max_x_coord(v, u, t, system)
    maximum(TrixiParticles.extract_svector(TrixiParticles.current_coordinates(u,
                                                                              system),
                                           Val(ndims(system)), p)[1]
            for p in TrixiParticles.eachparticle(system))
end

function interpolated_pressure(coord_top, coord_bottom, v, u, t, system)
    sol = (; u=[(; x=(v, u))])
    n_interpolation_points = 10
    interpolated_values = interpolate_line(coord_top, coord_bottom,
                                           n_interpolation_points, semi, system, sol,
                                           smoothing_length=2.0 *
                                                            system.smoothing_length)
    return sum(interpolated_values.pressure) / n_interpolation_points
end

pressure_sensors = [("pressure_$(name)",
                     (v, u, t, sys) -> interpolated_pressure(coord_top, coord_bottom, v,
                                                             u, t, sys))
                    for (coord_top, coord_bottom, name) in zip(pressure_sensor_top,
                                                               pressure_sensor_bottom,
                                                               sensor_names)]
named_sensors = (; (Symbol("$(name)") => func for (name, func) in pressure_sensors)...)
formatted_string = lpad(string(Int(particle_spacing *
                                   10^length(split(string(particle_spacing), ".")[2]))),
                        length(split(string(particle_spacing), ".")[2]) + 1, '0')

# EDAC simulation
############################################################################################
method = "edac"
postprocessing_cb = PostprocessCallback(; dt=0.02, output_directory="out",
                                        filename="validation_result_dam_break_" *
                                                 method * "_" * formatted_string,
                                        write_csv=false, max_x_coord, named_sensors...)

tank_edac = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density,
                            n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                            acceleration=(0.0, -gravity), state_equation=nothing)

alpha = 0.02
viscosity_edac = ViscosityAdami(nu=alpha * smoothing_length * sound_speed / 8)
fluid_system_edac = EntropicallyDampedSPHSystem(tank_edac.fluid, smoothing_kernel,
                                                smoothing_length,
                                                sound_speed, viscosity=viscosity_edac,
                                                acceleration=(0.0, -gravity))

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              smoothing_length=3.5 * particle_spacing,
              boundary_layers=4, state_equation=nothing,
              solution_prefix="validation_" * method * "_" * formatted_string,
              cfl=0.9, extra_callback=postprocessing_cb, tspan=tspan,
              fluid_system=fluid_system_edac, tank=tank_edac)

reference_file_edac_name = "validation/dam_break_2d/validation_reference_edac_0015.json"
run_file_edac_name = "out/validation_result_dam_break_edac_0015.json"

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
                                        write_csv=false, max_x_coord, named_sensors...)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing,
              smoothing_length=3.5 * particle_spacing,
              boundary_layers=4,
              solution_prefix="validation_" * method * "_" * formatted_string,
              extra_callback=postprocessing_cb, tspan=tspan)

reference_file_wcsph_name = "validation/dam_break_2d/validation_reference_wcsph_00075.json"
run_file_wcsph_name = "out/validation_result_dam_break_wcsph_00075.json"

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
