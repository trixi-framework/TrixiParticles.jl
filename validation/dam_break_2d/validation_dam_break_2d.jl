
using TrixiParticles

tspan = (0.0, 8.0 / sqrt(9.81))

#resolutions = [0.025, 0.0125, 0.005, 0.003125]
resolutions = [0.005]

H = 1.0
sensor_size = 0.009 / 0.6
P1_y_top = 160 / 600 * H
P1_y_bottom = P1_y_top - sensor_size
P2_y_top = (160 + 424) / 600 * H
P2_y_bottom = P2_y_top - sensor_size
P3_y_top = (160 + 424 + 416) / 600 * H
P3_y_bottom = P3_y_top - sensor_size

sensor_names = ["P1", "P2", "P3"]

for particle_spacing in resolutions
    tank_size_x = floor(5.366 / particle_spacing) * particle_spacing - 0.5 * particle_spacing
    pressure_sensor_top = [[tank_size_x, P1_y_top], [tank_size_x, P2_y_top], [tank_size_x, P3_y_top]]
    pressure_sensor_bottom= [[tank_size_x, P1_y_bottom], [tank_size_x, P2_y_bottom], [tank_size_x, P3_y_bottom]]

    function max_x_coord(v, u, t, system)
        system_coords = TrixiParticles.current_coordinates(u, system)
        max_x = 0.0
        for particle in TrixiParticles.eachparticle(system)
            p_x = TrixiParticles.extract_svector(system_coords, Val(ndims(system)),
                                      particle)[1]
            if max_x < p_x
                max_x = p_x
            end
        end
        return max_x
    end

    function interpolated_pressure(coord_top, coord_bottom, v, u, t, system)
        sol = (; u=[(; x=(v, u))])
        n_interpolation_points = 10
        interpolated_values = interpolate_line(coord_top, coord_bottom, n_interpolation_points, semi, system, sol, smoothing_length=2.0 * system.smoothing_length)
        return sum(interpolated_values.pressure)/n_interpolation_points
    end

    pressure_sensors = [
        ("pressure_$(name)", (v, u, t, sys) -> interpolated_pressure(coord_top, coord_bottom, v, u, t, sys))
        for (coord_top, coord_bottom, name) in zip(pressure_sensor_top, pressure_sensor_bottom, sensor_names)
    ]
    named_sensors = (; (Symbol("$(name)") => func for (name, func) in pressure_sensors)...)

    formatted_string = lpad(string(Int(particle_spacing * 10^length(split(string(particle_spacing), ".")[2]))), length(split(string(particle_spacing), ".")[2]) + 1, '0')
    postprocessing_cb = PostprocessCallback(; dt=0.025, output_directory="validation/dam_break_2d",
    filename="dam_break_"*formatted_string, write_csv=false, max_x_coord, named_sensors...)

    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                fluid_particle_spacing=particle_spacing,
                smoothing_length=3.5 * particle_spacing,
                boundary_layers=4, solution_prefix="validation_"*formatted_string,
                cfl=0.9, pp_callback=postprocessing_cb, tspan=tspan)
end
