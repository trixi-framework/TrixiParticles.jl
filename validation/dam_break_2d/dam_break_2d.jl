using TrixiParticles
using Printf
using PythonPlot
using JSON
using Glob

tspan = (0.0, 8.0 / sqrt(9.81))

#resolutions = [0.025, 0.0125, 0.005, 0.003125]
resolutions = [0.025]

H = 1.0
sensor_size = 0.009/0.6
P1_y_top = 160/600 * H
P1_y_bottom = P1_y_top - sensor_size
P2_y_top = (160 + 424)/600 * H
P2_y_bottom = P2_y_top - sensor_size
P3_y_top = (160 + 424 + 416)/600 * H
P3_y_bottom = P3_y_top - sensor_size

sensor_names = ["P1", "P2", "P3"]

for particle_spacing in resolutions
    tank_size_x = floor(5.366 / particle_spacing) * particle_spacing - 0.5 * particle_spacing
    pressure_sensor_top = [[tank_size_x, P1_y_top], [tank_size_x, P2_y_top], [tank_size_x, P3_y_top]]
    pressure_sensor_bottom= [[tank_size_x, P1_y_bottom], [tank_size_x, P2_y_bottom], [tank_size_x, P3_y_bottom]]

    formatted_string = lpad(string(Int(particle_spacing * 10^length(split(string(particle_spacing), ".")[2]))), length(split(string(particle_spacing), ".")[2]) + 1, '0')

    function max_x_coord(pp, t, system, u, v, system_name)
        system_coords = TrixiParticles.current_coordinates(u, system)
        max_x = 0.0
        for particle in TrixiParticles.eachparticle(system)
            particle_coords = TrixiParticles.extract_svector(system_coords, Val(ndims(system)),
                                      particle)
            if max_x < particle_coords[1]
                max_x = particle_coords[1]
            end
        end
        TrixiParticles.add_entry!(pp, "max_x", t, max_x, system_name)
    end

    # function interpolated_pressure(coord, sensor_name, pp, t, system, u, v, system_name)
    #     sol = (; u=[(; x=(v, u))])
    #     interpolated_values = interpolate_point(coord, semi, system, sol)
    #     TrixiParticles.add_entry!(pp, "pressure_"*sensor_name, t, interpolated_values.pressure, system_name)
    # end

    # pressure_sensors = [
    #     (pp, t, sys, u, v, sys_name) -> interpolated_pressure(coord, name, pp, t, sys, u, v, sys_name)
    #     for (coord, name) in zip(pressure_sensor_coordinates, sensor_names)
    # ]

    function interpolated_pressure(coord_top, coord_bottom, sensor_name, pp, t, system, u, v, system_name)
        sol = (; u=[(; x=(v, u))])
        interpolated_values = interpolate_line(coord_top, coord_bottom, 10, semi, system, sol)

        avg_p = 0.0
        for pressure in interpolated_values.pressure
            avg_p += pressure
        end
        avg_p=avg_p/10

        TrixiParticles.add_entry!(pp, "pressure_"*sensor_name, t, avg_p, system_name)
    end

    pressure_sensors = [
        (pp, t, sys, u, v, sys_name) -> interpolated_pressure(coord_top, coord_bottom, name, pp, t, sys, u, v, sys_name)
        for (coord_top, coord_bottom, name) in zip(pressure_sensor_top, pressure_sensor_bottom, sensor_names)
    ]


    #pressure_sensors = (pp, t, sys, u, v, sys_name) -> interpolated_pressure(pressure_sensor_coordinates, sensor_names, pp, t, sys, u, v, sys_name)

    pp_callback = PostprocessCallback([max_x_coord; pressure_sensors...], dt=0.01, filename="dam_break_"*formatted_string)

    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                fluid_particle_spacing=particle_spacing,
                smoothing_length=3.5 * particle_spacing,
                boundary_layers=4, prefix="validation_"*formatted_string,
                abstol=1e-8, reltol=1e-6, pp_callback=pp_callback, tspan=tspan)
end

normalization_factor_time = sqrt(9.81/0.6)
normalization_factor_pressure = 1000 * 9.81

json_files = glob("dam_break_*.json", ".")
for json_file in json_files
    json_data = JSON.parsefile(json_file)

    for (key, value) in json_data
        if startswith(key, "pressure_") && isa(value, Dict)
            time = value["time"]
            values = value["values"]
            plot(time.*normalization_factor_time, values./normalization_factor_pressure, label=key)
        end
    end

    ylim([-0.1, 1.0])
    xlabel("Time")
    ylabel("Pressure")
    title("Pressure vs Time")
    legend()
    plotshow()
end
