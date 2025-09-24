using TrixiParticles

v_x_interpolated(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function v_x_interpolated(system::TrixiParticles.AbstractFluidSystem{3},
                          dv_ode, du_ode, v_ode, u_ode, semi, t)
    start_point = [flow_length / 2, -pipe_radius, 0.0]
    end_point = [flow_length / 2, pipe_radius, 0.0]

    values = interpolate_line(start_point, end_point, 100, semi, system, v_ode, u_ode;
                              cut_off_bnd=true, clip_negative_pressure=false)

    return values.velocity[1, :]
end

particle_spacing_factor = 10
output_directory = joinpath(validation_dir(), "pulsatile_channel_flow_3d")
pp_callback = PostprocessCallback(; dt=0.01, output_directory=output_directory,
                                  v_x=v_x_interpolated,
                                  filename="result_vx" * "_dp_$particle_spacing_factor",
                                  write_csv=true, write_file_interval=1)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "pulsatile_channel_flow_3d.jl"),
              saving_callback=nothing, tspan=(0.0, 12.6), extra_callback=pp_callback,
              particle_spacing_factor=particle_spacing_factor)
