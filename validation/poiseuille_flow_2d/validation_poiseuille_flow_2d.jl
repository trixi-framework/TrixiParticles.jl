using TrixiParticles

v_x_interpolated(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function v_x_interpolated(system::TrixiParticles.AbstractFluidSystem{2},
                          dv_ode, du_ode, v_ode, u_ode, semi, t)
    start_point = [flow_length / 2, 0.0]
    end_point = [flow_length / 2, wall_distance]

    values = interpolate_line(start_point, end_point, 100, semi, system, v_ode, u_ode;
                              cut_off_bnd=true, clip_negative_pressure=false)

    return values.velocity[1, :]
end

output_directory = joinpath(validation_dir(), "poiseuille_flow_2d")
pp_callback = PostprocessCallback(; dt=0.02,
                                  output_directory=output_directory,
                                  v_x=v_x_interpolated, filename="result_vx",
                                  write_csv=true, write_file_interval=1)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              saving_callback=nothing, tspan=(0.0, 2.0), extra_callback=pp_callback)
