# This file computes the velocity profiles of the 2D Poiseuille flow setup described in
#
# Shuoguo Zhang, Yu Fan, Dong Wu, Chi Zhang, Xiangyu Hu.
# "Dynamical pressure boundary condition for weakly compressible smoothed particle hydrodynamics".
# Physics of Fluids 37, 027193 (2025).
# https://doi.org/10.1063/5.0254575

include("../validation_util.jl")

using TrixiParticles

# `particle_spacing_factor` is the number of particles across the channel height.
# Reference files are available for 30 and 50.
# Use `particle_spacing_factor = 50` to match the resolution used by Zhang et al. (2025).
particle_spacing_factor = 30

tspan = (0.0, 2.0)

function v_x_interpolated(system::TrixiParticles.AbstractFluidSystem{2},
                          dv_ode, du_ode, v_ode, u_ode, semi, t)
    start_point = [channel_length / 2, 0.0]
    end_point = [channel_length / 2, channel_height]

    values = interpolate_line(start_point, end_point, 100, semi, system, v_ode, u_ode;
                              cut_off_bnd=true, clip_negative_pressure=false,
                              include_wall_velocity=true)

    return values.velocity[1, :]
end
v_x_interpolated(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

pp_callback = PostprocessCallback(; dt=0.02, output_directory="out",
                                  v_x=v_x_interpolated,
                                  filename="validation_run_poiseuille_flow_2d_$particle_spacing_factor",
                                  write_csv=false, write_file_interval=0)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              saving_callback=nothing, tspan=tspan, extra_callback=pp_callback,
              particle_spacing_factor=particle_spacing_factor)
