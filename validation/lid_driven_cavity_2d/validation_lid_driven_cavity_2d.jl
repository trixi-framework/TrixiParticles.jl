using TrixiParticles

# ==========================================================================================
# ==== Resolution
particle_spacings = [0.02, 0.01, 0.005]

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 25.0)
reynolds_numbers = [100.0, 1000.0, 10_000.0]

interpolated_velocity(v, u, t, system) = nothing

function interpolated_velocity(v, u, t, system::TrixiParticles.FluidSystem)
    if t < 24.8
        return nothing
    end

    n_particles_xy = round(Int, 1.0 / system.initial_condition.particle_spacing)

    values_y = interpolate_line([0.5, 0.0], [0.5, 1.0], n_particles_xy, semi, system,
                                v, u; endpoint=true, cut_off_bnd=true)

    values_x = interpolate_line([0.0, 0.5], [1.0, 0.5], n_particles_xy, semi, system,
                                v, u; endpoint=true, cut_off_bnd=true)

    vy_y = stack(values_y.velocity)[2, :]
    vy_x = stack(values_y.velocity)[1, :]

    vx_y = stack(values_x.velocity)[2, :]
    vx_x = stack(values_x.velocity)[1, :]

    file = joinpath(output_directory, "interpolated_velocities.csv")

    if isfile(file)
        data = TrixiParticles.CSV.read(file, TrixiParticles.DataFrame)
        vy_y_ = (data.vy_y .+ vy_y)
        vy_x_ = (data.vy_x .+ vy_x)
        vx_y_ = (data.vx_y .+ vx_y)
        vx_x_ = (data.vx_x .+ vx_x)

        n_evaluations = first(data.counter) + 1

        df = TrixiParticles.DataFrame(pos=data.pos, counter=n_evaluations,
                                      vy_y=vy_y_, vy_x=vy_x_, vx_y=vx_y_, vx_x=vx_x_)

        TrixiParticles.CSV.write(output_directory * "/interpolated_velocities.csv", df)
    else
        df = TrixiParticles.DataFrame(pos=collect(LinRange(0.0, 1.0, n_particles_xy)),
                                      counter=1, vy_y=vy_y, vy_x=vy_x, vx_y=vx_y, vx_x=vx_x)

        TrixiParticles.CSV.write(output_directory * "/interpolated_velocities.csv", df)
    end

    return nothing
end

for particle_spacing in particle_spacings, reynolds_number in reynolds_numbers
    n_particles_xy = round(Int, 1.0 / particle_spacing)

    Re = Int(reynolds_number)

    global output_directory = joinpath("out_ldc",
                                       "validation_run_lid_driven_cavity_2d_nparticles_$(n_particles_xy)x$(n_particles_xy)_Re_$Re")

    saving_callback = SolutionSavingCallback(dt=0.1, output_directory=output_directory)

    pp_callback = PostprocessCallback(; dt=0.02,
                                      interpolated_velocity=interpolated_velocity,
                                      filename="interpolated_velocities",
                                      write_file_interval=0)

    # Import variables into scope
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fluid", "lid_driven_cavity_2d.jl"),
                  saving_callback=saving_callback, tspan=tspan, pp_callback=pp_callback,
                  particle_spacing=particle_spacing, reynolds_number=reynolds_number)

    file = joinpath(output_directory, "interpolated_velocities.csv")

    data = TrixiParticles.CSV.read(file, TrixiParticles.DataFrame)

    n_evaluations = first(data.counter)

    df = TrixiParticles.DataFrame(pos=data.pos,
                                  avg_vx_x=data.vx_x ./ n_evaluations,
                                  avg_vx_y=data.vx_y ./ n_evaluations,
                                  avg_vy_x=data.vy_x ./ n_evaluations,
                                  avg_vy_y=data.vy_y ./ n_evaluations,
                                  counter=n_evaluations)

    TrixiParticles.CSV.write(output_directory * "/interpolated_velocities.csv", df)
end
