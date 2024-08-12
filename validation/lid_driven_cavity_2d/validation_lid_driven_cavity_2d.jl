using TrixiParticles

# ==========================================================================================
# ==== Resolution
particle_spacings = [0.02, 0.01, 0.005]

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 0.001)
reynolds_numbers = [100.0, 1000.0, 10_000.0]

for particle_spacing in particle_spacings, reynolds_number in reynolds_numbers
    n_particles_xy = round(Int, 1.0 / particle_spacing)

    Re = Int(reynolds_number)

    output_directory = joinpath("out_ldc",
                                "validation_run_lid_driven_cavity_2d_nparticles_$(n_particles_xy)x$(n_particles_xy)_Re_$Re")

    saving_callback = SolutionSavingCallback(dt=0.02, output_directory=output_directory)

    # Import variables into scope
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fluid", "lid_driven_cavity_2d.jl"),
                  saving_callback=saving_callback, tspan=tspan,
                  particle_spacing=particle_spacing, reynolds_number=reynolds_number)

    values_y = interpolate_line([0.5, 0.0], [0.5, 1.0], n_particles_xy, semi, fluid_system,
                                sol; endpoint=true, cut_off_bnd=true)

    values_x = interpolate_line([0.0, 0.5], [1.0, 0.5], n_particles_xy, semi, fluid_system,
                                sol; endpoint=true, cut_off_bnd=true)

    vy_y = stack(values_y.velocity)[2, :]
    vy_x = stack(values_y.velocity)[1, :]

    vx_y = stack(values_x.velocity)[2, :]
    vx_x = stack(values_x.velocity)[1, :]

    df = TrixiParticles.DataFrame(pos=collect(LinRange(0.0, 1.0, n_particles_xy)),
                                  vy_y=vy_y, vy_x=vy_x, vx_y=vx_y, vx_x=vx_x)

    TrixiParticles.CSV.write(output_directory * "/interpolated_velocities.csv", df)
end
