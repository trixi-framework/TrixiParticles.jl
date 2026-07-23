@trixi_testset "Restart" begin
    @trixi_testset "Poiseuille Flow Half-Simulation Restart" begin
        # Run full simulation
        trixi_include(@__MODULE__,
                      joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
                      tspan=(0.0, 0.6), sound_speed_factor=10, particle_spacing=4e-5,
                      info_callback=nothing, sol=nothing)

        sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                    dt=1.0, save_everystep=false,
                    callback=CallbackSet(callbacks, StepsizeCallback(cfl=1.5)))

        # Since this is an open boundary simulation, the number of active particles may
        # differ. The results must be interpolated to enable comparison with the restart
        # simulation. The fluid domain starts at `x = 10 * particle_spacing`.
        n_interpolation_points = 10
        start_point = [0.0 + 10 * particle_spacing, channel_height / 2]
        end_point = [channel_length - 10 * particle_spacing, channel_height / 2]
        result_full = interpolate_line(start_point, end_point, n_interpolation_points,
                                       semi, fluid_system, sol, cut_off_bnd=false)

        # Run half simulation
        trixi_include(@__MODULE__,
                      joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
                      tspan=(0.0, 0.3), sound_speed_factor=10, particle_spacing=4e-5,
                      info_callback=nothing, sol=nothing)

        sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                    dt=1.0, save_everystep=false,
                    callback=CallbackSet(callbacks, StepsizeCallback(cfl=1.5)))

        iter = saving_callback.affect!.affect!.latest_saved_iter
        fluid_restart = joinpath("out", "fluid_1_$iter.vtu")
        open_boundary_restart = joinpath("out", "open_boundary_1_$iter.vtu")
        boundary_restart = joinpath("out", "boundary_1_$iter.vtu")

        ode_restart = semidiscretize(semi, (0.3, 0.6);
                                     restart_with=(fluid_restart, open_boundary_restart,
                                                   boundary_restart))

        sol_restart = solve(ode_restart, CarpenterKennedy2N54(williamson_condition=false),
                            dt=1.0, save_everystep=false,
                            callback=CallbackSet(UpdateCallback(),
                                                 StepsizeCallback(cfl=1.5)))

        result_restart = interpolate_line(start_point, end_point,
                                          n_interpolation_points, sol_restart.prob.p.semi,
                                          sol_restart.prob.p.semi.systems[1],
                                          sol_restart, cut_off_bnd=false)

        @test isapprox(result_full.velocity, result_restart.velocity, rtol=5e-5)
        @test isapprox(result_full.density, result_restart.density, rtol=5e-6)
        @test isapprox(result_full.pressure, result_restart.pressure, rtol=5e-4)
    end

    @trixi_testset "Poiseuille Flow Restore Previous State" begin
        R1 = 1.7714
        R2 = 106.66
        C = 1.1808e-2
        pressure_model = RCRWindkesselModel(; peripheral_resistance=R2,
                                            compliance=C,
                                            characteristic_resistance=R1)
        # Run half simulation
        trixi_include(@__MODULE__,
                      joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
                      outlet_reference_pressure=pressure_model, info_callback=nothing,
                      tspan=(0.0, 0.3), sound_speed_factor=10, particle_spacing=4e-5)

        iter = saving_callback.affect!.affect!.latest_saved_iter
        fluid_restart = joinpath("out", "fluid_1_$iter.vtu")
        open_boundary_restart = joinpath("out", "open_boundary_1_$iter.vtu")
        boundary_restart = joinpath("out", "boundary_1_$iter.vtu")

        ode_restart = semidiscretize(semi, (0.3, 0.6);
                                     restart_with=(fluid_restart, open_boundary_restart,
                                                   boundary_restart))
        restart_pressure = ode_restart.p.semi.systems[2].cache.pressure_reference_values[2].pressure[]
        restart_flow_rate = ode_restart.p.semi.systems[2].cache.pressure_reference_values[2].flow_rate[]

        @test isapprox(restart_pressure,
                       open_boundary.cache.pressure_reference_values[2].pressure[])
        @test isapprox(restart_flow_rate,
                       open_boundary.cache.pressure_reference_values[2].flow_rate[])
    end

    @trixi_testset "Oscillating Beam Half-Simulation Restart" begin
        # Run full simulation
        trixi_include(@__MODULE__,
                      joinpath(examples_dir(), "structure", "oscillating_beam_2d.jl"),
                      tspan=(0.0, 1.0), n_particles_y=5, info_callback=nothing)

        # Store the final solution for comparison
        full_sol = sol
        positions_full = full_sol.u[end].x[2]

        # Run half simulation
        trixi_include(@__MODULE__,
                      joinpath(examples_dir(), "structure", "oscillating_beam_2d.jl"),
                      tspan=(0.0, 0.5), n_particles_y=5)

        # Get latest iteration and create restart
        iter = saving_callback.affect!.affect!.latest_saved_iter
        restart_file = joinpath("out", "structure_1_$iter.vtu")

        ode_restart = semidiscretize(semi, (0.5, 1.0);
                                     restart_with=restart_file)

        sol_restart = solve(ode_restart, RDPK3SpFSAL49(), save_everystep=false, dt=1e-6)

        # Compare the final particle velocities
        @test isapprox(sol_restart.u[end].x[2], positions_full, rtol=8e-5)
    end
end
