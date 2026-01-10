@trixi_testset "Restart" begin
    # Run full simulation
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
                  tspan=(0.0, 0.6), sound_speed_factor=10, particle_spacing=4e-5)

    # Since this is an open boundary simulation, the number of active particles may
    # differ. The results must be interpolated to enable comparison with the restart
    # simulation. The fluid domain starts at `x = 10 * particle_spacing`.
    n_interpolation_points = 10
    start_point = [0.0 + 10 * particle_spacing, wall_distance / 2]
    end_point = [flow_length - 10 * particle_spacing, wall_distance / 2]
    result_full = interpolate_line(start_point, end_point, n_interpolation_points,
                                   semi, fluid_system, sol, cut_off_bnd=false)

    # Run half simulation
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
                  tspan=(0.0, 0.3), sound_speed_factor=10, particle_spacing=4e-5)

    iter = round(Int, 0.3 / 0.02)
    fluid_restart = joinpath("out", "fluid_1_$iter.vtu")
    open_boundary_restart = joinpath("out", "open_boundary_1_$iter.vtu")
    boundary_restart = joinpath("out", "boundary_1_$iter.vtu")

    ode_restart = semidiscretize(semi, (0.3, 0.6);
                                 restart_with=(fluid_restart, open_boundary_restart,
                                               boundary_restart))

    sol_restart = solve(ode_restart, RDPK3SpFSAL35(), abstol=1e-5, reltol=1e-3, dtmax=1e-2,
                        save_everystep=false, callback=UpdateCallback())

    result_restart = interpolate_line(start_point, end_point,
                                      n_interpolation_points, sol_restart.prob.p,
                                      sol_restart.prob.p.systems[1],
                                      sol_restart, cut_off_bnd=false)

    @test isapprox(result_full.velocity, result_restart.velocity, rtol=8e-3)
    @test isapprox(result_full.density, result_restart.density, rtol=8e-4)
    @test isapprox(result_full.pressure, result_restart.pressure, rtol=7e-2)
end
