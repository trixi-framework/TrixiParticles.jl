# Smoke tests, i.e., tests to verify that examples are running without crashing,
# but without checking the correctness of the solution.
@testset verbose=true "Examples" begin
    include("examples_fluid.jl")

    @testset verbose=true "Structure" begin
        @trixi_testset "structure/oscillating_beam_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "structure",
                                                      "oscillating_beam_2d.jl"),
                                             tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "structure/oscillating_beam_2d.jl with penalty force and viscosity" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "structure",
                                                      "oscillating_beam_2d.jl"),
                                             tspan=(0.0, 0.1),
                                             penalty_force=PenaltyForceGanzenmueller(alpha=0.1),
                                             viscosity=ArtificialViscosityMonaghan(alpha=0.01))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "structure/oscillating_beam_2d.jl with rotating clamp" begin
            # Simple rotation
            movement_function(x,
                              t) = SVector(cos(2pi * t) * x[1] - sin(2pi * t) * x[2],
                                           sin(2pi * t) * x[1] + cos(2pi * t) * x[2])
            is_moving(t) = t < 0.1
            prescribed_motion = PrescribedMotion(movement_function, is_moving)

            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "structure",
                                                      "oscillating_beam_2d.jl"),
                                             tspan=(0.0, 0.2),
                                             penalty_force=PenaltyForceGanzenmueller(alpha=0.1),
                                             clamped_particles_motion=prescribed_motion)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end
    end

    @testset verbose=true "FSI" begin
        @trixi_testset "fsi/falling_water_column_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "falling_water_column_2d.jl"),
                                             tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fsi/dam_break_plate_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "dam_break_plate_2d.jl"),
                                             # Use rounded dimensions to avoid warnings
                                             initial_fluid_size=(0.15, 0.29),
                                             tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fsi/dam_break_plate_2d.jl split integration" begin
            # Test that this example does not work with only 500 iterations
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "dam_break_plate_2d.jl"),
                                             # Use rounded dimensions to avoid warnings
                                             initial_fluid_size=(0.15, 0.29),
                                             # Move plate closer to be able to use a shorter
                                             # tspan and make CI faster.
                                             plate_position=(0.2, 0.0),
                                             tspan=(0.0, 0.2),
                                             E=1e7, # Stiffer plate
                                             maxiters=500) [
                r"┌ Warning: Interrupted. Larger maxiters is needed.*\n",
                r"└ @ SciMLBase.*\n"
            ]
            @test sol.retcode == ReturnCode.MaxIters
            @test count_rhs_allocations(sol, semi) == 0

            # Now use split integration and verify that we need less than 400 iterations
            split_integration = SplitIntegrationCallback(RDPK3SpFSAL35(), adaptive=false,
                                                         dt=5e-5)
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "dam_break_plate_2d.jl"),
                                             # Use rounded dimensions to avoid warnings
                                             initial_fluid_size=(0.15, 0.29),
                                             # Move plate closer to be able to use a shorter
                                             # tspan and make CI faster.
                                             plate_position=(0.2, 0.0),
                                             tspan=(0.0, 0.2),
                                             E=1e7, # Stiffer plate
                                             maxiters=400,
                                             extra_callback=split_integration)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0

            # Now use split integration and verify that it is actually used for TLSPH
            # by using a time step that is too large and verifying that it is crashing.
            split_integration = SplitIntegrationCallback(RDPK3SpFSAL35(), adaptive=false,
                                                         dt=2e-4)
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "dam_break_plate_2d.jl"),
                                             # Use rounded dimensions to avoid warnings
                                             initial_fluid_size=(0.15, 0.29),
                                             # Move plate closer to be able to use a shorter
                                             # tspan and make CI faster.
                                             plate_position=(0.2, 0.0),
                                             tspan=(0.0, 0.2),
                                             E=1e7, # Stiffer plate
                                             maxiters=500,
                                             extra_callback=split_integration) [
                "┌ Warning: Instability detected. Aborting\n",
                r".*dt was forced below floating point epsilon.*\n",
                r"└ @ SciMLBase.*\n"
            ]
            @test sol.retcode == ReturnCode.Unstable
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fsi/dam_break_gate_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "dam_break_gate_2d.jl"),
                                             tspan=(0.0, 0.4),
                                             dtmax=1e-3)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fsi/falling_spheres_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "falling_spheres_2d.jl"),
                                             tspan=(0.0, 1.0))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end
    end

    @testset verbose=true "N-Body" begin
        @trixi_testset "n_body/n_body_solar_system.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "n_body",
                                                      "n_body_solar_system.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "n_body/n_body_benchmark_trixi.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "n_body",
                                                      "n_body_benchmark_trixi.jl")) [
                r"WARNING: Method definition interact!.*\n"
            ]
        end

        @trixi_testset "n_body/n_body_benchmark_reference.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "n_body",
                                                      "n_body_benchmark_reference.jl"))
        end

        @trixi_testset "n_body/n_body_benchmark_reference_faster.jl" begin
            @trixi_test_nowarn trixi_include(joinpath(examples_dir(), "n_body",
                                                      "n_body_benchmark_reference_faster.jl"))
        end
    end

    @testset verbose=true "Postprocessing" begin
        @trixi_testset "postprocessing/interpolation_plane.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "postprocessing",
                                                      "interpolation_plane.jl"),
                                             tspan=(0.0, 0.01)) [
                r"WARNING: importing deprecated binding Makie.*\n",
                r"WARNING: using deprecated binding Colors.*\n",
                r"WARNING: using deprecated binding PlotUtils.*\n",
                r"WARNING: Makie.* is deprecated.*\n",
                r"  likely near none:1\n",
                r", use .* instead.\n",
                r"┌ Info: The desired face size is not a multiple of the resolution [0-9.]+\.\n└ New resolution is set to [0-9.]+\.\n"
            ]
            @test sol.retcode == ReturnCode.Success
        end
        @trixi_testset "postprocessing/interpolation_point_line.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "postprocessing",
                                                      "interpolation_point_line.jl"))
            @test sol.retcode == ReturnCode.Success
        end
        @trixi_testset "postprocessing/postprocessing.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(),
                                                      "postprocessing",
                                                      "postprocessing.jl"))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "Restart" begin
            # Use an analytical velocity profile to test the custom function functionality
            # of open boundaries during restart and to enable a 10x lower sound speed factor
            # for faster simulation.
            # Analytical velocity evolution given in eq. 16 (Zhang et al., 2025)
            function poiseuille_velocity(y, t)

                # Base profile (stationary part)
                base_profile = (pressure_drop / (2 * dynamic_viscosity * flow_length)) * y *
                               (y - wall_distance)

                # Transient terms (Fourier series)
                transient_sum = 0.0

                for n in 0:10  # Limit to 10 terms for convergence
                    coefficient = (4 * pressure_drop * wall_distance^2) /
                                  (dynamic_viscosity * flow_length * pi^3 * (2 * n + 1)^3)

                    sine_term = sin(pi * y * (2 * n + 1) / wall_distance)

                    exp_term = exp(-((2 * n + 1)^2 * pi^2 * dynamic_viscosity * t) /
                                   (fluid_density * wall_distance^2))

                    transient_sum += coefficient * sine_term * exp_term
                end

                # Total velocity
                v_x = base_profile + transient_sum

                return -v_x
            end
            reference_velocity_in = (pos, t) -> SVector(poiseuille_velocity(pos[2], t), 0.0)

            # Run full simulation
            trixi_include(@__MODULE__,
                          joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
                          tspan=(0.0, 1.0), sound_speed_factor=10, particle_spacing=4e-5,
                          reference_velocity_in=reference_velocity_in)

            # Since this is an open boundary simulation, the number of active particles may
            # differ. The results must be interpolated to enable comparison with the restart
            # simulation. The fluid domain starts at `x = 10 * particle_spacing`.
            n_interpolation_points = 10
            start_point = [0.0 + 10 * particle_spacing, wall_distance / 2]
            end_point = [flow_length - 10 * particle_spacing, wall_distance / 2]
            result_full = interpolate_line(start_point, end_point, n_interpolation_points,
                                           semi, fluid_system, sol, cut_off_bnd=false)

            # Run half simulation and safe checkpoint
            trixi_include(@__MODULE__,
                          joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
                          tspan=(0.0, 0.5), sound_speed_factor=10, particle_spacing=4e-5,
                          reference_velocity_in=reference_velocity_in)

            tmp_dir = mktempdir()
            file_checkpoint = save_checkpoint(sol; output_directory=tmp_dir)

            # Load checkpoint and run remaining simulation
            sol_checkpoint = load_checkpoint(file_checkpoint)

            tspan = (0.5, 1.0)
            ode_checkpoint = semidiscretize_from_checkpoint(sol_checkpoint, tspan)

            callbacks = CallbackSet(UpdateCallback())

            sol_restart = solve(ode_checkpoint, RDPK3SpFSAL35(), abstol=1e-6, reltol=1e-4,
                                dtmax=1e-2, save_everystep=false, callback=callbacks)
            result_restart = interpolate_line(start_point, end_point,
                                              n_interpolation_points, sol_restart.prob.p,
                                              sol_restart.prob.p.systems[1],
                                              sol_restart, cut_off_bnd=false)

            @test isapprox(result_full.velocity, result_restart.velocity, rtol=2e-3)
            @test isapprox(result_full.density, result_restart.density, rtol=8e-5)
            @test isapprox(result_full.pressure, result_restart.pressure, rtol=8e-3)
        end
    end

    @testset verbose=true "Preprocessing" begin
        @trixi_testset "preprocessing/packing_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "preprocessing",
                                                      "packing_2d.jl"))
            @test sol.retcode == ReturnCode.Terminated
        end
        @trixi_testset "preprocessing/packing_2d.jl validation" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "preprocessing",
                                                      "packing_2d.jl"),
                                             particle_spacing=0.4)
            expected_coordinates = [-0.540548 -0.189943 0.191664 0.542741 -0.629391 -0.196159 0.197725 0.63081 -0.629447 -0.196158 0.19779 0.631121 -0.540483 -0.190015 0.191345 0.540433;
                                    -0.541127 -0.630201 -0.630119 -0.539294 -0.190697 -0.196942 -0.196916 -0.190324 0.190875 0.197074 0.196955 0.190973 0.541206 0.630323 0.630178 0.541314]

            @test isapprox(packed_ic.coordinates, expected_coordinates, atol=1e-5)
        end
        @trixi_testset "preprocessing/packing_3d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "preprocessing",
                                                      "packing_3d.jl"))
        end
    end

    @testset verbose=true "DEM" begin
        @trixi_testset "dem/rectangular_tank_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "dem",
                                                      "rectangular_tank_2d.jl"),
                                             tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end
    end
    @trixi_testset "dem/collapsing_sand_pile_3d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "dem",
                                                  "collapsing_sand_pile_3d.jl"),
                                         tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end
end
