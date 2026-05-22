# Smoke tests, i.e., tests to verify that examples are running without crashing,
# but without checking the correctness of the solution.
@testset verbose=true "Examples" begin
    include("examples_fluid.jl")

    @testset verbose=true "Structure" begin
        @trixi_testset "structure/oscillating_beam_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "structure",
                                                      "oscillating_beam_2d.jl"),
                                             tspan=(0.0, 0.1)) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "structure/oscillating_beam_2d.jl with penalty force" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "structure",
                                                      "oscillating_beam_2d.jl"),
                                             tspan=(0.0, 0.1),
                                             penalty_force=PenaltyForceGanzenmueller(alpha=1.0),
                                             sol=nothing) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n"
            ]
            # Use a fixed time step, tuned to the maximum stable step size for this example.
            # Together with the very large penalty force alpha, this test will crash with
            # "Instability detected" if the penalty force is not working correctly.
            callbacks = CallbackSet((@__MODULE__).callbacks, StepsizeCallback(cfl=1.6))
            sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0,
                        save_everystep=false, callback=callbacks)
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "structure/oscillating_beam_2d.jl with penalty force and viscosity" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "structure",
                                                      "oscillating_beam_2d.jl"),
                                             tspan=(0.0, 0.1),
                                             penalty_force=PenaltyForceGanzenmueller(alpha=0.1),
                                             viscosity=ArtificialViscosityMonaghan(alpha=0.01)) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            else
                @test count_rhs_allocations(sol) == 0
            end
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
                                             clamped_particles_motion=prescribed_motion) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "structure/oscillating_beam_2d.jl with MechanicalWorkCalculatorCallback" begin
            # Load variables from the example
            trixi_include(@__MODULE__,
                          joinpath(examples_dir(), "structure", "oscillating_beam_2d.jl"),
                          ode=nothing, sol=nothing, E=1e5)

            # We simply clamp all particles, move them up against gravity, and verify that
            # the energy calculated is just the potential energy difference.
            movement_function(x, t) = x + SVector(0.0, t)
            is_moving(t) = true

            tests = Dict(
                "all particles clamped" => eachparticle(structure),
                # Clamp everything but the top two layers of the beam
                "some particles clamped" => 1:(nparticles(structure) - 162)
            )
            rtol = Dict(
                "all particles clamped" => sqrt(eps()),
                # Clamp everything but the top two layers of the beam.
                # We don't expect very accurate results here, this is more of a smoke test.
                "some particles clamped" => 0.1
            )
            @testset "$name" for (name, clamped_particles) in tests
                # We need a new `PrescribedMotion` for each test due to
                # https://github.com/trixi-framework/TrixiParticles.jl/issues/1020
                prescribed_motion = PrescribedMotion(movement_function, is_moving)
                structure_system = TotalLagrangianSPHSystem(structure; smoothing_kernel,
                                                            smoothing_length,
                                                            young_modulus=material.E,
                                                            poisson_ratio=material.nu,
                                                            clamped_particles,
                                                            acceleration=(0.0, -gravity),
                                                            clamped_particles_motion=prescribed_motion)

                semi = Semidiscretization(structure_system)
                ode = semidiscretize(semi, (0.0, 1.0))
                system = ode.p.semi.systems[1]

                mechanical_work_calculator = MechanicalWorkCalculatorCallback(system, semi;
                                                                              interval=1)

                sol = @trixi_test_nowarn solve(ode, RDPK3SpFSAL49(), save_everystep=false,
                                               callback=mechanical_work_calculator)

                @test sol.retcode == ReturnCode.Success
                @test count_rhs_allocations(sol) == 0

                # Potential energy difference should be m * g * h
                @test isapprox(calculated_mechanical_work(mechanical_work_calculator),
                               sum(system.mass) * gravity * 1,
                               rtol=rtol[name])
            end
        end

        @trixi_testset "structure/colliding_rigid_spheres_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "structure",
                                                      "colliding_rigid_spheres_2d.jl"),
                                             tspan=(0.0, 0.6))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol) == 0
        end
    end

    @testset verbose=true "FSI" begin
        @trixi_testset "fluid/hydrostatic_water_column_2d.jl with MechanicalWorkCalculatorCallback and moving TLSPH walls" begin
            # In this test, we move a water-filled tank up against gravity by 1 unit
            # and verify that the work accumulated by the `MechanicalWorkCalculatorCallback`
            # matches the expected potential energy difference.

            # Load variables from the example
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fluid",
                                                      "hydrostatic_water_column_2d.jl"),
                                             # Use tank without airspace
                                             tank_size=(1.0, 0.9),
                                             # Higher speed of sound for smaller error
                                             # due to compressibility.
                                             sound_speed=50.0,
                                             ode=nothing, sol=nothing)

            # Move the tank up against gravity smoothly by 1 unit over 1 second
            function movement_function(x, t)
                return x + SVector(0.0, 0.5 * sin(pi * (t - 0.5)) + 0.5)
            end
            is_moving(t) = true
            prescribed_motion = PrescribedMotion(movement_function, is_moving)

            # Create TLSPH system for the tank walls and clamp all particles.
            # This is identical to a `WallBoundarySystem`, but now we can
            # use the `MechanicalWorkCalculatorCallback` to compute the mechanical work.
            boundary_spacing = tank.boundary.particle_spacing
            tlsph_kernel = WendlandC2Kernel{2}()
            tlsph_smoothing_length = sqrt(2) * boundary_spacing

            tlsph_system = TotalLagrangianSPHSystem(tank.boundary;
                                                    smoothing_kernel=tlsph_kernel,
                                                    smoothing_length=tlsph_smoothing_length,
                                                    young_modulus=1.0e6,
                                                    poisson_ratio=0.3,
                                                    clamped_particles=eachparticle(tank.boundary),
                                                    acceleration=(0.0, -gravity),
                                                    clamped_particles_motion=prescribed_motion,
                                                    boundary_model)

            semi = Semidiscretization(fluid_system, tlsph_system,
                                      parallelization_backend=PolyesterBackend())
            ode = semidiscretize(semi, (0.0, 1.0))
            fluid_system_new = ode.p.semi.systems[1]
            tlsph_system_new = ode.p.semi.systems[2]

            # Mechanical work calculators for fluid + tank and fluid only
            mechanical_work_calculator1 = MechanicalWorkCalculatorCallback(tlsph_system_new,
                                                                           semi;
                                                                           interval=1)
            mechanical_work_calculator2 = MechanicalWorkCalculatorCallback(tlsph_system_new,
                                                                           semi;
                                                                           interval=1,
                                                                           only_compute_force_on_fluid=true)

            sol = @trixi_test_nowarn solve(ode, RDPK3SpFSAL35(), save_everystep=false,
                                           callback=CallbackSet(info_callback,
                                                                mechanical_work_calculator1,
                                                                mechanical_work_calculator2))

            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol) == 0

            # Potential energy difference should be m * g * h and h = 1.
            # Since all particles are clamped, the work done through clamped particles
            # should be the same as that done on the fluid.
            expected_energy_fluid = sum(fluid_system_new.mass) * gravity * 1
            expected_energy_tank = sum(tlsph_system_new.mass) * gravity * 1

            # We don't expect very accurate results here because the fluid is weakly
            # compressible and is deformed during the simulation.
            # A slower prescribed motion (e.g., over 2 seconds instead of 1) or a higher
            # speed of sound in the fluid would improve accuracy (and increase runtime).
            @test isapprox(calculated_mechanical_work(mechanical_work_calculator1),
                           expected_energy_fluid + expected_energy_tank, rtol=5e-4)
            @test isapprox(calculated_mechanical_work(mechanical_work_calculator2),
                           expected_energy_fluid,
                           rtol=5e-4)
        end

        @trixi_testset "fsi/falling_water_column_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "falling_water_column_2d.jl"),
                                             tspan=(0.0, 0.4)) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "fsi/dam_break_plate_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "dam_break_plate_2d.jl"),
                                             # Use rounded dimensions to avoid warnings
                                             initial_fluid_size=(0.15, 0.29),
                                             tspan=(0.0, 0.4)) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "fsi/dam_break_plate_2d.jl with SortingCallback" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "dam_break_plate_2d.jl"),
                                             # Use rounded dimensions to avoid warnings
                                             initial_fluid_size=(0.15, 0.29),
                                             tspan=(0.0, 0.05),
                                             extra_callback=SortingCallback(dt=0.02)) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            else
                @test count_rhs_allocations(sol) == 0
            end
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
                r"\[ Info: To create the self-interaction neighborhood search.*\n",
                "┌ Warning: Verbosity toggle: max_iters \n",
                r".*Interrupted. Larger maxiters is needed.*\n",
                r"└ @ SciMLBase.*\n"
            ]
            @test sol.retcode == ReturnCode.MaxIters
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            else
                @test count_rhs_allocations(sol) == 0
            end

            # Use split integration and verify that we need fewer than 400 iterations
            split_integration = SplitIntegrationCallback(CarpenterKennedy2N54(williamson_condition=false),
                                                         dt=5e-5)
            callbacks = CallbackSet(info_callback, saving_callback, split_integration)
            # Don't use `sol` here, as this local variable would shadow the global `sol`
            # from the `trixi_include` above and break the `sol.retcode` above.
            sol2 = @trixi_test_nowarn solve(ode, RDPK3SpFSAL49(), maxiters=400,
                                            save_everystep=false, callback=callbacks)

            @test sol2.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol2; split_integration) < 200
            else
                @test count_rhs_allocations(sol2; split_integration) == 0
            end

            # Use stage-level coupling and verify that it is not compatible with
            # the fluid time integration scheme `RDPK3SpFSAL49`.
            split_integration = SplitIntegrationCallback(CarpenterKennedy2N54(williamson_condition=false),
                                                         dt=5e-5, stage_coupling=true)
            callbacks = CallbackSet(info_callback, saving_callback, split_integration)

            msg = "stage-level coupling with `SplitIntegrationCallback` requires that"
            @test_throws msg solve(ode, RDPK3SpFSAL49(), maxiters=400,
                                   save_everystep=false, callback=callbacks)

            # Use stage-level coupling with a compatible fluid time integration scheme
            split_integration = SplitIntegrationCallback(CarpenterKennedy2N54(williamson_condition=false),
                                                         stage_coupling=true, dt=5e-5)
            stepsize_callback = StepsizeCallback(cfl=1.2)
            callbacks = CallbackSet(info_callback, saving_callback, split_integration,
                                    stepsize_callback)
            sol2 = @trixi_test_nowarn solve(ode,
                                            CarpenterKennedy2N54(williamson_condition=false),
                                            maxiters=400, dt=1.0,
                                            save_everystep=false, callback=callbacks)

            @test sol2.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol2; split_integration) < 200
            else
                @test count_rhs_allocations(sol2; split_integration) == 0
            end

            # Use split integration and verify that it is actually used for TLSPH
            # by using a time step that is too large and verifying that it is crashing.
            split_integration = SplitIntegrationCallback(CarpenterKennedy2N54(williamson_condition=false),
                                                         stage_coupling=true, dt=2e-4)
            callbacks = CallbackSet(info_callback, saving_callback, split_integration,
                                    stepsize_callback)

            msg = "`SplitIntegrationCallback` failed with return code Unstable."
            @test_throws msg solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                                   maxiters=400, dt=1.0,
                                   save_everystep=false, callback=callbacks)
        end

        @trixi_testset "fsi/dam_break_gate_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "dam_break_gate_2d.jl"),
                                             tspan=(0.0, 0.4),
                                             dtmax=1e-3) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "fsi/falling_spheres_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "falling_spheres_2d.jl"),
                                             tspan=(0.0, 1.0)) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION > v"1.11"
                # Newer Version than 1.11 produce more allocations
                # todo: unclear where this is from
                @test count_rhs_allocations(sol) < 1000
            else
                # Older Julia versions than 1.12 produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 200
            end
        end

        @trixi_testset "fsi/falling_rigid_spheres_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "falling_rigid_spheres_2d.jl"),
                                             tspan=(0.0, 0.5))
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 500
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "fsi/hydrostatic_water_column_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "hydrostatic_water_column_2d.jl"),
                                             tspan=(0.0, 0.1), n_particles_plate_y=3) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n",
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 500
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "fsi/falling_rotating_rigid_squares_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "falling_rotating_rigid_squares_2d.jl"),
                                             tspan=(0.0, 0.5))
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 500
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "fsi/falling_rotating_rigid_squares_w_buoys_2d.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "falling_rotating_rigid_squares_w_buoys_2d.jl"),
                                             tspan=(0.0, 0.5)) [
                r"WARNING: Method definition structure_boundary_model.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH or rigid.
                @test count_rhs_allocations(sol) < 2000
            else
                @test count_rhs_allocations(sol) == 0
            end
        end

        @trixi_testset "fsi/hydrostatic_water_column_2d.jl with EDAC" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fsi",
                                                      "hydrostatic_water_column_2d.jl"),
                                             tspan=(0.0, 0.1), n_particles_plate_y=3,
                                             use_edac=true) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n",
            ]
            @test sol.retcode == ReturnCode.Success
            if VERSION < v"1.12"
                # Older Julia versions produce allocations because `get_neighborhood_search`
                # is not type-stable with TLSPH.
                @test count_rhs_allocations(sol) < 500
            else
                @test count_rhs_allocations(sol) == 0
            end
        end
    end

    @testset verbose=true "N-Body" begin
        include("n_body_system.jl")

        @trixi_testset "n_body/n_body_newtonian_gravity.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "n_body",
                                                      "n_body_newtonian_gravity.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol) == 0
        end

        @trixi_testset "n_body/n_body_solar_system.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "n_body",
                                                      "n_body_solar_system.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol) == 0
        end

        @trixi_testset "n_body/n_body_benchmark_trixi.jl" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "n_body",
                                                      "n_body_benchmark_trixi.jl")) [
                r"WARNING: Method definition interact!.*\n"
            ]

            v_ode = vec(velocity)
            u_ode = vec(coordinates)
            dv_ode = similar(v_ode)
            du_ode = similar(u_ode)
            p = (; semi, split_integration_data=nothing)

            # Keep the benchmark system's generic `kick!`/`drift!` path allocation-free.
            # This avoids noisy timing assertions while still catching practical regressions.
            filename = tempname()
            try
                open(filename, "w") do f
                    redirect_stderr(f) do
                        TrixiParticles.disable_debug_timings()
                    end
                end

                TrixiParticles.kick!(dv_ode, v_ode, u_ode, p, 0.0)
                TrixiParticles.drift!(du_ode, v_ode, u_ode, p, 0.0)

                @test @allocated(TrixiParticles.kick!(dv_ode, v_ode, u_ode, p, 0.0)) == 0
                @test @allocated(TrixiParticles.drift!(du_ode, v_ode, u_ode, p, 0.0)) == 0
            finally
                open(filename, "w") do f
                    redirect_stderr(f) do
                        TrixiParticles.enable_debug_timings()
                    end
                end
            end
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
            @test count_rhs_allocations(sol) == 0
        end
    end
    @trixi_testset "dem/collapsing_sand_pile_3d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "dem",
                                                  "collapsing_sand_pile_3d.jl"),
                                         tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol) == 0
    end
end
