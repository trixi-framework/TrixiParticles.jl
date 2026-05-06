using OrdinaryDiffEqLowStorageRK

@testset verbose=true "SolutionSavingCallback" begin
    function pvd_filenames(collection)
        return [match.captures[1] for match in eachmatch(r"file=\"([^\"]+)\"", collection)]
    end

    function run_solution_saving_test(callback; tspan=(0.0, 0.01), dt=0.005)
        coordinates = [0.0 0.1 0.0 0.1
                       0.0 0.0 0.1 0.1]
        velocity = zeros(size(coordinates))

        initial_condition = InitialCondition(; coordinates, velocity, density=1000.0,
                                             pressure=900.0, particle_spacing=0.1)
        fluid_system = EntropicallyDampedSPHSystem(initial_condition;
                                                   smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                                   smoothing_length=1.5,
                                                   sound_speed=1.5)
        semi = Semidiscretization(fluid_system)
        ode = semidiscretize(semi, tspan)

        return solve(ode, RDPK3SpFSAL35(); dt, adaptive=false,
                     save_everystep=false, callback)
    end

    @testset verbose=true "show" begin
        out = joinpath(tempdir(), "trixi_out")
        output_directory_padded = out * " "^(65 - length(out))

        @testset verbose=true "dt" begin
            callback = SolutionSavingCallback(dt=0.02, prefix="test", output_directory=out)

            show_compact = "SolutionSavingCallback(dt=0.02)"
            @test repr(callback) == show_compact

            show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ SolutionSavingCallback                                                                           │
            │ ══════════════════════                                                                           │
            │ dt: ……………………………………………………………………… 0.02                                                             │
            │ custom quantities: ……………………………… nothing                                                          │
            │ save initial solution: …………………… yes                                                              │
            │ save final solution: ………………………… yes                                                              │
            │ overwrite solution: …………………………… no                                                               │
            │ output directory: ………………………………… $(output_directory_padded)│
            │ prefix: …………………………………………………………… test                                                             │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
            @test repr("text/plain", callback) == show_box
        end

        @testset verbose=true "interval" begin
            callback = SolutionSavingCallback(interval=100, prefix="test", overwrite=true,
                                              output_directory=out)

            show_compact = "SolutionSavingCallback(interval=100)"
            @test repr(callback) == show_compact

            show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ SolutionSavingCallback                                                                           │
            │ ══════════════════════                                                                           │
            │ interval: ……………………………………………………… 100                                                              │
            │ custom quantities: ……………………………… nothing                                                          │
            │ save initial solution: …………………… yes                                                              │
            │ save final solution: ………………………… yes                                                              │
            │ overwrite solution: …………………………… yes                                                              │
            │ output directory: ………………………………… $(output_directory_padded)│
            │ prefix: …………………………………………………………… test                                                             │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
            @test repr("text/plain", callback) == show_box
        end

        @testset verbose=true "save_times" begin
            callback = SolutionSavingCallback(save_times=[1.0, 2.0, 3.0], prefix="test",
                                              output_directory=out)

            show_compact = "SolutionSavingCallback(save_times=[1.0, 2.0, 3.0])"
            @test repr(callback) == show_compact

            show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ SolutionSavingCallback                                                                           │
            │ ══════════════════════                                                                           │
            │ save_times: ………………………………………………… [1.0, 2.0, 3.0]                                                  │
            │ custom quantities: ……………………………… nothing                                                          │
            │ save initial solution: …………………… yes                                                              │
            │ save final solution: ………………………… yes                                                              │
            │ overwrite solution: …………………………… no                                                               │
            │ output directory: ………………………………… $(output_directory_padded)│
            │ prefix: …………………………………………………………… test                                                             │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
            @test repr("text/plain", callback) == show_box
        end
    end

    @testset verbose=true "custom quantities" begin
        # Test that `custom_quantity` correctly chooses the correct method
        quantity1(system, data, t) = data
        quantity2(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = 2
        quantity3() = 3

        system = Val(:mock_system)
        TrixiParticles.system_data(::Val{:mock_system}, dv_ode, du_ode, v_ode, u_ode,
                                   semi) = 1

        data = v_ode = u_ode = dv_ode = du_ode = semi = t = nothing

        @test TrixiParticles.custom_quantity(quantity1, system, dv_ode, du_ode, v_ode,
                                             u_ode, semi, t) == 1
        @test TrixiParticles.custom_quantity(quantity2, system, dv_ode, du_ode, v_ode,
                                             u_ode, semi, t) == 2
        @test_throws MethodError TrixiParticles.custom_quantity(quantity3, system, dv_ode,
                                                                du_ode, v_ode, u_ode,
                                                                semi, t)
    end

    @testset verbose=true "save_times writes metadata and requested/final times" begin
        mktempdir() do tmp_dir
            callback = SolutionSavingCallback(save_times=[0.003],
                                              output_directory=tmp_dir)
            run_solution_saving_test(callback)

            collection = read(joinpath(tmp_dir, "fluid_1.pvd"), String)
            filenames = pvd_filenames(collection)
            meta_data = JSON.parsefile(joinpath(tmp_dir, "meta.json"))
            solver_version = meta_data["simulation_info"]["solver_version"]

            @test solver_version == TrixiParticles.compute_git_hash()
            @test length(collect(eachmatch(r"DataSet", collection))) == 3
            @test all(file -> isfile(joinpath(tmp_dir, file)), filenames)
            @test occursin("timestep=\"0.0\"", collection)
            @test occursin("timestep=\"0.003\"", collection)
            @test occursin("timestep=\"0.01\"", collection)
        end
    end

    @testset verbose=true "save_times respects requested initial time" begin
        mktempdir() do tmp_dir
            callback = SolutionSavingCallback(save_times=[0.0, 0.003],
                                              save_initial_solution=false,
                                              save_final_solution=false,
                                              output_directory=tmp_dir)
            run_solution_saving_test(callback)

            collection = read(joinpath(tmp_dir, "fluid_1.pvd"), String)
            filenames = pvd_filenames(collection)

            @test length(collect(eachmatch(r"DataSet", collection))) == 2
            @test all(file -> isfile(joinpath(tmp_dir, file)), filenames)
            @test occursin("timestep=\"0.0\"", collection)
            @test occursin("timestep=\"0.003\"", collection)
            @test !occursin("timestep=\"0.01\"", collection)
        end
    end

    @testset verbose=true "callback PVD collection resets stale entries" begin
        mktempdir() do tmp_dir
            stale_callback = SolutionSavingCallback(interval=1, output_directory=tmp_dir,
                                                    save_final_solution=false)
            run_solution_saving_test(stale_callback)

            reset_callback = SolutionSavingCallback(interval=1, output_directory=tmp_dir,
                                                    save_initial_solution=false,
                                                    save_final_solution=false)
            run_solution_saving_test(reset_callback)

            reset_collection = read(joinpath(tmp_dir, "fluid_1.pvd"), String)
            @test length(collect(eachmatch(r"DataSet", reset_collection))) == 2
            @test !occursin("fluid_1_0.vtu", reset_collection)
            @test occursin("fluid_1_1.vtu", reset_collection)
            @test occursin("fluid_1_2.vtu", reset_collection)
        end
    end

    @testset verbose=true "callback PVD collection tracks overwritten file" begin
        mktempdir() do tmp_dir
            overwrite_callback = SolutionSavingCallback(interval=1, output_directory=tmp_dir,
                                                        prefix="overwrite",
                                                        overwrite=true)
            run_solution_saving_test(overwrite_callback)

            overwrite_collection = read(joinpath(tmp_dir, "overwrite_fluid_1.pvd"),
                                        String)
            @test length(collect(eachmatch(r"DataSet", overwrite_collection))) == 1
            @test isfile(joinpath(tmp_dir, "overwrite_fluid_1_current.vtu"))
            @test !isfile(joinpath(tmp_dir, "overwrite_fluid_1_1.vtu"))
            @test occursin("overwrite_fluid_1_current.vtu", overwrite_collection)
        end
    end
end
