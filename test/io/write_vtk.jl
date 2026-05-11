@testset verbose=true "`trixi2vtk`" begin
    mktempdir() do tmp_dir
        coordinates = fill(1.0, 2, 12)
        velocity = fill(2.0, 2, 12)

        initial_condition = InitialCondition(; coordinates, velocity, density=1000.0,
                                             pressure=900.0, particle_spacing=0.1)
        fluid_system = EntropicallyDampedSPHSystem(initial_condition;
                                                   smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                                   smoothing_length=1.5,
                                                   sound_speed=1.5)

        # Overwrite values because we skip the update step
        fluid_system.cache.density .= initial_condition.density

        semi = Semidiscretization(fluid_system)

        v = fill(2.0, ndims(fluid_system), nparticles(fluid_system))
        pressure = fill(3.0, nparticles(fluid_system))
        v_ode = vec([v; pressure'])
        u = fill(1.0, ndims(fluid_system), nparticles(fluid_system))
        u_ode = vec(u)
        dv_ode = zero(v_ode)
        du_ode = zero(u_ode)
        x = (; v_ode, u_ode)
        vu_ode = (; x)
        dvdu_ode = (; x=(; v_ode=dv_ode, u_ode=du_ode))

        @testset verbose=true "PVD Collection Appends Iterations" begin
            trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.0,
                      nothing; system_name="tmp_file_fluid_collection",
                      output_directory=tmp_dir, iter=0)
            trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.1,
                      nothing; system_name="tmp_file_fluid_collection",
                      output_directory=tmp_dir, iter=1)

            collection = read(joinpath(tmp_dir, "tmp_file_fluid_collection.pvd"),
                              String)

            @test occursin("DataSet", collection)
            @test occursin("tmp_file_fluid_collection_0.vtu", collection)
            @test occursin("tmp_file_fluid_collection_1.vtu", collection)
        end

        @testset verbose=true "PVD Collection Default Overwrite" begin
            ode = semidiscretize(semi, (0.0, 1.0))
            trixi2vtk(ode.u0, semi, 0.75; output_directory=tmp_dir,
                      prefix="tmp_file_fluid_default")

            collection = read(joinpath(tmp_dir, "tmp_file_fluid_default_fluid_1.pvd"),
                              String)

            @test length(collect(eachmatch(r"DataSet", collection))) == 1
            @test isfile(joinpath(tmp_dir,
                                  "tmp_file_fluid_default_fluid_1_current.vtu"))
            @test occursin("timestep=\"0.75\"", collection)
            @test occursin("tmp_file_fluid_default_fluid_1_current.vtu",
                           collection)
        end

        @testset verbose=true "VTK Metadata Contains Solver Version" begin
            ode = semidiscretize(semi, (0.0, 1.0))
            trixi2vtk(ode.u0, semi, 0.75; output_directory=tmp_dir,
                      prefix="tmp_file_fluid_metadata")

            vtk_output = read(joinpath(tmp_dir,
                                       "tmp_file_fluid_metadata_fluid_1_current.vtu"),
                              String)

            @test occursin("solver_version=\"$(TrixiParticles.compute_git_hash())\"",
                           vtk_output)
        end

        @testset verbose=true "PVD Collection Tracks Overwritten File" begin
            trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.25,
                      nothing; system_name="tmp_file_fluid_overwrite",
                      output_directory=tmp_dir, iter=0, overwrite=true)
            trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.5,
                      nothing; system_name="tmp_file_fluid_overwrite",
                      output_directory=tmp_dir, iter=1, overwrite=true)

            current_collection = read(joinpath(tmp_dir, "tmp_file_fluid_overwrite.pvd"),
                                      String)

            @test length(collect(eachmatch(r"DataSet", current_collection))) == 1
            @test occursin("timestep=\"0.5\"", current_collection)
            @test occursin("tmp_file_fluid_overwrite_current.vtu",
                           current_collection)
        end
    end
end
