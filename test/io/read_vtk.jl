@testset verbose=true "`vtk2trixi`" begin
    mktempdir() do tmp_dir
        expected_ic = InitialCondition(; coordinates=rand(2, 12), velocity=rand(2, 12),
                                       density=rand(), pressure=rand(), mass=rand())

        @testset verbose=true "`InitialCondition`" begin
            trixi2vtk(expected_ic; filename="tmp_initial_condition",
                      output_directory=tmp_dir)

            test_ic = vtk2trixi(joinpath(tmp_dir, "tmp_initial_condition.vtu"))

            @test isapprox(expected_ic.coordinates, test_ic.coordinates, rtol=1e-5)
            @test isapprox(expected_ic.velocity, test_ic.velocity, rtol=1e-5)
            @test isapprox(expected_ic.density, test_ic.density, rtol=1e-5)
            @test isapprox(expected_ic.pressure, test_ic.pressure, rtol=1e-5)
        end

        @testset verbose=true "`FluidSystem`" begin
            fluid_system = EntropicallyDampedSPHSystem(expected_ic,
                                                       SchoenbergCubicSplineKernel{2}(),
                                                       rand(), rand())

            # Overwrite values because we skip the update step
            fluid_system.cache.density .= expected_ic.density

            # Create random ODE solutions
            u = rand(TrixiParticles.u_nvariables(fluid_system),
                     TrixiParticles.n_moving_particles(fluid_system))
            v = rand(TrixiParticles.v_nvariables(fluid_system),
                     TrixiParticles.n_moving_particles(fluid_system))

            # Write out `FluidSystem` Simulation-File
            trixi2vtk(v, u, 0.0, fluid_system,
                      nothing; system_name="tmp_file_fluid", output_directory=tmp_dir,
                      iter=1)
            # Load `fluid_system` Simulation-File
            test = vtk2trixi(joinpath(tmp_dir, "tmp_file_fluid_1.vtu"))

            @test isapprox(u, test.coordinates, rtol=1e-5)
            # Pressure is saved in `v`
            @test isapprox(v[1:2, :], test.velocity, rtol=1e-5)
            @test isapprox(v[end, :], test.pressure, rtol=1e-5)
            @test isapprox(fluid_system.cache.density, test.density, rtol=1e-5)
        end

        @testset verbose=true "`BoundarySystem`" begin
            boundary_model = BoundaryModelDummyParticles(expected_ic.density,
                                                         expected_ic.mass,
                                                         SummationDensity(),
                                                         SchoenbergCubicSplineKernel{2}(),
                                                         rand())

            # Overwrite values because we skip the update step
            boundary_model.pressure .= expected_ic.pressure
            boundary_model.cache.density .= expected_ic.density

            boundary_system = BoundarySPHSystem(expected_ic, boundary_model)

            # Write out `BoundarySystem` Simulation-File
            # There are no ODE solutions for `BoundarySystem`
            trixi2vtk(0.0, 0.0, 0.0, boundary_system,
                      nothing; system_name="tmp_file_boundary", output_directory=tmp_dir,
                      iter=1)

            # Load `boundary_system` Simulation-File
            test = vtk2trixi(joinpath(tmp_dir, "tmp_file_boundary_1.vtu"))

            @test isapprox(boundary_system.coordinates, test.coordinates, rtol=1e-5)
            # The velocity is always zero for `BoundarySystem`
            @test isapprox(zeros(size(test.velocity)), test.velocity, rtol=1e-5)
            @test isapprox(boundary_model.pressure, test.pressure, rtol=1e-5)
            @test isapprox(boundary_model.cache.density, test.density, rtol=1e-5)
        end
    end
end
