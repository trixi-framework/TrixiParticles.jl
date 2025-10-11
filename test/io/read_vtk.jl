@testset verbose=true "`vtk2trixi`" begin
    mktempdir() do tmp_dir
        coordinates = fill(1.0, 2, 12)
        velocity = fill(2.0, 2, 12)

        expected_ic = InitialCondition(; coordinates=coordinates, velocity=velocity,
                                       density=1000.0, pressure=900.0, mass=50.0)

        @testset verbose=true "`InitialCondition`" begin
            trixi2vtk(expected_ic; filename="tmp_initial_condition",
                      output_directory=tmp_dir)

            test_ic = vtk2trixi(joinpath(tmp_dir, "tmp_initial_condition.vtu"))

            @test isapprox(expected_ic.coordinates, test_ic.coordinates, rtol=1e-5)
            @test isapprox(expected_ic.velocity, test_ic.velocity, rtol=1e-5)
            @test isapprox(expected_ic.density, test_ic.density, rtol=1e-5)
            @test isapprox(expected_ic.pressure, test_ic.pressure, rtol=1e-5)
        end

        @testset verbose=true "`AbstractFluidSystem`" begin
            fluid_system = EntropicallyDampedSPHSystem(expected_ic,
                                                       SchoenbergCubicSplineKernel{2}(),
                                                       1.5, 1.5)

            # Overwrite values because we skip the update step
            fluid_system.cache.density .= expected_ic.density

            semi = Semidiscretization(fluid_system)

            # Create random ODE solutions
            dvdu_ode = nothing
            v = fill(2.0, ndims(fluid_system), nparticles(fluid_system))
            pressure = fill(3.0, nparticles(fluid_system))
            v_ode = vec([v; pressure'])
            u = fill(1.0, ndims(fluid_system), nparticles(fluid_system))
            u_ode = vec(u)
            x = (; v_ode, u_ode)
            vu_ode = (; x)

            # Write out `AbstractFluidSystem` Simulation-File
            trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.0,
                      nothing; system_name="tmp_file_fluid", output_directory=tmp_dir,
                      iter=1)

            # Load `AbstractFluidSystem` Simulation-File
            test = vtk2trixi(joinpath(tmp_dir, "tmp_file_fluid_1.vtu"))

            @test isapprox(u, test.coordinates, rtol=1e-5)
            @test isapprox(v, test.velocity, rtol=1e-5)
            @test isapprox(pressure, test.pressure, rtol=1e-5)
            @test isapprox(fluid_system.cache.density, test.density, rtol=1e-5)
        end

        @testset verbose=true "`WallBoundarySystem`" begin
            boundary_model = BoundaryModelDummyParticles(expected_ic.density,
                                                         expected_ic.mass,
                                                         SummationDensity(),
                                                         SchoenbergCubicSplineKernel{2}(),
                                                         1.5)

            # Overwrite values because we skip the update step
            boundary_model.pressure .= expected_ic.pressure
            boundary_model.cache.density .= expected_ic.density

            boundary_system = WallBoundarySystem(expected_ic, boundary_model)
            semi = Semidiscretization(boundary_system)

            # Create dummy ODE solutions
            dvdu_ode = nothing
            v_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))
            u_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))
            x = (; v_ode, u_ode)
            vu_ode = (; x)

            # Write out `WallBoundarySystem` Simulation-File
            trixi2vtk(boundary_system, dvdu_ode, vu_ode, semi, 0.0,
                      nothing; system_name="tmp_file_boundary", output_directory=tmp_dir,
                      iter=1)

            # Load `WallBoundarySystem` Simulation-File
            test = vtk2trixi(joinpath(tmp_dir, "tmp_file_boundary_1.vtu"))

            @test isapprox(boundary_system.coordinates, test.coordinates, rtol=1e-5)
            # The velocity is always zero for `WallBoundarySystem`
            @test isapprox(zeros(size(test.velocity)), test.velocity, rtol=1e-5)
            @test isapprox(boundary_model.pressure, test.pressure, rtol=1e-5)
            @test isapprox(boundary_model.cache.density, test.density, rtol=1e-5)
        end
    end
end
