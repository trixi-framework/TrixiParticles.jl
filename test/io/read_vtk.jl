@testset verbose=true "`vtk2trixi`" begin
    # Make sure that the `rand` calls below are deterministic
    Random.seed!(1)

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

            semi = Semidiscretization(fluid_system)

            # Create random ODE solutions
            v = rand(ndims(fluid_system), nparticles(fluid_system))
            pressure = rand(nparticles(fluid_system))
            v_ode = vec([v; pressure'])

            u = rand(ndims(fluid_system), nparticles(fluid_system))
            u_ode = vec(u)

            # Write out `FluidSystem` Simulation-File
            trixi2vtk(fluid_system, v_ode, u_ode, semi, 0.0,
                      nothing; system_name="tmp_file_fluid", output_directory=tmp_dir,
                      iter=1)

            # Load `FluidSystem` Simulation-File
            test = vtk2trixi(joinpath(tmp_dir, "tmp_file_fluid_1.vtu"))

            @test isapprox(u, test.coordinates, rtol=1e-5)
            @test isapprox(v, test.velocity, rtol=1e-5)
            @test isapprox(pressure, test.pressure, rtol=1e-5)
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
            semi = Semidiscretization(boundary_system)

            # Create dummy ODE solutions
            v_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))
            u_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))

            # Write out `BoundarySystem` Simulation-File
            trixi2vtk(boundary_system, v_ode, u_ode, semi, 0.0,
                      nothing; system_name="tmp_file_boundary", output_directory=tmp_dir,
                      iter=1)

            # Load `BoundarySystem` Simulation-File
            test = vtk2trixi(joinpath(tmp_dir, "tmp_file_boundary_1.vtu"))

            @test isapprox(boundary_system.coordinates, test.coordinates, rtol=1e-5)
            # The velocity is always zero for `BoundarySystem`
            @test isapprox(zeros(size(test.velocity)), test.velocity, rtol=1e-5)
            @test isapprox(boundary_model.pressure, test.pressure, rtol=1e-5)
            @test isapprox(boundary_model.cache.density, test.density, rtol=1e-5)
        end
    end
end
