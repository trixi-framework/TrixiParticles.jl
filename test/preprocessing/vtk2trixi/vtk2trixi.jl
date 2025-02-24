@testset verbose=true "`vtk2trixi`" begin
    output_directory = joinpath("test/preprocessing/readvtk")

    smoothing_kernel = Val(:smoothing_kernel)
    TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2

    expected_ic = InitialCondition(; coordinates=rand(2, 12), velocity=rand(2, 12),
                                   density=rand(), pressure=rand(), mass=rand())

    @testset verbose=true "`InitialCondition`" begin
        trixi2vtk(expected_ic; output_directory,
                  filename=joinpath("test_initial_condition"))

        test_ic = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                     "test_initial_condition.vtu"))

        @test isapprox(expected_ic.coordinates, test_ic.coordinates, rtol=1e-5)
        @test isapprox(expected_ic.velocity, test_ic.velocity, rtol=1e-5)
        @test isapprox(expected_ic.density, test_ic.density, rtol=1e-5)
        @test isapprox(expected_ic.pressure, test_ic.pressure, rtol=1e-5)
        # TODO: Test the mass once it is written out with `write2vtk`.
    end

    @testset verbose=true "`FluidSystem`" begin
        fluid_system = EntropicallyDampedSPHSystem(expected_ic, smoothing_kernel, rand(),
                                                   rand())

        # Overwrite values because we skip the update step
        fluid_system.cache.density .= expected_ic.density

        # Create random ODE solutions
        u = rand(TrixiParticles.u_nvariables(fluid_system),
                 TrixiParticles.n_moving_particles(fluid_system))
        v = rand(TrixiParticles.v_nvariables(fluid_system),
                 TrixiParticles.n_moving_particles(fluid_system))

        # Write out `FluidSystem` Simulation-File
        trixi2vtk(v, u, 0.0, fluid_system,
                  nothing; output_directory, system_name=joinpath("test_fluid_system"),
                  iter=1)
        # Load `fluid_system` Simulation-File
        test = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                  "test_fluid_system_1.vtu"))

        @test isapprox(u, test.coordinates, rtol=1e-5)
        # Pressure is saved in `v`
        @test isapprox(v[1:2, :], test.velocity, rtol=1e-5)
        @test isapprox(v[end, :], test.pressure, rtol=1e-5)
        @test isapprox(fluid_system.cache.density, test.density, rtol=1e-5)
        # TODO: Test the mass once it is written out with `write2vtk`.
    end

    @testset verbose=true "`BoundarySystem`" begin
        boundary_model = BoundaryModelDummyParticles(expected_ic.density, expected_ic.mass,
                                                     SummationDensity(), smoothing_kernel,
                                                     rand())

        # Overwrite values because we skip the update step
        boundary_model.pressure .= expected_ic.pressure
        boundary_model.cache.density .= expected_ic.density

        boundary_system = BoundarySPHSystem(expected_ic, boundary_model)

        # Write out `BoundarySystem` Simulation-File
        # There are no ODE solutions for `BoundarySystem`
        trixi2vtk(0.0, 0.0, 0.0, boundary_system,
                  nothing; output_directory, system_name=joinpath("test_boundary_system"),
                  iter=1)

        # Load `boundary_system` Simulation-File
        test = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                  "test_boundary_system_1.vtu"))

        @test isapprox(boundary_system.coordinates, test.coordinates, rtol=1e-5)
        # The velocity is always zero for `BoundarySystem`
        @test isapprox(zeros(size(test.velocity)), test.velocity, rtol=1e-5)
        @test isapprox(boundary_model.pressure, test.pressure, rtol=1e-5)
        @test isapprox(boundary_model.cache.density, test.density, rtol=1e-5)
        # TODO: Test the mass once it is written out with `write2vtk`.
    end
end