@testset verbose=true "vtk2trixi" begin
    output_directory = joinpath("test/preprocessing/readvtk")

    Random.seed!(1)
    expected_ic = InitialCondition(; coordinates=rand(2, 12), velocity=rand(2, 12),
                                   density=1.0, mass=2.0,
                                   pressure=3.0)

    # `InitialCondition`-Files
    trixi2vtk(expected_ic; output_directory,
              filename="test_initial_condition")

    test_ic = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                 "test_initial_condition.vtu"))

    @test isapprox(expected_ic.coordinates, test_ic.coordinates, rtol=1e-5)
    @test isapprox(expected_ic.velocity, test_ic.velocity, rtol=1e-5)
    @test isapprox(expected_ic.density, test_ic.density, rtol=1e-5)
    @test isapprox(expected_ic.pressure, test_ic.pressure, rtol=1e-5)
    #@test isapprox(expected_ic.mass, test_ic.mass, rtol=1e-5) #TODO: wait until mass is written out with `write2vtk`

    # Simulations-Files

    # ==== Fluid System
    smoothing_kernel = Val(:smoothing_kernel)
    TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2

    fluid_system = EntropicallyDampedSPHSystem(expected_ic, smoothing_kernel,
                                               1.0, 10.0)

    # Overwrite values because we skip the update step
    fluid_system.cache.density .= expected_ic.density
    v = [expected_ic.velocity; expected_ic.pressure']

    # Write out `fluid_system` Simulation-File
    trixi2vtk(v, expected_ic.coordinates, 0.0, fluid_system,
              nothing;
              output_directory,
              system_name="test_fluid_system", iter=1)
    # Load `fluid_system` Simulation-File
    test_fluid = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                    "test_fluid_system_1.vtu"))

    @test isapprox(expected_ic.coordinates, test_fluid.coordinates, rtol=1e-5)
    @test isapprox(expected_ic.velocity, test_fluid.velocity, rtol=1e-5)
    @test isapprox(expected_ic.density, test_fluid.density, rtol=1e-5)
    @test isapprox(expected_ic.pressure, test_fluid.pressure, rtol=1e-5)
    #@test isapprox(expected_ic.test_fluid.mass, test_fluid.mass, rtol=1e-5) #TODO: wait until mass is written out with `write2vtk`

    # ==== Boundary System
    boundary_model = BoundaryModelDummyParticles(expected_ic.density,
                                                 expected_ic.mass,
                                                 SummationDensity(),
                                                 smoothing_kernel,
                                                 1.0)

    # Overwrite values because we skip the update step
    boundary_model.pressure .= expected_ic.pressure
    boundary_model.cache.density .= expected_ic.density

    boundary_system = BoundarySPHSystem(expected_ic, boundary_model)

    # Write out `boundary_system` Simulation-File
    trixi2vtk(expected_ic.velocity, expected_ic.coordinates, 0.0, boundary_system,
              nothing;
              output_directory,
              system_name="test_boundary_system", iter=1)
    # Load `boundary_system` Simulation-File
    test_boundary = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                       "test_boundary_system_1.vtu"))

    @test isapprox(expected_ic.coordinates, test_boundary.coordinates, rtol=1e-5)
    @test isapprox(zeros(2, 12), test_boundary.velocity, rtol=1e-5)
    @test isapprox(expected_ic.density, test_boundary.density, rtol=1e-5)
    @test isapprox(expected_ic.pressure, test_boundary.pressure, rtol=1e-5)
    #@test isapprox(expected_ic.mass, test_boundary.mass, rtol=1e-5) #TODO: wait until mass is written out with `write2vtk`
end