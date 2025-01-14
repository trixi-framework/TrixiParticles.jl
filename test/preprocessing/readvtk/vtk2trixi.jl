@testset verbose=true "vtk2trixi" begin
    output_directory = "test/preprocessing/readvtk"

    Random.seed!(1)
    saved_ic = InitialCondition(; coordinates=rand(2, 12), velocity=rand(2, 12),
                                density=1.0, mass=2.0,
                                pressure=3.0)

    # `InitialCondition`-Files
    trixi2vtk(saved_ic; output_directory,
              filename="test_initial_condition")

    loaded_ic = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                   "test_initial_condition.vtu"))

    @test isapprox(saved_ic.coordinates, loaded_ic.coordinates, rtol=1e-5)
    @test isapprox(saved_ic.velocity, loaded_ic.velocity, rtol=1e-5)
    @test isapprox(saved_ic.density, loaded_ic.density, rtol=1e-5)
    @test isapprox(saved_ic.pressure, loaded_ic.pressure, rtol=1e-5)
    #@test isapprox(saved_ic.mass, loaded_ic.mass, rtol=1e-5) #TODO: wait until mass is written out with `write2vtk`

    # Simulations-Files

    # ==== Fluid System
    smoothing_kernel = Val(:smoothing_kernel)
    TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2

    fluid_system = EntropicallyDampedSPHSystem(saved_ic, smoothing_kernel,
                                               1.0, 10.0)

    # Overwrite values because we skip the update step
    fluid_system.cache.density .= saved_ic.density
    fluid_velocity = [saved_ic.velocity; saved_ic.pressure']

    # Write out `fluid_system` Simulation-File
    trixi2vtk(fluid_velocity, saved_ic.coordinates, 0.0, fluid_system,
              nothing;
              output_directory,
              system_name="test_fluid_system", iter=1)
    # Load `fluid_system` Simulation-File
    loaded_fluid = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                      "test_fluid_system_1.vtu"))

    @test isapprox(saved_ic.coordinates, loaded_fluid.coordinates, rtol=1e-5)
    @test isapprox(saved_ic.velocity, loaded_fluid.velocity, rtol=1e-5)
    @test isapprox(saved_ic.density, loaded_fluid.density, rtol=1e-5)
    @test isapprox(saved_ic.pressure, loaded_fluid.pressure, rtol=1e-5)
    #@test isapprox(saved_ic.loaded_fluid.mass, loaded_fluid.mass, rtol=1e-5) #TODO: wait until mass is written out with `write2vtk`

    # ==== Boundary System
    boundary_model = BoundaryModelDummyParticles(saved_ic.density,
                                                 saved_ic.mass,
                                                 SummationDensity(),
                                                 smoothing_kernel,
                                                 1.0)

    # Overwrite values because we skip the update step
    boundary_model.pressure .= saved_ic.pressure
    boundary_model.cache.density .= saved_ic.density

    boundary_system = BoundarySPHSystem(saved_ic, boundary_model)

    # Write out `boundary_system` Simulation-File
    trixi2vtk(saved_ic.velocity, saved_ic.coordinates, 0.0, boundary_system,
              nothing;
              output_directory,
              system_name="test_boundary_system", iter=1)
    # Load `boundary_system` Simulation-File
    loaded_boundary = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                         "test_boundary_system_1.vtu"))

    @test isapprox(saved_ic.coordinates, loaded_boundary.coordinates, rtol=1e-5)
    @test isapprox(zeros(2, 12), loaded_boundary.velocity, rtol=1e-5)
    @test isapprox(saved_ic.density, loaded_boundary.density, rtol=1e-5)
    @test isapprox(saved_ic.pressure, loaded_boundary.pressure, rtol=1e-5)
    #@test isapprox(saved_ic.mass, loaded_boundary.mass, rtol=1e-5) #TODO: wait until mass is written out with `write2vtk`
end