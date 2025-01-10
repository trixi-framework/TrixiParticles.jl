@testset verbose=true "vtk2trixi" begin
    @testset verbose=true "Functionality Check - 2D" begin
        output_directory = "test/preprocessing/readvtk"

        Random.seed!(1)
        expected = InitialCondition(; coordinates=rand(2, 12), velocity=rand(2, 12),
                                    density=1.0, mass=2.0,
                                    pressure=3.0)

        # `InitialCondition`-Files
        trixi2vtk(expected; output_directory,
                  filename="test_initial_condition")

        initial_condition = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                               "test_initial_condition.vtu"))

        @test isapprox(expected.coordinates, initial_condition.coordinates, rtol=1e-5)
        @test isapprox(expected.velocity, initial_condition.velocity, rtol=1e-5)
        @test isapprox(expected.density, initial_condition.density, rtol=1e-5)
        @test isapprox(expected.pressure, initial_condition.pressure, rtol=1e-5)
        #@test isapprox(expected.mass, initial_condition.mass, rtol=1e-5) #TODO: wait until mass is written out with `write2vtk`

        # Simulations-Files

        # ==== Fluid System
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2

        fluid_system = EntropicallyDampedSPHSystem(expected, smoothing_kernel,
                                                   1.0, 10.0)

        fluid_system.cache.density .= expected.density

        # Write out `fluid_system` Simulation-File
        trixi2vtk(expected.velocity, expected.coordinates, 0.0, fluid_system,
                  nothing;
                  output_directory,
                  system_name="test_fluid_system", iter=1)
        # Load `fluid_system` Simulation-File
        fluid = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                   "test_fluid_system_1.vtu"))

        @test isapprox(expected.coordinates, fluid.coordinates, rtol=1e-5)
        @test isapprox(expected.velocity, fluid.velocity, rtol=1e-5)
        @test isapprox(expected.density, fluid.density, rtol=1e-5)
        @test isapprox(expected.pressure, fluid.pressure, rtol=1e-5)
        #@test isapprox(expected.fluid.mass, fluid.mass, rtol=1e-5) #TODO: wait until mass is written out with `write2vtk`

        # ==== Boundary System
        boundary_model = BoundaryModelDummyParticles(expected.density,
                                                     expected.mass,
                                                     SummationDensity(),
                                                     smoothing_kernel,
                                                     1.0)

        boundary_system = BoundarySPHSystem(expected, boundary_model)

        # Write out `boundary_system` Simulation-File
        trixi2vtk(expected.velocity, expected.coordinates, 0.0, boundary_system,
                  nothing;
                  output_directory,
                  system_name="test_boundary_system", iter=1)
        # Load `boundary_system` Simulation-File
        boundary = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                      "test_boundary_system_1.vtu"))

        @test isapprox(expected.coordinates, boundary.coordinates, rtol=1e-5)
        @test isapprox(expected.velocity, boundary.velocity, rtol=1e-5)
        @test isapprox(expected.density, boundary.density, rtol=1e-5)
        @test isapprox(expected.pressure, boundary.pressure, rtol=1e-5)
        #@test isapprox(expected.mass, boundary.mass, rtol=1e-5) #TODO: wait until mass is written out with `write2vtk`
    end
end