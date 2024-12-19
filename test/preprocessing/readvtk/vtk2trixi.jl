@testset verbose=true "vtk2trixi" begin
    @testset verbose=true "Functionality Check - 2D" begin

        # 'InitialCondition'-Files
        saved_ic = RectangularShape(0.1, (10, 20),
                                    (0, 0), density=1.5,
                                    velocity=(1.0, 2.0), pressure=1000.0)

        file = trixi2vtk(saved_ic; output_directory="test/preprocessing/readvtk",
                         filename="initial_condition")

        loaded_ic = vtk2trixi(joinpath("test/preprocessing/readvtk",
                                       "initial_condition.vtu"))

        @test isapprox(saved_ic.coordinates, loaded_ic.coordinates)
        @test isapprox(saved_ic.velocity, loaded_ic.velocity)
        @test isapprox(saved_ic.density, loaded_ic.density)
        @test isapprox(saved_ic.pressure, loaded_ic.pressure)
        @test isapprox(saved_ic.mass, loaded_ic.mass)

        # 'Fluidsystem'-Files
        state_equation = StateEquationCole(; sound_speed=1.0, reference_density=1.0,
                                           exponent=1, clip_negative_pressure=false)
        fluid_system = WeaklyCompressibleSPHSystem(saved_ic,
                                                   SummationDensity(), state_equation,
                                                   WendlandC2Kernel{2}(), 0.2)

        saved_fs = trixi2vtk(saved_ic.velocity, saved_ic.coordinates, 0.0, fluid_system,
                             nothing;
                             output_directory="test/preprocessing/readvtk", iter=1)

        loaded_fs = vtk2trixi(joinpath("test/preprocessing/readvtk", "fluid_1.vtu"))

        @test isapprox(saved_fs.coordinates, loaded_fs.coordinates)
        @test isapprox(saved_fs.velocity, loaded_fs.velocity)
        @test isapprox(saved_fs.density, loaded_fs.density)
        @test isapprox(saved_fs.pressure, loaded_fs.pressure)
        @test isapprox(saved_fs.mass, loaded_fs.mass)
    end
end