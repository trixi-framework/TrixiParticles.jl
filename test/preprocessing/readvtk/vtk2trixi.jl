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
                                                   ContinuityDensity(), state_equation,
                                                   WendlandC2Kernel{2}(), 0.2)

        trixi2vtk(saved_ic.velocity, saved_ic.coordinates, 0.0, fluid_system, nothing;)
        # trixi2vtk(sol.u[end], semi, 0.0, iter=1, output_directory="output",
        #           prefix="solution")

    end
end