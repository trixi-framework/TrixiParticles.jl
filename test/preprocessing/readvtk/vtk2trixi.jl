@testset verbose=true "vtk2trixi" begin
    @testset verbose=true "Basic Functionality - Inital File" begin
        min_coordinates = [(0, 0), (0, 0, 0)]
        n_particles_per_dimension = [(10, 20), (10, 20, 30)]
        velocity = [(1.0, 2.0), (1.0, 2.0, 3.0)]

        for i in 1:2 # 2d and 3d case
            saved_ic = RectangularShape(0.1, n_particles_per_dimension[i],
                                        min_coordinates[i], density=1.5,
                                        velocity=velocity[i], pressure=1000.0)
            filename = "is_write_out"
            file = trixi2vtk(saved_ic; filename=filename)

            loaded_ic = vtk2trixi(joinpath("out", filename * ".vtu"))

            @test saved_ic.coordinates == loaded_ic.coordinates
            @test saved_ic.velocity == loaded_ic.velocity
            @test saved_ic.density == loaded_ic.density
            @test saved_ic.pressure == loaded_ic.pressure
            #@test saved_ic.mass == loaded_ic.mass
        end
    end

    @testset verbose=true "Basic Functionality - Simulation File" begin end
end