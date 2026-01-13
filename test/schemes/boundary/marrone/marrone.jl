@testset verbose=true "Dummy Particles with `MarronePressureExtrapolation`" begin
    @testset "Compute Boundary Normal Vectors" begin
        @testset "2D `RectangularTank` Normals" begin
            particle_spacing = 1.0
            n_particles = 2
            n_layers = 1
            width = particle_spacing * n_particles
            height = particle_spacing * n_particles
            density = 257

            tank = RectangularTank(particle_spacing, (width, height), (width, height),
                                   density, n_layers=n_layers,
                                   faces=(true, true, true, false), normal=true)

            (; normals) = tank.boundary
            normals_reference = [[-0.5 -0.5 0.5 0.5 0.0 0.0 -0.5 0.5]
                                 [0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5]]

            @test normals == normals_reference
        end
        @testset "3D `RectangularTank` Normals" begin
            particle_spacing = 1.0
            n_particles = 2
            n_layers = 1
            tank_length = particle_spacing * n_particles
            density = 257

            tank = RectangularTank(particle_spacing,
                                   (tank_length, tank_length, tank_length),
                                   (tank_length, tank_length, tank_length),
                                   density, n_layers=n_layers,
                                   faces=(true, true, true, true, true, false), normal=true)

            (; normals) = tank.boundary
            normals_reference = [[-0.5 -0.5 -0.5 -0.5 0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5 0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.0 -0.5 -0.5 0.5 0.5 -0.5 -0.5 0.5 0.5]
                                 [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5 0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.0 -0.5 -0.5 0.5 0.5 -0.5 -0.5 0.5 0.5 -0.5 -0.5 0.5 0.5 0.0 0.0 0.0 0.0 -0.5 0.5 -0.5 0.5]
                                 [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5]]
            @test normals == normals_reference
        end

        @testset "2D `SphereShape` Normals" begin 
            return 0
        end
        @testset "3D `SphereShape` Normals" begin 
            return 0
        end 
    end
end
