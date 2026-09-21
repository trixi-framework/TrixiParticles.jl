@testset verbose=true "StructureMotionCalculator" begin
    # 4x3 grid of particles with spacing 1 in the region [0, 3] x [0, 2]
    particle_spacing = 1.0
    initial_condition = RectangularShape(particle_spacing, (4, 3), (0.0, 0.0),
                                         density=2.0, place_on_shell=true)
    smoothing_length = 1.5
    system_ = TotalLagrangianSPHSystem(initial_condition;
                                       smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                       smoothing_length, young_modulus=1.0,
                                       poisson_ratio=0.4)
    semi = Semidiscretization(system_)
    system = semi.systems[1]

    # Point in the middle of the right end of the structure
    position = (3.0, 1.0)

    @testset "Constructor" begin
        calculator = StructureMotionCalculator(system, semi, position)

        @test calculator.system_index == 1
        @test calculator.position == [3.0, 1.0]
        @test calculator.position isa SVector{2, Float64}
        @test calculator.quantity === identity

        # The compact support of the cubic spline kernel is `2 * smoothing_length = 3`,
        # so all particles with an initial distance < 3 from `position` are included.
        # Particles exactly on the boundary of the compact support have a zero weight
        # and are skipped.
        initial_coordinates = initial_condition.coordinates
        expected_particles = filter(eachparticle(system)) do particle
            return norm(initial_coordinates[:, particle] .- position) < 3
        end
        @test calculator.particles == expected_particles

        # Weights are normalized (Shepard correction)
        @test isapprox(sum(calculator.weights), 1.0)
        @test all(>(0), calculator.weights)

        # Particles closer to `position` have larger weights
        distances = [norm(initial_coordinates[:, particle] .- position)
                     for particle in calculator.particles]
        @test issorted(calculator.weights[sortperm(distances)], rev=true)
    end

    @testset "Position outside the kernel support" begin
        error_string = "`position` is not inside the kernel support of any particle " *
                       "of the system in the initial configuration"
        @test_throws ArgumentError(error_string) StructureMotionCalculator(system, semi,
                                                                           (100.0, 0.0))
    end

    @testset "Rigid translation" begin
        calculator = StructureMotionCalculator(system, semi, position)

        offset = SVector(0.3, -1.2)
        system.current_coordinates .= system.initial_coordinates .+ offset
        for particle in eachparticle(system)
            system.deformation_grad[:, :, particle] = [1.0 0.0; 0.0 1.0]
        end

        motion = TrixiParticles.structure_motion(calculator, system)

        # The Shepard-corrected interpolation is exact for a constant displacement
        @test isapprox(motion.displacement, offset)
        @test isapprox(motion.deformation_gradient, [1.0 0.0; 0.0 1.0])
        @test isapprox(motion.rotation, 0.0)

        @test calculator(system, nothing, nothing, nothing, nothing, semi, 0.0) == motion
    end

    @testset "Other systems are ignored" begin
        semi_2 = Semidiscretization(system_, system_)
        calculator = StructureMotionCalculator(semi_2.systems[2], semi_2, position)

        @test calculator.system_index == 2
        @test isnothing(calculator(semi_2.systems[1], nothing, nothing, nothing, nothing,
                                   semi_2, 0.0))
        @test !isnothing(calculator(semi_2.systems[2], nothing, nothing, nothing, nothing,
                                    semi_2, 0.0))
    end

    @testset "Rigid rotation by $(round(Int, rad2deg(angle)))°" for angle in
                                                                    [0.3, -1.2, 3.0]
        calculator = StructureMotionCalculator(system, semi, position)

        rotation_matrix = [cos(angle) -sin(angle); sin(angle) cos(angle)]
        system.current_coordinates .= rotation_matrix * system.initial_coordinates
        for particle in eachparticle(system)
            system.deformation_grad[:, :, particle] = rotation_matrix
        end

        motion = TrixiParticles.structure_motion(calculator, system)

        @test isapprox(motion.deformation_gradient, rotation_matrix)
        @test isapprox(motion.rotation, angle)
    end

    @testset "Pure stretch" begin
        calculator = StructureMotionCalculator(system, semi, position)

        for particle in eachparticle(system)
            system.deformation_grad[:, :, particle] = [2.0 0.0; 0.0 3.0]
        end

        # A symmetric positive definite deformation gradient contains no rotation
        @test isapprox(TrixiParticles.structure_motion(calculator, system).rotation, 0.0)
    end

    @testset "Keyword `quantity`" begin
        deflection_x = StructureMotionCalculator(system, semi, position,
                                                 quantity=motion -> motion.displacement[1])
        deflection_y = StructureMotionCalculator(system, semi, position,
                                                 quantity=motion -> motion.displacement[2])

        offset = SVector(0.3, -1.2)
        system.current_coordinates .= system.initial_coordinates .+ offset

        @test isapprox(deflection_x(system, nothing, nothing, nothing, nothing, semi, 0.0),
                       offset[1])
        @test isapprox(deflection_y(system, nothing, nothing, nothing, nothing, semi, 0.0),
                       offset[2])
    end

    @testset "3D" begin
        initial_condition_3d = RectangularShape(particle_spacing, (4, 3, 3),
                                                (0.0, 0.0, 0.0),
                                                density=2.0, place_on_shell=true)
        system_3d_ = TotalLagrangianSPHSystem(initial_condition_3d;
                                              smoothing_kernel=SchoenbergCubicSplineKernel{3}(),
                                              smoothing_length, young_modulus=1.0,
                                              poisson_ratio=0.4)
        semi_3d = Semidiscretization(system_3d_)
        system_3d = semi_3d.systems[1]

        calculator = StructureMotionCalculator(system_3d, semi_3d, (3.0, 1.0, 1.0))

        offset = SVector(0.3, -1.2, 0.7)
        system_3d.current_coordinates .= system_3d.initial_coordinates .+ offset
        for particle in eachparticle(system_3d)
            system_3d.deformation_grad[:, :, particle] = [1.0 0.0 0.0
                                                          0.0 1.0 0.0
                                                          0.0 0.0 1.0]
        end

        motion = TrixiParticles.structure_motion(calculator, system_3d)

        @test isapprox(motion.displacement, offset)

        # There is no single rotation angle in 3D
        @test isnan(motion.rotation)
    end
end
