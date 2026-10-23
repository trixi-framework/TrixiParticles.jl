@testset verbose=true "StructureMotionCalculator" begin
    # 4x3 grid of particles with spacing 1 in the region [0, 4] x [0, 3].
    particle_spacing = 1.0
    initial_condition = RectangularShape(particle_spacing, (4, 3), (0.0, 0.0), density=2.0)
    smoothing_length = 1.5
    system_ = TotalLagrangianSPHSystem(initial_condition;
                                       smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                       smoothing_length, young_modulus=1.0,
                                       poisson_ratio=0.4)
    semi = Semidiscretization(system_)
    system = semi.systems[1]

    # Point in the middle of the right end of the structure.
    position = (4.0, 1.5)

    @testset "Constructor" begin
        calculator = StructureMotionCalculator(system, semi, position)

        @test calculator.system_index == 1
        @test calculator.position == [4.0, 1.5]
        @test calculator.position isa SVector{2, Float64}
        @test calculator.quantity === identity

        # The compact support of the cubic spline kernel is `2 * smoothing_length = 3`,
        # so all particles with an initial distance < 3 from `position` are included.
        initial_coordinates = initial_condition.coordinates
        expected_particles = filter(eachparticle(system)) do particle
            return norm(initial_coordinates[:, particle] .- position) < 3
        end
        @test calculator.particles == expected_particles

        # The corrected weights reproduce constant and linear fields.
        @test isapprox(sum(calculator.weights), 1.0)
        @test isapprox(sum(calculator.weights[i] * initial_coordinates[:, particle]
                           for (i, particle) in enumerate(calculator.particles)),
                       collect(position))

        # `position` is on the right end of the structure, so the kernel support is
        # truncated, and the weights of the particles furthest to the left are negative.
        @test any(<(0), calculator.weights)
    end

    @testset "Position outside the kernel support" begin
        error_string = "`position` is not inside the kernel support"
        @test_throws error_string StructureMotionCalculator(system, semi, (100.0, 0.0))
    end

    @testset "Collinear neighbors" begin
        # A single row of particles cannot reconstruct the motion in y-direction.
        initial_condition_row = RectangularShape(particle_spacing, (4, 1), (0.0, 0.0),
                                                 density=2.0, place_on_shell=true)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        system_row_ = TotalLagrangianSPHSystem(initial_condition_row; smoothing_kernel,
                                               smoothing_length, young_modulus=1.0,
                                               poisson_ratio=0.4)
        semi_row = Semidiscretization(system_row_)
        system_row = semi_row.systems[1]

        error_string = "the particles in the kernel support of `position` must not all lie"
        @test_throws error_string StructureMotionCalculator(system_row, semi_row,
                                                            (3.0, 0.0))
    end

    @testset "Rigid translation" begin
        calculator = StructureMotionCalculator(system, semi, position)

        offset = SVector(0.3, -1.2)
        system.current_coordinates .= system.initial_coordinates .+ offset
        for particle in eachparticle(system)
            system.deformation_grad[:, :, particle] = I(2)
        end

        motion = TrixiParticles.structure_motion(calculator, system)

        # The corrected interpolation is exact for a constant displacement.
        @test isapprox(motion.displacement, offset)
        @test isapprox(motion.deformation_gradient, I(2))
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

    @testset "Rigid rotation by $(angle)° is exact at the surface" for angle in [17, -70, 170]
        calculator = StructureMotionCalculator(system, semi, position)

        angle_rad = deg2rad(angle)
        rotation_matrix = [cos(angle_rad) -sin(angle_rad); sin(angle_rad) cos(angle_rad)]
        system.current_coordinates .= rotation_matrix * system.initial_coordinates
        for particle in eachparticle(system)
            system.deformation_grad[:, :, particle] = rotation_matrix
        end

        motion = TrixiParticles.structure_motion(calculator, system)

        # Note that Shepard interpolation alone would be inaccurate for `position`
        # on the surface, where the kernel support is truncated.
        @test isapprox(motion.displacement,
                       rotation_matrix * collect(position) - collect(position))
        @test isapprox(motion.deformation_gradient, rotation_matrix)
        @test isapprox(rad2deg(motion.rotation), angle)
    end

    @testset "Pure stretch" begin
        calculator = StructureMotionCalculator(system, semi, position)

        for particle in eachparticle(system)
            system.deformation_grad[:, :, particle] = [2.0 0.0; 0.0 3.0]
        end

        # A symmetric positive definite deformation gradient contains no rotation
        @test isapprox(TrixiParticles.structure_motion(calculator, system).rotation, 0.0)
    end

    @testset "Affine deformation is reconstructed exactly at the surface" begin
        calculator = StructureMotionCalculator(system, semi, position)

        # Combine rotation, stretch, shear and translation.
        deformation_grad = [1.2 0.3; -0.1 0.9]
        offset = [0.3, -1.2]
        system.current_coordinates .= deformation_grad * system.initial_coordinates .+
                                      offset
        for particle in eachparticle(system)
            system.deformation_grad[:, :, particle] = deformation_grad
        end

        motion = TrixiParticles.structure_motion(calculator, system)

        # Note that Shepard interpolation alone would be inaccurate for `position`
        # on the surface, where the kernel support is truncated.
        expected_displacement = (deformation_grad - I) * collect(position) + offset
        @test isapprox(motion.displacement, expected_displacement)
        @test isapprox(motion.deformation_gradient, deformation_grad)
    end

    @testset "3D" begin
        initial_condition_3d = RectangularShape(particle_spacing, (4, 3, 3),
                                                (0.0, 0.0, 0.0), density=2.0)
        system_3d_ = TotalLagrangianSPHSystem(initial_condition_3d;
                                              smoothing_kernel=SchoenbergCubicSplineKernel{3}(),
                                              smoothing_length, young_modulus=1.0,
                                              poisson_ratio=0.4)
        semi_3d = Semidiscretization(system_3d_)
        system_3d = semi_3d.systems[1]

        # Place the probe point at a corner of the 3D block.
        calculator = StructureMotionCalculator(system_3d, semi_3d, (4.0, 1.5, 3.0))

        offset = SVector(0.3, -1.2, 0.7)
        system_3d.current_coordinates .= system_3d.initial_coordinates .+ offset
        for particle in eachparticle(system_3d)
            system_3d.deformation_grad[:, :, particle] = I(3)
        end

        motion = TrixiParticles.structure_motion(calculator, system_3d)

        @test isapprox(motion.displacement, offset)

        # There is no single rotation angle in 3D.
        @test isnan(motion.rotation)

        # As in 2D: combine rotation, stretch, shear and translation.
        deformation_grad = [1.2 0.3 0.0
                            -0.1 0.9 0.2
                            0.1 0.0 1.1]
        system_3d.current_coordinates .= deformation_grad * system_3d.initial_coordinates .+
                                         offset
        for particle in eachparticle(system_3d)
            system_3d.deformation_grad[:, :, particle] = deformation_grad
        end

        motion = TrixiParticles.structure_motion(calculator, system_3d)

        @test isapprox(motion.displacement,
                       (deformation_grad - I) * [4.0, 1.5, 3.0] + offset)
    end
end
