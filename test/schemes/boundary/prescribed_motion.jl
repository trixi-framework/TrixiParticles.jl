# Note that more tests for `PrescribedMotion` are in `test/systems/boundary_system.jl`.
@testset verbose=true "PrescribedMotion" begin
    @testset "OscillatingMotion2D" begin
        @testset "Simple Rotation" begin
            frequency = 1.0
            translation_vector = SVector(0.0, 0.0)
            rotation_angle = pi
            rotation_center = SVector(0.0, 0.0)

            motion = OscillatingMotion2D(; frequency, translation_vector, rotation_angle,
                                         rotation_center)
            movement_function = motion.movement_function

            @test isapprox(movement_function([1.0, 0.0], 0.0), [1.0, 0.0], atol=eps())
            # Initial position after one phase
            @test isapprox(movement_function([1.0, 0.0], 1.0), [1.0, 0.0], atol=10eps())
            # Initial position after one half phase
            @test isapprox(movement_function([1.0, 0.0], 0.5), [1.0, 0.0], atol=10eps())
            # Half rotation after one quarter phase (rotation angle is one half rotation)
            @test isapprox(movement_function([1.0, 0.0], 0.25), [-1.0, 0.0], atol=10eps())
            # Quarter rotation (0.5 * rotation_angle = 90 degrees)
            @test isapprox(movement_function([1.0, 0.0], asin(0.5) / 2pi), [0.0, 1.0],
                           atol=10eps())
        end

        @testset "Simple Translation" begin
            frequency = 1.0
            translation_vector = SVector(1.0, 0.0)
            rotation_angle = 0.0
            rotation_center = SVector(0.0, 0.0)

            motion = OscillatingMotion2D(; frequency, translation_vector, rotation_angle,
                                         rotation_center)
            movement_function = motion.movement_function

            @test isapprox(movement_function([0.0, 0.0], 0.0), [0.0, 0.0], atol=eps())
            @test isapprox(movement_function([0.0, 0.0], 0.25), [1.0, 0.0], atol=10eps())
            @test isapprox(movement_function([0.0, 0.0], 0.5), [0.0, 0.0], atol=10eps())
            @test isapprox(movement_function([0.0, 0.0], 0.75), [-1.0, 0.0], atol=10eps())
            @test isapprox(movement_function([0.0, 0.0], 1.0), [0.0, 0.0], atol=10eps())
        end

        @testset "Simple Translation tspan" begin
            frequency = 1.0
            translation_vector = SVector(1.0, 0.0)
            rotation_angle = 0.0
            rotation_center = SVector(0.0, 0.0)
            tspan = (0.4, 1.4)

            motion = OscillatingMotion2D(; frequency, translation_vector, rotation_angle,
                                         rotation_center, tspan)
            movement_function = motion.movement_function

            @test isapprox(movement_function([0.0, 0.0], 0.4), [0.0, 0.0], atol=eps())
            @test isapprox(movement_function([0.0, 0.0], 0.65), [1.0, 0.0], atol=10eps())
            @test isapprox(movement_function([0.0, 0.0], 0.9), [0.0, 0.0], atol=10eps())
            @test isapprox(movement_function([0.0, 0.0], 1.15), [-1.0, 0.0], atol=10eps())
            @test isapprox(movement_function([0.0, 0.0], 1.4), [0.0, 0.0], atol=10eps())
        end

        @testset "Combined Rotation and Translation" begin
            frequency = 1.0
            translation_vector = SVector(1.0, 0.0)
            rotation_angle = pi / 2
            rotation_center = SVector(0.0, 0.0)

            motion = OscillatingMotion2D(; frequency, translation_vector, rotation_angle,
                                         rotation_center)
            movement_function = motion.movement_function

            @test isapprox(movement_function([1.0, 0.0], 0.0), [1.0, 0.0], atol=eps())
            @test isapprox(movement_function([1.0, 0.0], 0.25), [1.0, 1.0], atol=10eps())
            @test isapprox(movement_function([1.0, 0.0], 0.5), [1.0, 0.0], atol=10eps())
            @test isapprox(movement_function([1.0, 0.0], 0.75), [-1.0, -1.0], atol=10eps())
            @test isapprox(movement_function([1.0, 0.0], 1.0), [1.0, 0.0], atol=10eps())
        end

        @testset "Phase Offset" begin
            frequency = 1.0
            translation_vector = SVector(1.0, 0.0)
            rotation_angle = pi / 2
            rotation_center = SVector(0.0, 0.0)
            rotation_phase_offset = 0.25

            motion = OscillatingMotion2D(; frequency, translation_vector, rotation_angle,
                                         rotation_center, rotation_phase_offset)
            movement_function = motion.movement_function

            @test isapprox(movement_function([1.0, 0.0], 0.0), [0.0, -1.0], atol=eps())
            @test isapprox(movement_function([1.0, 0.0], 0.25), [2.0, 0.0], atol=10eps())
            @test isapprox(movement_function([1.0, 0.0], 0.5), [0.0, 1.0], atol=10eps())
            @test isapprox(movement_function([1.0, 0.0], 0.75), [0.0, 0.0], atol=10eps())
            @test isapprox(movement_function([1.0, 0.0], 1.0), [0.0, -1.0], atol=10eps())
        end

        @testset "Ramp-up" begin
            frequency = 1.0
            translation_vector = SVector(1.0, 0.0)
            rotation_angle = 0.0
            rotation_center = SVector(0.0, 0.0)
            ramp_up_tspan = (0.0, 1.0)

            motion = OscillatingMotion2D(; frequency, translation_vector, rotation_angle,
                                         rotation_center, ramp_up_tspan)
            movement_function = motion.movement_function

            @test isapprox(movement_function([0.0, 0.0], 0.0), [0.0, 0.0], atol=eps())
            # During ramp-up, amplitude is reduced
            @test isapprox(movement_function([0.0, 0.0], 0.25), [0.15625, 0.0],
                           atol=10eps())
            @test isapprox(movement_function([0.0, 0.0], 0.5), [0.0, 0.0], atol=10eps())
            # Less reduction at later times during ramp-up
            @test isapprox(movement_function([0.0, 0.0], 0.75), [-0.84375, 0.0],
                           atol=10eps())
            @test isapprox(movement_function([0.0, 0.0], 1.0), [0.0, 0.0], atol=10eps())
            # After ramp-up, full amplitude
            @test isapprox(movement_function([0.0, 0.0], 1.25), [1.0, 0.0], atol=10eps())
        end

        @testset "tspan" begin
            frequency = 1.0
            translation_vector = SVector(1.0, 0.0)
            rotation_angle = 0.0
            rotation_center = SVector(0.0, 0.0)
            tspan = (0.5, 1.5)

            motion = OscillatingMotion2D(; frequency, translation_vector, rotation_angle,
                                         rotation_center, tspan)
            ismoving = motion.is_moving

            @test !ismoving(-0.5)
            @test !ismoving(0.0)
            @test !ismoving(0.25)
            @test ismoving(0.5)
            @test ismoving(1.0)
            @test ismoving(1.5)
            @test !ismoving(1.75)
        end
    end
end
