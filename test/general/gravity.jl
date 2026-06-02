@trixi_testset "Gravity" begin
    default_gravity = NewtonianGravity()

    @test DEFAULT_GRAVITATIONAL_CONSTANT === 1.0
    @test default_gravity.gravitational_constant === DEFAULT_GRAVITATIONAL_CONSTANT
    @test default_gravity.softening isa NoSoftening
    @test default_gravity.cutoff_radius === Inf

    gravity = NewtonianGravity(; gravitational_constant=1.0,
                               softening=PlummerSoftening(0.1),
                               cutoff_radius=2.0)

    @test gravity isa TrixiParticles.AbstractGravityModel
    @test gravity.gravitational_constant == 1.0
    @test gravity.softening isa PlummerSoftening
    @test gravity.softening.softening_length == 0.1
    @test gravity.cutoff_radius == 2.0

    gravity_float32 = NewtonianGravity(; gravitational_constant=1.0f0)

    @test gravity_float32.gravitational_constant === 1.0f0
    @test gravity_float32.softening isa NoSoftening
    @test gravity_float32.cutoff_radius === Inf32

    @test NoSoftening()(2.0) == 0.125
    @test NoSoftening()(SVector(2.0, 0.0)) == SVector(0.25, 0.0)
    @test NoSoftening()(SVector(2.0, 0.0), 2.0) == SVector(0.25, 0.0)
    @test PlummerSoftening(1.0)(2.0) ≈ inv(5 * sqrt(5))
    @test PlummerSoftening(1.0)(SVector(2.0, 0.0)) ≈
          SVector(2 / (5 * sqrt(5)), 0.0)
    @test PlummerSoftening(; softening_length=1.0)(SVector(2.0, 0.0), 2.0) ≈
          SVector(2 / (5 * sqrt(5)), 0.0)

    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=-1.0)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=Inf)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=NaN)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=1.0,
                                                softening=nothing)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=1.0,
                                                softening=0.1)
    @test_throws ArgumentError PlummerSoftening(-0.1)
    @test_throws ArgumentError PlummerSoftening(Inf)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=1.0,
                                                cutoff_radius=0.0)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=1.0,
                                                cutoff_radius=NaN)

    struct GravitySystem <: TrixiParticles.AbstractSystem{2} end
    system = GravitySystem()
    neighbor_system = GravitySystem()
    mock_coordinates = [0.0 2.0;
                        0.0 0.0]
    mock_velocity = [1.0 3.0;
                     2.0 4.0]
    mock_mass = [2.0, 3.0]

    @test !(:current_velocity in names(TrixiParticles))
    @test TrixiParticles.gravity_model(system) === nothing
    @test TrixiParticles.gravity_model(system, neighbor_system) === nothing
    @test_throws MethodError TrixiParticles.gravitational_mass(system, 1)
    @test TrixiParticles.current_position(mock_coordinates, system, 2) ==
          SVector(2.0, 0.0)
    @test TrixiParticles.current_velocity(mock_velocity, system, 2) ==
          SVector(3.0, 4.0)

    TrixiParticles.gravitational_mass(::GravitySystem, particle) = mock_mass[particle]
    dv = zeros(2, 2)
    returned = gravity_acceleration!(dv, NewtonianGravity(), system, neighbor_system,
                                     1, 2, SVector(-2.0, 0.0), 2.0)

    @test returned === dv
    @test dv[:, 1] ≈ [0.75, 0.0]
    @test dv[:, 2] == [0.0, 0.0]

    dv_particle = Ref(SVector(1.0, -2.0))
    returned = TrixiParticles.gravity_interaction!(dv_particle, nothing,
                                                   system, neighbor_system,
                                                   1, 2, SVector(1.0, 0.0),
                                                   1.0, 1.0, 2.0)

    @test returned === dv_particle
    @test dv_particle[] == SVector(1.0, -2.0)

    struct UnknownGravity <: TrixiParticles.AbstractGravityModel end

    @test_throws MethodError TrixiParticles.gravity_interaction!(dv_particle,
                                                                 UnknownGravity(),
                                                                 system, neighbor_system,
                                                                 1, 2,
                                                                 SVector(1.0, 0.0),
                                                                 1.0, 1.0, 2.0)

    unsoftened_gravity = NewtonianGravity(; gravitational_constant=1.0)
    dv_particle = Ref(SVector(1.0, -2.0))
    returned = TrixiParticles.gravity_interaction!(dv_particle, unsoftened_gravity,
                                                   system, neighbor_system,
                                                   1, 2, SVector(2.0, 0.0),
                                                   2.0, 1.0, 3.0)

    @test returned === dv_particle
    @test dv_particle[] == SVector(0.25, -2.0)
    @test TrixiParticles.gravity_acceleration(unsoftened_gravity,
                                              SVector(0.0, 0.0),
                                              0.0, 3.0) == SVector(0.0, 0.0)

    softened_gravity = NewtonianGravity(; gravitational_constant=1.0,
                                        softening=PlummerSoftening(1.0))
    acceleration = TrixiParticles.gravity_acceleration(softened_gravity,
                                                       SVector(2.0, 0.0),
                                                       2.0, 3.0)

    @test acceleration ≈ SVector(-6 / (5 * sqrt(5)), 0.0)

    cutoff_gravity = NewtonianGravity(; gravitational_constant=1.0,
                                      cutoff_radius=2.0)
    @test TrixiParticles.gravity_acceleration(cutoff_gravity,
                                              SVector(2.0, 0.0),
                                              2.0, 3.0) == SVector(-0.75, 0.0)
    @test TrixiParticles.gravity_acceleration(cutoff_gravity,
                                              SVector(3.0, 0.0),
                                              3.0, 3.0) == SVector(0.0, 0.0)

    softened_cutoff_gravity = NewtonianGravity(; gravitational_constant=1.0,
                                               softening=PlummerSoftening(1.0),
                                               cutoff_radius=2.0)
    acceleration = TrixiParticles.gravity_acceleration(softened_cutoff_gravity,
                                                       SVector(2.0, 0.0),
                                                       2.0, 3.0)

    @test acceleration ≈ SVector(-6 / (5 * sqrt(5)), 0.0)
    @test TrixiParticles.gravity_acceleration(softened_cutoff_gravity,
                                              SVector(3.0, 0.0),
                                              3.0, 3.0) == SVector(0.0, 0.0)
end
