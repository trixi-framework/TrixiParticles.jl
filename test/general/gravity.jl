@trixi_testset "Gravity" begin
    # Construction preserves explicit parameters and the input precision used by defaults.
    gravity = NewtonianGravity(; gravitational_constant=1.0,
                               softening_length=0.1,
                               cutoff_radius=2.0)

    @test gravity.gravitational_constant == 1.0
    @test gravity.softening_length == 0.1
    @test gravity.cutoff_radius == 2.0

    gravity_float32 = NewtonianGravity(; gravitational_constant=1.0f0)

    @test gravity_float32.gravitational_constant === 1.0f0
    @test gravity_float32.softening_length === 0.0f0
    @test gravity_float32.cutoff_radius === Inf32

    # Reject values outside the physical domains and non-real inputs at the API boundary.
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=-1.0)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=Inf)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=NaN)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=1.0,
                                                softening_length=-0.1)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=1.0,
                                                softening_length=NaN)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=1.0,
                                                cutoff_radius=0.0)
    @test_throws ArgumentError NewtonianGravity(; gravitational_constant=1.0,
                                                cutoff_radius=NaN)
    @test_throws TypeError NewtonianGravity(; gravitational_constant=1 + 0im)
    @test_throws TypeError NewtonianGravity(; gravitational_constant=1.0,
                                            softening_length=1 + 0im)
    @test_throws TypeError NewtonianGravity(; gravitational_constant=1.0,
                                            cutoff_radius=missing)

    # The unsoftened law follows G*m/r^3 and is undefined at exact coincidence.
    unsoftened_gravity = NewtonianGravity(; gravitational_constant=1.0)
    @test TrixiParticles.gravity_acceleration(unsoftened_gravity,
                                              SVector(2.0, 0.0),
                                              2.0, 3.0) == SVector(-0.75, 0.0)
    @test_throws DomainError TrixiParticles.gravity_acceleration(unsoftened_gravity,
                                                                 SVector(0.0, 0.0),
                                                                 0.0, 3.0)

    # Plummer softening regularizes the denominator and gives zero force at coincidence.
    softened_gravity = NewtonianGravity(; gravitational_constant=1.0,
                                        softening_length=1.0)
    acceleration = TrixiParticles.gravity_acceleration(softened_gravity,
                                                       SVector(2.0, 0.0),
                                                       2.0, 3.0)

    @test acceleration ≈ SVector(-6 / (5 * sqrt(5)), 0.0)
    @test TrixiParticles.gravity_acceleration(softened_gravity,
                                              SVector(0.0, 0.0),
                                              0.0, 3.0) == SVector(0.0, 0.0)

    # The cutoff includes particles exactly on its boundary and excludes particles beyond it.
    cutoff_gravity = NewtonianGravity(; gravitational_constant=1.0,
                                      cutoff_radius=2.0)
    @test TrixiParticles.gravity_acceleration(cutoff_gravity,
                                              SVector(2.0, 0.0),
                                              2.0, 3.0) == SVector(-0.75, 0.0)
    @test TrixiParticles.gravity_acceleration(cutoff_gravity,
                                              SVector(3.0, 0.0),
                                              3.0, 3.0) == SVector(0.0, 0.0)

    # Softening and cutoff remain active together rather than selecting only one option.
    softened_cutoff_gravity = NewtonianGravity(; gravitational_constant=1.0,
                                               softening_length=1.0,
                                               cutoff_radius=2.0)
    acceleration = TrixiParticles.gravity_acceleration(softened_cutoff_gravity,
                                                       SVector(2.0, 0.0),
                                                       2.0, 3.0)

    @test acceleration ≈ SVector(-6 / (5 * sqrt(5)), 0.0)
    @test TrixiParticles.gravity_acceleration(softened_cutoff_gravity,
                                              SVector(3.0, 0.0),
                                              3.0, 3.0) == SVector(0.0, 0.0)
end
