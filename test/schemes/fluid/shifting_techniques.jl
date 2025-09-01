@testset verbose=true "Shifting Techniques" begin
    @testset "Constructors" begin
        @test_nowarn TransportVelocityAdami(background_pressure=1.0)
        @test_nowarn ParticleShiftingTechniqueSun2017()
        @test_nowarn pst = ParticleShiftingTechniqueSun2017(v_max_factor=1.2)
        @test pst.v_factor == 1.2
        @test_nowarn ConsistentShiftingSun2019()
        @test_nowarn pst = ConsistentShiftingSun2019(sound_speed_factor=0.2)
        @test pst.v_factor == 0.2

        # Can't use both `v_max_factor` and `sound_speed_factor`
        @test_throws ArgumentError ParticleShiftingTechnique(v_max_factor=1.0,
                                                             sound_speed_factor=0.5)
        # At least one of `v_max_factor` and `sound_speed_factor` must be positive
        @test_throws ArgumentError ParticleShiftingTechnique(v_max_factor=0.0,
                                                             sound_speed_factor=0.0)
        # Can't update every stage if not integrating shifting velocity
        @test_throws ArgumentError ParticleShiftingTechnique(integrate_shifting_velocity=false,
                                                             update_everystage=true)
        # Can't modify continuity equation if not integrating shifting velocity
        @test_throws ArgumentError ParticleShiftingTechnique(integrate_shifting_velocity=false,
                                                             modify_continuity_equation=true)
        # Can't modify momentum equation if not integrating shifting velocity
        @test_throws ArgumentError ParticleShiftingTechnique(integrate_shifting_velocity=false,
                                                             modify_momentum_equation=true)
        # Can't use second continuity equation term if not modifying continuity equation
        @test_throws ArgumentError ParticleShiftingTechnique(integrate_shifting_velocity=true,
                                                             modify_continuity_equation=false,
                                                             second_continuity_equation_term=true)
    end
end
