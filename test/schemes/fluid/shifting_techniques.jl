@testset verbose=true "Shifting Techniques" begin
    @testset "Constructors" begin
        @test_nowarn TransportVelocityAdami(background_pressure=1.0)
        @test_nowarn ParticleShiftingTechniqueSun2017()
        pst = @test_nowarn ParticleShiftingTechniqueSun2017(v_max_factor=1.2)
        @test pst.v_factor == 1.2
        @test_nowarn ConsistentShiftingSun2019()
        pst = @test_nowarn ConsistentShiftingSun2019(sound_speed_factor=0.2)
        @test pst.v_factor == 0.2
        treatment = FreeSurfaceTangentialShifting()
        pst = @test_nowarn ConsistentShiftingSun2019(; free_surface_treatment=treatment)
        @test pst.free_surface_treatment === treatment
        @test_throws ArgumentError ParticleShiftingTechnique(free_surface_treatment=:invalid)
        css = SurfaceTensionMomentumMorris(; surface_tension_coefficient=1.0)
        @test_throws ArgumentError TrixiParticles.validate_free_surface_shifting(pst,
                                                                                 nothing,
                                                                                 css)
        @test_throws ArgumentError TrixiParticles.validate_free_surface_shifting(pst,
                                                                                 ColorfieldSurfaceNormal(),
                                                                                 nothing)
        @test_nowarn TrixiParticles.validate_free_surface_shifting(pst,
                                                                   ColorfieldSurfaceNormal(),
                                                                   css)
        @test_nowarn TrixiParticles.validate_free_surface_shifting(pst,
                                                                   CorrectedCSFSurfaceNormal(),
                                                                   SurfaceTensionMorris())

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
                                                             momentum_equation_term=MomentumEquationTermSun2019())
        # Can't use second continuity equation term if not modifying continuity equation
        @test_throws ArgumentError ParticleShiftingTechnique(integrate_shifting_velocity=true,
                                                             modify_continuity_equation=false,
                                                             second_continuity_equation_term=ContinuityEquationTermSun2019())
    end

    @testset "Tangential free-surface projection" begin
        shifting_velocity = [3.0, 4.0]
        normal = [1.0, 0.0]

        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity, normal,
                                                          0.0) ≈ [3.0, 4.0]
        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity, normal,
                                                          0.5) ≈ [1.5, 4.0]
        tangential = TrixiParticles.tangential_shifting_velocity(shifting_velocity,
                                                                 normal, 1.0)
        @test tangential ≈ [0.0, 4.0]
        @test dot(tangential, normal) ≈ 0.0
        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity,
                                                          zeros(2), 1.0) ==
              shifting_velocity

        particle_spacing = 0.1
        initial_condition = RectangularShape(particle_spacing, (2, 2), (0.0, 0.0);
                                             density=1.0)
        surface_tension = SurfaceTensionMomentumMorris(;
                                                       surface_tension_coefficient=1.0)
        surface_normal_method = ColorfieldSurfaceNormal()
        shifting_technique = ConsistentShiftingSun2019(;
                                                       free_surface_treatment=FreeSurfaceTangentialShifting())
        system = WeaklyCompressibleSPHSystem(initial_condition;
                                             smoothing_kernel=WendlandC2Kernel{2}(),
                                             smoothing_length=1.4 * particle_spacing,
                                             density_calculator=ContinuityDensity(),
                                             state_equation=StateEquationCole(;
                                                                              sound_speed=10.0,
                                                                              reference_density=1.0,
                                                                              exponent=7),
                                             surface_tension, surface_normal_method,
                                             shifting_technique,
                                             reference_particle_spacing=particle_spacing)
        system.cache.delta_v .= 0
        system.cache.surface_normal .= 0
        system.cache.interface_activity .= 0
        system.cache.delta_v[:, 1] .= shifting_velocity
        system.cache.surface_normal[:, 1] .= normal
        system.cache.interface_activity[1] = 1

        TrixiParticles.modify_shifting_with_surface_normal!(system,
                                                            FreeSurfaceTangentialShifting(),
                                                            DummySemidiscretization())
        @test system.cache.delta_v[:, 1] ≈ [0.0, 4.0]
        @test all(iszero, system.cache.delta_v[:, 2:end])
    end
end
