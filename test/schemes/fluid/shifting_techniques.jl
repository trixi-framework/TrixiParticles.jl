@testset verbose=true "Shifting Techniques" begin
    @testset "Constructors" begin
        @test_nowarn TransportVelocityAdami(background_pressure=1.0)
        @test_nowarn ParticleShiftingTechniqueSun2017()
        pst = @test_nowarn ParticleShiftingTechniqueSun2017(v_max_factor=1.2)
        @test pst.v_factor == 1.2
        @test_nowarn ConsistentShiftingSun2019()
        pst = @test_nowarn ConsistentShiftingSun2019(sound_speed_factor=0.2)
        @test pst.v_factor == 0.2
        @test isnothing(pst.free_surface_treatment)

        treatment = FreeSurfaceTangentialShifting()
        pst = @test_nowarn ConsistentShiftingSun2019(; free_surface_treatment=treatment)
        @test pst.free_surface_treatment === treatment
        callback_pst = @test_nowarn ParticleShiftingTechniqueSun2017(;
                                                                     free_surface_treatment=treatment)
        @test callback_pst.free_surface_treatment === treatment
        @test_throws ArgumentError ParticleShiftingTechnique(free_surface_treatment=:invalid)

        css = SurfaceTensionMomentumMorris(; surface_tension_coefficient=1.0)
        morris = SurfaceTensionMorris(; surface_tension_coefficient=1.0)
        @test_throws ArgumentError TrixiParticles.validate_free_surface_shifting(pst,
                                                                                 nothing,
                                                                                 css)
        @test_throws ArgumentError TrixiParticles.validate_free_surface_shifting(pst,
                                                                                 ColorfieldSurfaceNormal(),
                                                                                 nothing)
        @test_throws ArgumentError TrixiParticles.validate_free_surface_shifting(pst,
                                                                                 ColorfieldSurfaceNormal(),
                                                                                 SurfaceTensionAkinci())
        @test_throws ArgumentError TrixiParticles.validate_free_surface_shifting(pst,
                                                                                 CorrectedCSFSurfaceNormal(),
                                                                                 css)
        @test_nowarn TrixiParticles.validate_free_surface_shifting(pst,
                                                                   ColorfieldSurfaceNormal(),
                                                                   css)
        @test_nowarn TrixiParticles.validate_free_surface_shifting(pst,
                                                                   ColorfieldSurfaceNormal(),
                                                                   morris)
        @test_nowarn TrixiParticles.validate_free_surface_shifting(pst,
                                                                   CorrectedCSFSurfaceNormal(),
                                                                   morris)

        system_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(system_data, pst)
        @test system_data["shifting_technique"]["free_surface_treatment"] ==
              "FreeSurfaceTangentialShifting"
        default_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(default_data, ConsistentShiftingSun2019())
        @test isnothing(default_data["shifting_technique"]["free_surface_treatment"])

        particle_spacing = 0.1
        initial_condition = RectangularShape(particle_spacing, (2, 2), (0.0, 0.0);
                                             density=1.0)
        smoothing_kernel = WendlandC2Kernel{2}()
        normal_method = ColorfieldSurfaceNormal()
        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length=1.4particle_spacing,
                                                               density_calculator=ContinuityDensity(),
                                                               state_equation=StateEquationCole(;
                                                                                                sound_speed=10.0,
                                                                                                reference_density=1.0,
                                                                                                exponent=7),
                                                               surface_normal_method=normal_method,
                                                               shifting_technique=pst,
                                                               reference_particle_spacing=particle_spacing)
        @test_throws ArgumentError EntropicallyDampedSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length=1.4particle_spacing,
                                                               sound_speed=10.0,
                                                               density_calculator=ContinuityDensity(),
                                                               surface_normal_method=normal_method,
                                                               shifting_technique=pst,
                                                               reference_particle_spacing=particle_spacing)
        @test_nowarn EntropicallyDampedSPHSystem(initial_condition;
                                                 smoothing_kernel,
                                                 smoothing_length=1.4particle_spacing,
                                                 sound_speed=10.0,
                                                 density_calculator=ContinuityDensity(),
                                                 surface_tension=css,
                                                 surface_normal_method=normal_method,
                                                 shifting_technique=pst,
                                                 reference_particle_spacing=particle_spacing)

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
        shifting_velocity = SVector(3.0, 4.0)
        normal = SVector(1.0, 0.0)

        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity, normal,
                                                          0.0) == shifting_velocity
        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity, normal,
                                                          0.5) ≈ SVector(1.5, 4.0)
        tangential = TrixiParticles.tangential_shifting_velocity(shifting_velocity,
                                                                 normal, 1.0)
        @test tangential ≈ SVector(0.0, 4.0)
        @test dot(tangential, normal) ≈ 0.0
        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity,
                                                          2normal, 1.0) ≈ tangential
        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity,
                                                          normal, -1.0) == shifting_velocity
        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity,
                                                          normal, 2.0) ≈ tangential
        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity,
                                                          zero(normal), 1.0) ==
              shifting_velocity
        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity,
                                                          SVector(NaN, 0.0), 1.0) ==
              shifting_velocity
        @test TrixiParticles.tangential_shifting_velocity(shifting_velocity, normal,
                                                          NaN) == shifting_velocity

        particle_spacing = 0.1
        initial_condition = RectangularShape(particle_spacing, (2, 2), (0.0, 0.0);
                                             density=1.0)
        surface_tension = SurfaceTensionMomentumMorris(;
                                                       surface_tension_coefficient=1.0)
        surface_normal_method = ColorfieldSurfaceNormal(; normal_smoothing=true)
        treatment = FreeSurfaceTangentialShifting()
        shifting_technique = ConsistentShiftingSun2019(;
                                                       free_surface_treatment=treatment)
        system = WeaklyCompressibleSPHSystem(initial_condition;
                                             smoothing_kernel=WendlandC2Kernel{2}(),
                                             smoothing_length=1.4particle_spacing,
                                             density_calculator=ContinuityDensity(),
                                             state_equation=StateEquationCole(;
                                                                              sound_speed=10.0,
                                                                              reference_density=1.0,
                                                                              exponent=7),
                                             surface_tension, surface_normal_method,
                                             shifting_technique,
                                             reference_particle_spacing=particle_spacing)
        system.cache.delta_v .= reshape([3.0, 4.0], 2, 1)
        system.cache.surface_normal .= reshape([1.0, 0.0], 2, 1)
        system.cache.smoothed_surface_normal .= reshape([0.0, 1.0], 2, 1)
        system.cache.interface_activity .= [1.0, 0.5, 0.0, 1.0]
        system.cache.surface_normal[:, 4] .= 0

        TrixiParticles.modify_shifting_with_surface_normal!(system, treatment,
                                                            DummySemidiscretization())
        @test system.cache.delta_v[:, 1] ≈ [0.0, 4.0]
        @test system.cache.delta_v[:, 2] ≈ [1.5, 4.0]
        @test system.cache.delta_v[:, 3] ≈ [3.0, 4.0]
        @test system.cache.delta_v[:, 4] ≈ [3.0, 4.0]

        system.cache.delta_v .= reshape([3.0, 4.0], 2, 1)
        TrixiParticles.modify_shifting_with_surface_normal!(system, nothing,
                                                            DummySemidiscretization())
        @test all(particle -> system.cache.delta_v[:, particle] ≈ [3.0, 4.0],
                  eachparticle(system))
    end

    @testset "Integrated tangential shifting" begin
        function shifting_system(free_surface_treatment)
            particle_spacing = 0.1
            initial_condition = RectangularShape(particle_spacing, (7, 7), (0.0, 0.0);
                                                 density=1000.0)
            surface_tension = SurfaceTensionMomentumMorris(;
                                                           surface_tension_coefficient=0.072)
            surface_normal_method = ColorfieldSurfaceNormal(;
                                                            ideal_density_threshold=0.9,
                                                            normal_smoothing=true)
            shifting_technique = ConsistentShiftingSun2019(;
                                                           free_surface_treatment)
            system = WeaklyCompressibleSPHSystem(initial_condition;
                                                 smoothing_kernel=WendlandC2Kernel{2}(),
                                                 smoothing_length=1.4particle_spacing,
                                                 density_calculator=ContinuityDensity(),
                                                 state_equation=StateEquationCole(;
                                                                                  sound_speed=10.0,
                                                                                  reference_density=1000.0,
                                                                                  exponent=7),
                                                 surface_tension, surface_normal_method,
                                                 shifting_technique,
                                                 reference_particle_spacing=particle_spacing)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            TrixiParticles.update_systems_and_nhs(ode.u0.x..., semi, 0.0)
            return system
        end

        untreated = shifting_system(nothing)
        treated = shifting_system(FreeSurfaceTangentialShifting())
        @test treated.cache.surface_normal ≈ untreated.cache.surface_normal
        @test treated.cache.interface_activity ≈ untreated.cache.interface_activity

        expected = similar(untreated.cache.delta_v)
        for particle in eachparticle(treated)
            shifting_velocity = TrixiParticles.extract_svector(untreated.cache.delta_v,
                                                               untreated, particle)
            normal = TrixiParticles.surface_normal(treated, particle)
            activity = treated.cache.interface_activity[particle]
            expected[:,
                     particle] = TrixiParticles.tangential_shifting_velocity(shifting_velocity,
                                                                             normal,
                                                                             activity)
        end
        @test treated.cache.delta_v ≈ expected
        @test maximum(abs, treated.cache.delta_v - untreated.cache.delta_v) > 1.0e-8

        surface = findall(==(1), treated.cache.interface_activity)
        @test !isempty(surface)
        @test maximum(surface) do particle
            normal = TrixiParticles.surface_normal(treated, particle)
            shifting_velocity = TrixiParticles.extract_svector(treated.cache.delta_v,
                                                               treated, particle)
            abs(dot(shifting_velocity, normal))
        end < 1.0e-12
    end
end
