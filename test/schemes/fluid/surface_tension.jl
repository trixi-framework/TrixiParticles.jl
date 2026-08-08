
@testset verbose=true "Surface Tension" begin
    @testset "smooth interface activity" begin
        method = ColorfieldSurfaceNormal(; boundary_contact_threshold=1,
                                         interface_threshold=0.1f0,
                                         ideal_density_threshold=0.9,
                                         interface_taper_start=0.8,
                                         support_taper_width=0.05)
        @test method isa ColorfieldSurfaceNormal{Float64}
        @test method.interface_taper_start === 0.8
        @test method.support_taper_width === 0.05
        @test ColorfieldSurfaceNormal(1, 1, 0) isa ColorfieldSurfaceNormal{Float64}
        @test ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1f0,
                                      interface_threshold=0.01f0,
                                      ideal_density_threshold=0.0f0,
                                      interface_taper_start=0.8f0,
                                      support_taper_width=0.025f0) isa
              ColorfieldSurfaceNormal{Float32}

        for ELTYPE in (Float32, Float64)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(-1)) === ELTYPE(0)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(0)) === ELTYPE(0)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(0.5)) === ELTYPE(0.5)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(1)) === ELTYPE(1)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(2)) === ELTYPE(1)

            method_ = ColorfieldSurfaceNormal(; boundary_contact_threshold=ELTYPE(0.1),
                                              interface_threshold=ELTYPE(0.1),
                                              ideal_density_threshold=ELTYPE(0.9),
                                              interface_taper_start=ELTYPE(0.8),
                                              support_taper_width=ELTYPE(0.05))
            @test TrixiParticles.gradient_interface_activity(ELTYPE(0.08), one(ELTYPE),
                                                             method_) === ELTYPE(0)
            @test TrixiParticles.gradient_interface_activity(ELTYPE(0.09), one(ELTYPE),
                                                             method_) ≈ ELTYPE(0.5)
            @test TrixiParticles.gradient_interface_activity(ELTYPE(0.1), one(ELTYPE),
                                                             method_) === ELTYPE(1)
            @test TrixiParticles.support_interface_activity(ELTYPE(0.9), method_) ===
                  ELTYPE(1)
            @test TrixiParticles.support_interface_activity(ELTYPE(0.925), method_) ≈
                  ELTYPE(0.5)
            @test TrixiParticles.support_interface_activity(ELTYPE(0.95), method_) ===
                  ELTYPE(0)

            step = sqrt(eps(ELTYPE))
            derivative_at_zero = TrixiParticles.cubic_smoothstep(step) / step
            derivative_at_one = (one(ELTYPE) -
                                 TrixiParticles.cubic_smoothstep(one(ELTYPE) - step)) / step
            @test abs(derivative_at_zero) < 4step
            @test abs(derivative_at_one) < 4step
        end

        disabled = ColorfieldSurfaceNormal(; ideal_density_threshold=0.0)
        @test TrixiParticles.support_interface_activity(10.0, disabled) == 1.0
        @test TrixiParticles.normalized_surface_curvature(1.0, 0.0) == 0.0
        @test TrixiParticles.normalized_surface_curvature(1.0, eps()) == 0.0
        @test TrixiParticles.normalized_surface_curvature(2.0, 0.5) == 4.0

        for threshold in (-1, NaN, Inf)
            @test_throws ArgumentError ColorfieldSurfaceNormal(interface_threshold=threshold)
            @test_throws ArgumentError ColorfieldSurfaceNormal(ideal_density_threshold=threshold)
        end
        for taper_start in (-0.1, 1.0, NaN, Inf)
            @test_throws ArgumentError ColorfieldSurfaceNormal(;
                                                               interface_taper_start=taper_start)
        end
        for taper_width in (0.0, -0.1, NaN, Inf)
            @test_throws ArgumentError ColorfieldSurfaceNormal(;
                                                               support_taper_width=taper_width)
        end

        system_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(system_data, method)
        @test system_data["surface_normal_method"]["interface_threshold"] ≈ 0.1
        @test system_data["surface_normal_method"]["interface_taper_start"] === 0.8
        @test system_data["surface_normal_method"]["support_taper_width"] === 0.05
    end

    @testset verbose=true "`cohesion_force_akinci`" begin
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.0
        m_b = 1.0
        pos_diff = [1.0, 1.0]

        # These values can be extracted from the graphs in the paper by Akinci et al. or by manual calculation.
        # Additional digits have been accepted from the actual calculation.
        test_distance = 0.1
        val = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance) * test_distance
        @test isapprox(val[1], 0.1443038770421044, atol=6e-15)
        @test isapprox(val[2], 0.1443038770421044, atol=6e-15)

        # Maximum repulsion force
        test_distance = 0.01
        max = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance) * test_distance
        @test isapprox(max[1], 0.15913517632298307, atol=6e-15)
        @test isapprox(max[2], 0.15913517632298307, atol=6e-15)

        # Near 0
        test_distance = 0.2725
        zero = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance) * test_distance
        @test isapprox(zero[1], 0.0004360543645195717, atol=6e-15)
        @test isapprox(zero[2], 0.0004360543645195717, atol=6e-15)

        # Maximum attraction force
        test_distance = 0.5
        maxa = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance) * test_distance
        @test isapprox(maxa[1], -0.15915494309189535, atol=6e-15)
        @test isapprox(maxa[2], -0.15915494309189535, atol=6e-15)

        # Should be 0
        test_distance = 1.0
        zero = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance) * test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)
    end

    @testset verbose=true "adhesion_force_akinci" begin
        surface_tension = TrixiParticles.SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.0
        m_b = 1.0
        pos_diff = [1.0, 1.0]

        # These values can be extracted from the graphs in the paper by Akinci et al. or by manual calculation.
        # Additional digits have been accepted from the actual calculation.
        test_distance = 0.1
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        test_distance = 0.5
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        # Near 0
        test_distance = 0.51
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0) *
               test_distance
        @test isapprox(zero[1], -0.002619160170741761, atol=6e-15)
        @test isapprox(zero[2], -0.002619160170741761, atol=6e-15)

        # Maximum adhesion force
        test_distance = 0.75
        max = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance, 1.0) *
              test_distance
        @test isapprox(max[1], -0.004949747468305833, atol=6e-15)
        @test isapprox(max[2], -0.004949747468305833, atol=6e-15)

        # Should be 0
        test_distance = 1.0
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)
    end

    @testset "Morris CSF local force" begin
        function build_morris_system(solver, particle_count)
            coordinates = zeros(2, particle_count)
            coordinates[1, :] .= range(0.0; step=0.25, length=particle_count)
            initial_condition = InitialCondition(; coordinates,
                                                 velocity=zeros(2, particle_count),
                                                 mass=ones(particle_count),
                                                 density=ones(particle_count),
                                                 particle_spacing=0.25)
            smoothing_kernel = WendlandC2Kernel{2}()
            surface_tension = SurfaceTensionMorris(; surface_tension_coefficient=0.7)
            normal_method = ColorfieldSurfaceNormal(; interface_threshold=0.1)
            if solver == :wcsph
                return WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                   smoothing_length=0.5,
                                                   density_calculator=ContinuityDensity(),
                                                   state_equation=StateEquationCole(;
                                                                                    sound_speed=10.0,
                                                                                    reference_density=1.0,
                                                                                    exponent=1),
                                                   surface_tension,
                                                   surface_normal_method=normal_method,
                                                   reference_particle_spacing=0.25)
            end
            return EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                               smoothing_length=0.5, sound_speed=10.0,
                                               density_calculator=ContinuityDensity(),
                                               surface_tension,
                                               surface_normal_method=normal_method,
                                               reference_particle_spacing=0.25)
        end

        function morris_rhs_effect(system)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
            system.cache.surface_normal[1, :] .= 1.0
            system.cache.surface_normal[2, :] .= 0.0
            system.cache.curvature .= 3.0
            system.cache.delta_s .= 2.0
            system.cache.interface_activity .= 1.0

            return GC.@preserve v_ode u_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                u = TrixiParticles.wrap_u(u_ode, system, semi)
                rho_a = TrixiParticles.current_density(v, system, 1)
                expected = TrixiParticles.surface_tension_acceleration(system.surface_tension,
                                                                       system, 1, rho_a,
                                                                       SVector(0.0, 0.0))
                with_surface_tension = zeros(eltype(v), size(v))
                TrixiParticles.interact!(with_surface_tension, v, u, v, u,
                                         system, system, semi)
                system.cache.delta_s .= 0
                without_surface_tension = zeros(eltype(v), size(v))
                TrixiParticles.interact!(without_surface_tension, v, u, v, u,
                                         system, system, semi)
                return (with_surface_tension - without_surface_tension)[1:2, :],
                       expected
            end
        end

        effects = []
        for solver in (:wcsph, :edac), particle_count in (2, 4)
            effect,
            expected = morris_rhs_effect(build_morris_system(solver, particle_count))
            @test all(particle -> effect[:, particle] ≈ expected, axes(effect, 2))
            push!(effects, effect[:, 1])
        end
        @test all(effect -> effect ≈ first(effects), effects)

        system = build_morris_system(:wcsph, 2)
        system.cache.surface_normal .= [2.0 1.0; 0.0 1.0]
        system.cache.support_moment .= 0
        TrixiParticles.remove_invalid_normals!(system, system.surface_tension,
                                               system.surface_normal_method)
        @test system.cache.delta_s ≈ [4.0, 2sqrt(2)]
        @test system.cache.interface_activity == [1.0, 1.0]
        @test system.cache.surface_normal[:, 1] ≈ [1.0, 0.0]
        @test system.cache.surface_normal[:, 2] ≈ [1 / sqrt(2), 1 / sqrt(2)]
        system.cache.surface_normal[:, 1] .= [NaN, 0.0]
        TrixiParticles.remove_invalid_normals!(system, system.surface_tension,
                                               system.surface_normal_method)
        @test iszero(system.cache.surface_normal[:, 1])
        @test iszero(system.cache.delta_s[1])
        @test iszero(system.cache.interface_activity[1])

        system.cache.surface_normal[1, :] .= 1.0
        system.cache.surface_normal[2, :] .= 0.0
        system.cache.curvature .= 3.0
        system.cache.delta_s .= 2.0
        acceleration = TrixiParticles.surface_tension_acceleration(system.surface_tension,
                                                                   system, 1, 1.0,
                                                                   SVector(0.0, 0.0))
        @test acceleration ≈ SVector(-4.2, 0.0)
        system.cache.curvature[1] /= 2
        system.cache.delta_s[1] /= 2
        scaled_acceleration = TrixiParticles.surface_tension_acceleration(system.surface_tension,
                                                                          system, 1, 1.0,
                                                                          SVector(0.0,
                                                                                  0.0))
        @test scaled_acceleration ≈ acceleration / 4

        semi = Semidiscretization(system)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
        system.cache.surface_normal .= [1.0 0.0; 0.0 1.0]

        function curvature_with_neighbor_activity(activity)
            system.cache.interface_activity .= [1.0, activity]
            fill!(system.cache.curvature, 0)
            fill!(system.cache.correction_factor, 0)
            GC.@preserve v_ode u_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                u = TrixiParticles.wrap_u(u_ode, system, semi)
                TrixiParticles.calc_curvature!(system, system, u, v, v, u, semi,
                                               system.surface_normal_method,
                                               system.surface_normal_method)
            end
            denominator = system.cache.correction_factor[1]
            return TrixiParticles.normalized_surface_curvature(system.cache.curvature[1],
                                                               denominator)
        end

        curvature_zero = curvature_with_neighbor_activity(0.0)
        curvature_small = curvature_with_neighbor_activity(1.0e-6)
        curvature_full = curvature_with_neighbor_activity(1.0)
        @test iszero(curvature_zero)
        @test abs(curvature_small) < 1.0e-4 * abs(curvature_full)
        @test isfinite(curvature_full)

        curvature_numerator = copy(system.cache.curvature)
        correction_factor = copy(system.cache.correction_factor)
        GC.@preserve v_ode u_ode begin
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            TrixiParticles.calc_curvature!(system, system, u, v, v, u, semi,
                                           system.surface_normal_method,
                                           system.surface_normal_method)
        end
        @test system.cache.curvature ≈ 2curvature_numerator
        @test system.cache.correction_factor ≈ 2correction_factor
    end

    @testset "compute_stress_tensors! (MomentumMorris)" begin
        # 1. Define Minimal Initial Condition with 2 Particles in 2D
        coords = [0.0 1.0;
                  0.0 0.0]
        velocity = zeros(2, 2)
        mass = ones(2)
        density = ones(2)

        ic = InitialCondition(; coordinates=coords, velocity, mass, density,
                              particle_spacing=1.0)

        # 2. Define Density Calculator, State Equation, and Kernel
        density_calc = SummationDensity()
        eq_state = StateEquationCole(sound_speed=10.0,
                                     reference_density=1.0,
                                     exponent=1)
        kernel = WendlandC2Kernel{2}()
        smoothing_length = 0.5

        # 3. Create the WeaklyCompressibleSPHSystem with Surface Tension
        system = WeaklyCompressibleSPHSystem(ic; smoothing_kernel=kernel,
                                             smoothing_length,
                                             density_calculator=density_calc,
                                             state_equation=eq_state,
                                             surface_tension=SurfaceTensionMomentumMorris(surface_tension_coefficient=1.0),
                                             surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                                           ideal_density_threshold=0.9),
                                             reference_particle_spacing=1.0,)

        # 4. Verify Cache Contains Necessary Fields
        @test haskey(system.cache, :delta_s)
        @test haskey(system.cache, :surface_normal)
        @test haskey(system.cache, :stress_tensor)

        # 5. Manually Populate `delta_s` and `surface_normal`
        system.cache.delta_s .= [1.0, 2.0]
        system.cache.surface_normal .= hcat([1.0, 0.0], [1 / sqrt(2), 1 / sqrt(2)])
        system.cache.stress_tensor .= zeros(2, 2, 2)  # Reset to zero before computation

        # 6. Call `compute_stress_tensors!` with `SurfaceTensionMomentumMorris`
        TrixiParticles.compute_stress_tensors!(system,
                                               SurfaceTensionMomentumMorris(),
                                               nothing, nothing,  # v, u (not needed for stress computation)
                                               nothing, nothing,  # v_ode, u_ode (not needed)
                                               SerialBackend(),   # semi (only passed to `@threaded`)
                                               0.0)

        # 7. Define Reference Stress Tensors by Hand
        #
        # Reference calculations based on the formula:
        # σ_ij(a) = δs_a (δ_ij - n_i n_j) - δ_ij max(δs)
        #
        # For Particle 1:
        # δs = 1.0
        # n = (1.0, 0.0)
        # max(δs) = 2.0
        # σ_11 = 1*(1 - 1^2) - 1*2 = -2
        # σ_12 = 1*(0 - 1*0) - 0*2 = 0
        # σ_21 = 1*(0 - 1*0) - 0*2 = 0
        # σ_22 = 1*(1 - 0^2) - 1*2 = 1 - 2 = -1
        #
        # Resulting Stress Tensor for Particle 1:
        # [-2.0  0.0
        #   0.0 -1.0]
        #
        # For Particle 2:
        # δs = 2.0
        # n = (1/√2, 1/√2)
        # max(δs) = 2.0
        # σ_11 = 2*(1 - (1/√2)^2) - 1*2 = 2*(1 - 0.5) - 2 = 1 - 2 = -1
        # σ_12 = 2*(0 - (1/√2)^2) - 0*2 = 2*(0 - 0.5) = -1
        # σ_21 = 2*(0 - (1/√2)^2) - 0*2 = -1
        # σ_22 = 2*(1 - (1/√2)^2) - 1*2 = 2*(1 - 0.5) - 2 = 1 - 2 = -1
        #
        # Resulting Stress Tensor for Particle 2:
        # [-1.0 -1.0
        #  -1.0 -1.0]

        ref_particle_1 = [-2.0 0.0;
                          0.0 -1.0]
        ref_particle_2 = [-1.0 -1.0;
                          -1.0 -1.0]

        # 8. Retrieve Computed Stress Tensor
        computed = system.cache.stress_tensor

        # 9. Perform Assertions
        @test all(isfinite, computed)

        @test isapprox(computed[:, :, 1], ref_particle_1; atol=1e-14)
        @test isapprox(computed[:, :, 2], ref_particle_2; atol=1e-14)
    end
end
