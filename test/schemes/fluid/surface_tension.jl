
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
        @test !method.normal_smoothing
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
        for normal_smoothing in (0, 1, nothing)
            @test_throws ArgumentError ColorfieldSurfaceNormal(; normal_smoothing)
        end
        @test ColorfieldSurfaceNormal(; normal_smoothing=true).normal_smoothing

        system_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(system_data, method)
        @test system_data["surface_normal_method"]["interface_threshold"] ≈ 0.1
        @test system_data["surface_normal_method"]["interface_taper_start"] === 0.8
        @test system_data["surface_normal_method"]["support_taper_width"] === 0.05
        @test system_data["surface_normal_method"]["normal_smoothing"] === false
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
        function build_morris_system(solver, particle_count; normal_smoothing=false)
            coordinates = zeros(2, particle_count)
            coordinates[1, :] .= range(0.0; step=0.25, length=particle_count)
            initial_condition = InitialCondition(; coordinates,
                                                 velocity=zeros(2, particle_count),
                                                 mass=ones(particle_count),
                                                 density=ones(particle_count),
                                                 particle_spacing=0.25)
            smoothing_kernel = WendlandC2Kernel{2}()
            surface_tension = SurfaceTensionMorris(; surface_tension_coefficient=0.7)
            normal_method = ColorfieldSurfaceNormal(; interface_threshold=0.1,
                                                    normal_smoothing)
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

        smoothed_system = build_morris_system(:wcsph, 2; normal_smoothing=true)
        smoothed_system.cache.surface_normal .= [1.0 1.0; 0.0 0.0]
        smoothed_system.cache.smoothed_surface_normal .= [0.0 0.0; 1.0 1.0]
        smoothed_system.cache.curvature .= 3.0
        smoothed_system.cache.delta_s .= 2.0
        smoothed_acceleration = TrixiParticles.surface_tension_acceleration(smoothed_system.surface_tension,
                                                                            smoothed_system,
                                                                            1, 1.0,
                                                                            SVector(0.0,
                                                                                    0.0))
        @test smoothed_acceleration ≈ SVector(0.0, -4.2)
        @test TrixiParticles.surface_normal(smoothed_system, 1) == SVector(1.0, 0.0)
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
        system.cache.surface_normal[1, :] .= 1.0
        system.cache.surface_normal[2, :] .= 0.0
        system.cache.curvature .= 3.0
        system.cache.delta_s .= 2.0
        system.cache.interface_activity .= 1.0
        vtk = Dict{String, Any}()
        expected_vtk_acceleration = GC.@preserve v_ode u_ode begin
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            rho_a = TrixiParticles.current_density(v, system, 1)
            velocity = TrixiParticles.current_velocity(v, system, 1)
            expected = TrixiParticles.surface_tension_acceleration(system.surface_tension,
                                                                   system, 1, rho_a,
                                                                   velocity)
            TrixiParticles.write2vtk!(vtk, v, u, 0.0, system)
            expected
        end
        @test vtk["surface_tension"][:, 1] ≈ expected_vtk_acceleration
        @test vtk["surface_delta"] == system.cache.delta_s
        @test vtk["interface_activity"] == system.cache.interface_activity
        @test vtk["surface_support_moment"] == system.cache.support_moment
        @test vtk["surface_tension_normal"][1] == SVector(1.0, 0.0)
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

    @testset "balanced continuum surface stress" begin
        initial_condition = InitialCondition(; coordinates=[0.0 0.75; 0.0 0.0],
                                             velocity=zeros(2, 2), mass=[2.0, 3.0],
                                             density=ones(2), particle_spacing=0.5)
        surface_tension = SurfaceTensionMomentumMorris(;
                                                       surface_tension_coefficient=0.7)
        normal_method = ColorfieldSurfaceNormal(; interface_threshold=0.1)
        system = WeaklyCompressibleSPHSystem(initial_condition;
                                             smoothing_kernel=WendlandC2Kernel{2}(),
                                             smoothing_length=0.5,
                                             density_calculator=SummationDensity(),
                                             state_equation=StateEquationCole(;
                                                                              sound_speed=10.0,
                                                                              reference_density=1.0,
                                                                              exponent=1),
                                             surface_tension,
                                             surface_normal_method=normal_method,
                                             reference_particle_spacing=0.5)

        @test haskey(system.cache, :delta_s)
        @test haskey(system.cache, :interface_activity)
        @test haskey(system.cache, :divergence_correction)
        @test haskey(system.cache, :surface_normal)
        @test !haskey(system.cache, :stress_tensor)

        # Capture the one-phase surface delta before normalizing the color gradient.
        system.cache.surface_normal .= [2.0 1.0; 0.0 1.0]
        system.cache.divergence_correction .= 0
        TrixiParticles.remove_invalid_normals!(system, surface_tension, normal_method)
        @test system.cache.delta_s ≈ [4.0, 2sqrt(2)]
        @test system.cache.interface_activity == [1.0, 1.0]
        @test system.cache.surface_normal[:, 1] ≈ [1.0, 0.0]
        @test system.cache.surface_normal[:, 2] ≈ [1 / sqrt(2), 1 / sqrt(2)]

        grad_kernel = SVector(0.3, -0.4)
        stress_gradient_1 = 4.0 .* (grad_kernel - SVector(1.0, 0.0) * 0.3)
        normal_2 = SVector(1 / sqrt(2), 1 / sqrt(2))
        stress_gradient_2 = 2sqrt(2) .* (grad_kernel -
                             normal_2 * dot(normal_2, grad_kernel))
        @test TrixiParticles.surface_stress_times_gradient(system, 1, grad_kernel) ≈
              stress_gradient_1
        @test TrixiParticles.surface_stress_times_gradient(system, 2, grad_kernel) ≈
              stress_gradient_2

        rho_a = 2.0
        rho_b = 3.0
        system.cache.divergence_correction .= [0.5, 1.0]
        divergence_correction = 2 / (0.5 + 1.0)
        pos_diff = SVector(-0.75, 0.0)
        distance = norm(pos_diff)
        dv_a = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(dv_a, surface_tension, surface_tension,
                                              system, system, 1, 2, pos_diff, distance,
                                              rho_a, rho_b, grad_kernel, 4.0)
        expected = 3divergence_correction * surface_tension.surface_tension_coefficient /
                   (rho_a * rho_b) * (stress_gradient_1 + stress_gradient_2)
        @test dv_a[] ≈ expected

        # The symmetric stress divergence conserves pairwise momentum and deliberately
        # ignores the Akinci-specific correction factor passed above.
        dv_b = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(dv_b, surface_tension, surface_tension,
                                              system, system, 2, 1, -pos_diff, distance,
                                              rho_b, rho_a, -grad_kernel, 4.0)
        @test 2dv_a[] ≈ -3dv_b[]

        semi = Semidiscretization(system)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        vtk = Dict{String, Any}()
        GC.@preserve v_ode u_ode begin
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            TrixiParticles.write2vtk!(vtk, v, u, 0.0, system)
        end
        @test vtk["surface_delta"] == system.cache.delta_s
        @test vtk["interface_activity"] == system.cache.interface_activity
        @test vtk["surface_tension_normal"] == [TrixiParticles.surface_normal(system, 1),
            TrixiParticles.surface_normal(system, 2)]
        @test vtk["surface_divergence_correction"] == [0.5, 1.0]
        @test size(vtk["surface_stress_tensor"]) == (2, 2, 2)
        @test vtk["surface_stress_tensor"][:, :, 1] ≈ [0.0 0.0; 0.0 4.0]
        @test all(isfinite, vtk["surface_stress_tensor"])

        system.cache.divergence_correction .= 0
        unsupported_force = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(unsupported_force, surface_tension,
                                              surface_tension, system, system, 1, 2,
                                              pos_diff, distance, rho_a, rho_b,
                                              grad_kernel, 1.0)
        @test iszero(unsupported_force[])

        filtered_method = ColorfieldSurfaceNormal(; interface_threshold=0.1,
                                                  ideal_density_threshold=0.9,
                                                  support_taper_width=0.05)
        system.cache.surface_normal .= 0
        system.cache.surface_normal[1, 1] = 0.2
        system.cache.divergence_correction .= [0.925, 1.0]
        TrixiParticles.remove_invalid_normals!(system, surface_tension, filtered_method)
        @test system.cache.interface_activity[1] ≈ 0.5
        @test system.cache.delta_s[1] ≈ 0.2
        @test system.cache.surface_normal[:, 1] == [1.0, 0.0]
    end

    @testset "CSS static Laplace balance" begin
        reference_density = 1000.0
        target_particles = 375
        drop_volume = 1.0e-6
        particle_spacing = cbrt(drop_volume / target_particles)
        radius = cbrt(3drop_volume / (4pi))
        initial_condition = SphereShape(particle_spacing, radius + particle_spacing / 2,
                                        (0.0, 0.0, 0.0), reference_density;
                                        sphere_type=VoxelSphere())
        smoothing_kernel = WendlandC2Kernel{3}()
        smoothing_length = 1.4particle_spacing

        function initial_acceleration(system)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
            return GC.@preserve v_ode u_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                u = TrixiParticles.wrap_u(u_ode, system, semi)
                dv = zeros(eltype(v), size(v))
                TrixiParticles.interact!(dv, v, u, v, u, system, system, semi)
                Array(dv[1:3, :])
            end
        end

        coefficient = 1.0
        css = SurfaceTensionMomentumMorris(; surface_tension_coefficient=coefficient)
        css_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                 smoothing_length,
                                                 density_calculator=ContinuityDensity(),
                                                 state_equation=StateEquationCole(;
                                                                                  sound_speed=100.0,
                                                                                  reference_density,
                                                                                  exponent=1),
                                                 surface_tension=css,
                                                 surface_normal_method=ColorfieldSurfaceNormal(;
                                                                                               boundary_contact_threshold=Inf,
                                                                                               interface_threshold=0.01,
                                                                                               ideal_density_threshold=0.95),
                                                 reference_particle_spacing=particle_spacing)
        css_acceleration = initial_acceleration(css_system)

        pressure_basis = 1.0
        sound_speed = 100.0
        pressure_reference_density = reference_density - pressure_basis / sound_speed^2
        pressure_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                      smoothing_length,
                                                      density_calculator=ContinuityDensity(),
                                                      state_equation=StateEquationCole(;
                                                                                       sound_speed,
                                                                                       reference_density=pressure_reference_density,
                                                                                       exponent=1))
        pressure_acceleration = initial_acceleration(pressure_system) / pressure_basis

        interface = findall(>(0), css_system.cache.delta_s)
        capillary = vec(css_acceleration[:, interface])
        unit_pressure = vec(pressure_acceleration[:, interface])
        pressure_jump = -dot(capillary, unit_pressure) / dot(unit_pressure, unit_pressure)
        volume = sum(css_system.mass) / reference_density
        equivalent_radius = cbrt(3volume / (4pi))
        inferred_surface_tension = pressure_jump * equivalent_radius / 2
        total_force = vec(sum(css_acceleration .* reshape(css_system.mass, 1, :);
                              dims=2))

        @test inferred_surface_tension ≈ coefficient rtol = 0.05
        @test norm(total_force) < 1.0e-12
        @test all(isfinite, css_system.cache.divergence_correction)
        @test minimum(css_system.cache.divergence_correction) > 0
    end
end
