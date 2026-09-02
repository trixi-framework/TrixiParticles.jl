@testset verbose=true "Surface Tension" begin
    @testset "constructors and capabilities" begin
        constructors = (CohesionForceAkinci, SurfaceTensionAkinci,
                        SurfaceTensionMorris, SurfaceTensionMomentumMorris)

        for constructor in constructors
            model = constructor(surface_tension_coefficient=0.5f0)
            @test model.surface_tension_coefficient === 0.5f0
            @test iszero(constructor(surface_tension_coefficient=0).surface_tension_coefficient)

            for coefficient in (-1.0, NaN, Inf, -Inf, 1.0im, "invalid")
                @test_throws ArgumentError constructor(surface_tension_coefficient=coefficient)
            end
        end

        @test !TrixiParticles.requires_surface_normal(nothing)
        @test !TrixiParticles.requires_surface_normal(CohesionForceAkinci())
        @test TrixiParticles.requires_surface_normal(SurfaceTensionAkinci())
        @test TrixiParticles.requires_surface_normal(SurfaceTensionMorris())
        @test TrixiParticles.requires_surface_normal(SurfaceTensionMomentumMorris())

        normal_method = ColorfieldSurfaceNormal(boundary_contact_threshold=1,
                                                interface_threshold=0.1f0,
                                                ideal_density_threshold=0.25)
        @test normal_method isa ColorfieldSurfaceNormal{Float64}
        @test ColorfieldSurfaceNormal(boundary_contact_threshold=0.1f0,
                                      interface_threshold=0.01f0,
                                      ideal_density_threshold=0.0f0) isa
              ColorfieldSurfaceNormal{Float32}

        for normal_method in
            (() -> ColorfieldSurfaceNormal(boundary_contact_threshold=-0.1),
             () -> ColorfieldSurfaceNormal(boundary_contact_threshold=1.1),
             () -> ColorfieldSurfaceNormal(boundary_contact_threshold=NaN),
             () -> ColorfieldSurfaceNormal(boundary_contact_threshold="invalid"),
             () -> ColorfieldSurfaceNormal(interface_threshold=-0.1),
             () -> ColorfieldSurfaceNormal(interface_threshold=Inf),
             () -> ColorfieldSurfaceNormal(ideal_density_threshold=-0.1),
             () -> ColorfieldSurfaceNormal(ideal_density_threshold=1.1),
             () -> ColorfieldSurfaceNormal(interface_threshold=1.0im))
            @test_throws ArgumentError normal_method()
        end
    end

    @testset "cohesion-only systems do not require normals" begin
        coordinates = [0.0 1.0;
                       0.0 0.0]
        initial_condition = InitialCondition(; coordinates, density=ones(2),
                                             particle_spacing=1.0)
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = 1.0
        surface_tension = CohesionForceAkinci(surface_tension_coefficient=0.1)

        wcsph = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                            smoothing_length,
                                            density_calculator=SummationDensity(),
                                            state_equation=StateEquationCole(sound_speed=10.0,
                                                                             reference_density=1.0,
                                                                             exponent=1),
                                            surface_tension, color_value=1)
        edac = EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                           smoothing_length, sound_speed=10.0,
                                           density_calculator=SummationDensity(),
                                           surface_tension, color_value=0)

        for system in (wcsph, edac)
            @test isnothing(system.surface_method)
            @test !haskey(system.cache, :surface_normal)
            @test !haskey(system.cache, :neighbor_count)
            @test !haskey(system.cache, :reference_particle_spacing)

            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.1))
            v_ode, u_ode = ode.u0.x
            dv_ode = zero(v_ode)
            @test_nowarn TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
            @test all(isfinite, dv_ode)
            @test any(!iszero, dv_ode)
        end

        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               density_calculator=SummationDensity(),
                                                               state_equation=StateEquationCole(sound_speed=10.0,
                                                                                                reference_density=1.0,
                                                                                                exponent=1),
                                                               surface_tension=SurfaceTensionAkinci())
        @test_throws ArgumentError EntropicallyDampedSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               sound_speed=10.0,
                                                               density_calculator=SummationDensity(),
                                                               surface_tension=SurfaceTensionAkinci())

        full_akinci = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                  smoothing_length,
                                                  density_calculator=SummationDensity(),
                                                  state_equation=StateEquationCole(sound_speed=10.0,
                                                                                   reference_density=1.0,
                                                                                   exponent=1),
                                                  surface_tension=SurfaceTensionAkinci(),
                                                  reference_particle_spacing=1.0,
                                                  color_value=0)
        @test full_akinci.surface_method isa ColorfieldSurfaceNormal
        @test haskey(full_akinci.cache, :surface_normal)
    end

    @testset "zero Morris coefficient does not restrict the time step" begin
        function calculate_initial_dt(surface_tension)
            initial_condition = InitialCondition(; coordinates=[0.0 1.0; 0.0 0.0],
                                                 density=ones(2), particle_spacing=1.0)
            reference_particle_spacing = isnothing(surface_tension) ? 0 : 1.0
            system = WeaklyCompressibleSPHSystem(initial_condition;
                                                 smoothing_kernel=WendlandC2Kernel{2}(),
                                                 smoothing_length=1.0,
                                                 density_calculator=SummationDensity(),
                                                 state_equation=StateEquationCole(sound_speed=10.0,
                                                                                  reference_density=1.0,
                                                                                  exponent=1),
                                                 surface_tension,
                                                 reference_particle_spacing)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.1))
            v_ode, u_ode = ode.u0.x
            return TrixiParticles.calculate_dt(v_ode, u_ode, 0.25, semi.systems[1], semi)
        end

        dt_without_surface_tension = calculate_initial_dt(nothing)
        dt_with_zero_csf = calculate_initial_dt(SurfaceTensionMorris(;
                                                                     surface_tension_coefficient=0.0))
        dt_with_zero_css = calculate_initial_dt(SurfaceTensionMomentumMorris(;
                                                                             surface_tension_coefficient=0.0))

        @test dt_with_zero_csf == dt_without_surface_tension
        @test dt_with_zero_css == dt_without_surface_tension
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

        surface_tension_f32 = CohesionForceAkinci(surface_tension_coefficient=1.0f0)
        for support_radius in (1.0f12, 1.0f-12)
            distance = 0.75f0 * support_radius
            force = TrixiParticles.cohesion_force_akinci(surface_tension_f32,
                                                         support_radius, 1.0f0,
                                                         Float32[distance, 0], distance)
            expected = Float32(-32 / pi * (1 - 0.75)^3 * 0.75^3 /
                               Float64(support_radius)^3)
            @test eltype(force) == Float32
            @test all(isfinite, force)
            @test isapprox(force[1], expected; rtol=4eps(Float32))
            @test iszero(force[2])
        end
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

        support_radius_f32 = 15.594092f0
        distance_f32 = prevfloat(support_radius_f32)
        near_support = TrixiParticles.adhesion_force_akinci(surface_tension,
                                                            support_radius_f32, 1.0f0,
                                                            Float32[1, 0], distance_f32,
                                                            1.0f0)
        @test eltype(near_support) == Float32
        @test all(isfinite, near_support)
        @test 0 < norm(near_support) < eps(Float32)

        for support_radius in (1.0f12, 1.0f-13)
            distance = 0.75f0 * support_radius
            force = TrixiParticles.adhesion_force_akinci(surface_tension,
                                                         support_radius, 1.0f0,
                                                         Float32[distance, 0], distance,
                                                         1.0f0)
            expected = Float32(-0.007 / Float64(support_radius)^3 / sqrt(2))
            @test all(isfinite, force)
            @test isapprox(force[1], expected; rtol=4eps(Float32))
            @test iszero(force[2])
        end
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
                                             surface_method=ColorfieldSurfaceNormal(interface_threshold=0.0),
                                             reference_particle_spacing=1.0,)

        # 4. Verify Cache Contains Necessary Fields
        @test haskey(system.cache, :delta_s)
        @test haskey(system.cache, :surface_normal)
        @test haskey(system.cache, :stress_tensor)

        # Filtering retains the raw gradient magnitude as the surface delta before
        # normalizing the direction used in the stress tensor.
        system.cache.surface_normal .= [3.0 0.0;
                                        4.0 2.0]
        system.cache.neighbor_count .= 10
        TrixiParticles.finalize_surface!(system, system.surface_tension,
                                         system.surface_method, SerialBackend())
        @test system.cache.surface_normal[:, 1] ≈ [0.6, 0.8]
        @test system.cache.surface_normal[:, 2] ≈ [0.0, 1.0]
        @test system.cache.delta_s ≈ [5.0, 2.0]

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
