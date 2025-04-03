
@testset verbose=true "Surface Tension" begin
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

    @testset "compute_stress_tensors! (MomentumMorris)" begin
        # 1. Define Minimal Initial Condition with 2 Particles in 2D
        coords = [0.0 1.0;
                  0.0 0.0]
        velocity = zeros(2, 2)
        mass = ones(2)
        density = ones(2)

        ic = InitialCondition(coordinates=coords,
                              velocity=velocity,
                              mass=mass,
                              density=density,
                              particle_spacing=1.0)

        # 2. Define Density Calculator, State Equation, and Kernel
        density_calc = SummationDensity()
        eq_state = StateEquationCole(sound_speed=10.0,
                                     reference_density=1.0,
                                     exponent=1)
        kernel = WendlandC2Kernel{2}()
        smoothing_length = 1.0

        # 3. Create the WeaklyCompressibleSPHSystem with Surface Tension
        system = WeaklyCompressibleSPHSystem(ic,
                                             density_calc,
                                             eq_state,
                                             kernel,
                                             smoothing_length;
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
                                               nothing,           # semi (not needed)
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
