@testset verbose=true "Pressure Acceleration" begin
    @testset verbose=true "Corresponding Density Calculator Formulation" begin
        f_1 = TrixiParticles.choose_pressure_acceleration_formulation(nothing,
                                                                      SummationDensity(),
                                                                      2, Float64, nothing)

        @test f_1 == TrixiParticles.pressure_acceleration_summation_density

        f_2 = TrixiParticles.choose_pressure_acceleration_formulation(nothing,
                                                                      ContinuityDensity(),
                                                                      2, Float64, nothing)
        @test f_2 == TrixiParticles.pressure_acceleration_continuity_density
    end

    @testset "Algebraic formulations and asymmetric conservation" begin
        m_a, m_b = 1.2, 0.8
        rho_a, rho_b = 1000.0, 980.0
        p_a, p_b = 2.0, 3.0
        W_a = SVector(0.2, -0.1)
        W_b = -W_a

        summation = TrixiParticles.pressure_acceleration_summation_density
        continuity = TrixiParticles.pressure_acceleration_continuity_density
        interparticle = TrixiParticles.inter_particle_averaged_pressure

        @test summation(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a) ≈
              -m_b * (p_a / rho_a^2 + p_b / rho_b^2) * W_a
        @test continuity(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a) ≈
              -m_b * (p_a + p_b) / (rho_a * rho_b) * W_a

        volume_term = ((m_a / rho_a)^2 + (m_b / rho_b)^2) / m_a
        pressure_tilde = (rho_b * p_a + rho_a * p_b) / (rho_a + rho_b)
        @test interparticle(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a) ≈
              -volume_term * pressure_tilde * W_a
        @test tensile_instability_control(m_a, m_b, rho_a, rho_b, -p_a, p_b, W_a) ≈
              -m_b * (p_a + p_b) / (rho_a * rho_b) * W_a

        for pressure_formulation in (summation, continuity, interparticle)
            symmetric = pressure_formulation(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
            asymmetric = pressure_formulation(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                              W_a, W_b)
            @test asymmetric ≈ symmetric
            @test pressure_formulation(m_a, m_b, rho_a, rho_b, 0.0, 0.0,
                                       W_a, W_b) == zero(W_a)

            acceleration_a = pressure_formulation(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                  W_a, W_b)
            acceleration_b = pressure_formulation(m_b, m_a, rho_b, rho_a, p_b, p_a,
                                                  W_b, W_a)
            @test m_a * acceleration_a + m_b * acceleration_b ≈ zero(W_a) atol = eps()
        end
        @test tensile_instability_control(m_a, m_b, rho_a, rho_b, 0.0, 0.0,
                                          W_a) == zero(W_a)
    end

    @testset verbose=true "Illegal Inputs" begin
        correction_dict_1 = Dict(
            "KernelCorrection" => KernelCorrection(),
            "GradientCorrection" => GradientCorrection(),
            "BlendedGradientCorrection" => BlendedGradientCorrection(0.5),
            "MixedKernelGradientCorrection" => MixedKernelGradientCorrection()
        )

        function p_fun_1(a::Float64, b::Float64, c::Float64,
                         d::Float64, e::Float64, f::Float64,
                         g::SVector{2, Float64})
            return 0.0
        end

        error_str = "when a correction with an asymmetric kernel gradient is " *
                    "used, the passed pressure acceleration formulation must " *
                    "provide a version with the arguments " *
                    "`m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, W_b`"

        @testset "$correction_name" for correction_name in keys(correction_dict_1)
            @test_throws ArgumentError(error_str) TrixiParticles.choose_pressure_acceleration_formulation(p_fun_1,
                                                                                                          1.0,
                                                                                                          2,
                                                                                                          Float64,
                                                                                                          correction_dict_1[correction_name])
        end

        correction_dict_2 = Dict(
            "No Correction" => nothing,
            "ShepardKernelCorrection" => ShepardKernelCorrection(),
            "AkinciFreeSurfaceCorrection" => AkinciFreeSurfaceCorrection(1.0)
        )

        function p_fun_2(a::Float64, b::Float64, c::Float64,
                         d::Float64, e::Float64, f::Float64,
                         g::SVector{2, Float64}, h::SVector{2, Float64})
            return 0.0
        end

        error_str = "when not using a correction with an asymmetric kernel " *
                    "gradient, the passed pressure acceleration formulation must " *
                    "provide a version with the arguments " *
                    "`m_a, m_b, rho_a, rho_b, p_a, p_b, W_a`, " *
                    "using the symmetry of the kernel gradient"

        @testset "$correction_name" for correction_name in keys(correction_dict_2)
            @test_throws ArgumentError(error_str) TrixiParticles.choose_pressure_acceleration_formulation(p_fun_2,
                                                                                                          1.0,
                                                                                                          2,
                                                                                                          Float64,
                                                                                                          correction_dict_2[correction_name])
        end
    end
end
