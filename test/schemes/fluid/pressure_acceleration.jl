@testset verbose=true "Pressure Acceleration" begin
    @testset verbose=true "Corresponding Density Calculator Formulation" begin
        f_1 = TrixiParticles.choose_pressure_acceleration_formulation(nothing,
                                                                      SummationDensity(),
                                                                      nothing, nothing)

        @test f_1 == TrixiParticles.pressure_acceleration_summation_density

        f_2 = TrixiParticles.choose_pressure_acceleration_formulation(nothing,
                                                                      ContinuityDensity(),
                                                                      nothing, nothing)
        @test f_2 == TrixiParticles.pressure_acceleration_continuity_density
    end

    @testset verbose=true "Illegal Inputs" begin
        # Mock initial condition
        initial_condition = Val(:initial_condition)
        Base.ndims(::Val{:initial_condition}) = 2
        Base.eltype(::Val{:initial_condition}) = Float64

        correction_dict_1 = Dict(
            "KernelCorrection" => KernelCorrection(),
            "GradientCorrection" => GradientCorrection(),
            "BlendedGradientCorrection" => BlendedGradientCorrection(0.5),
            "MixedKernelGradientCorrection" => MixedKernelGradientCorrection(),
        )

        p_fun_1 = function pressure_acceleration_1(a::Float64, b::Float64, c::Float64,
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
                                                                                                          initial_condition,
                                                                                                          correction_dict_1[correction_name])
        end

        correction_dict_2 = Dict(
            "No Correction" => nothing,
            "ShepardKernelCorrection" => ShepardKernelCorrection(),
            "AkinciFreeSurfaceCorrection" => AkinciFreeSurfaceCorrection(1.0),
        )

        p_fun_2 = function pressure_acceleration_2(a::Float64, b::Float64, c::Float64,
                                                   d::Float64, e::Float64, f::Float64,
                                                   g::SVector{2, Float64},
                                                   h::SVector{2, Float64})
            return 0.0
        end

        error_str = "when no correction or `AkinciFreeSurfaceCorrection` is " *
                    "used, the passed pressure acceleration formulation must " *
                    "provide a version with the arguments " *
                    "`m_a, m_b, rho_a, rho_b, p_a, p_b, W_a`, " *
                    "using the symmetry of the kernel gradient"

        @testset "$correction_name" for correction_name in keys(correction_dict_2)
            @test_throws ArgumentError(error_str) TrixiParticles.choose_pressure_acceleration_formulation(p_fun_2,
                                                                                                          1.0,
                                                                                                          initial_condition,
                                                                                                          correction_dict_2[correction_name])
        end
    end
end
