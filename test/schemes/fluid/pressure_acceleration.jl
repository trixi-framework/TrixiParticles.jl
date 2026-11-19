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

    @testset "Custom pressure formulations with TVF" begin
        initial_condition = InitialCondition(coordinates=zeros(2, 1), density=1000.0,
                                             particle_spacing=1.0)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        sound_speed = 10.0
        state_equation = StateEquationCole(; sound_speed, reference_density=1000.0,
                                           exponent=1)
        shifting_technique = TransportVelocityAdami(background_pressure=100.0)

        linear_pressure(args...) = TrixiParticles.pressure_acceleration_continuity_density(args...)
        scalar_pressure(m_a, m_b, rho_a, rho_b, p_a::Real, p_b::Real,
                        gradients...) = linear_pressure(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                        gradients...)

        error = ArgumentError("`TransportVelocityAdami` requires a custom pressure " *
                              "acceleration formulation to accept `SMatrix` tensors in " *
                              "the pressure arguments for the selected correction.")
        constructors = ((WeaklyCompressibleSPHSystem, (; state_equation)),
                        (EntropicallyDampedSPHSystem, (; sound_speed)))
        for (constructor, kwargs) in constructors,
            correction in (nothing, GradientCorrection())
            parameters = (; smoothing_kernel, smoothing_length=1.0, correction,
                          density_calculator=ContinuityDensity(), kwargs...)
            @test_nowarn constructor(initial_condition; parameters...,
                                     pressure_acceleration=scalar_pressure)
            @test_throws error constructor(initial_condition; parameters...,
                                           pressure_acceleration=scalar_pressure,
                                           shifting_technique)
            @test_nowarn constructor(initial_condition; parameters...,
                                     pressure_acceleration=linear_pressure,
                                     shifting_technique)
        end
    end
end
