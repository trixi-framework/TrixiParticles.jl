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

    @testset "Interface-aware tensile control" begin
        control = InterfaceAwareTensileInstabilityControl()
        m_a = m_b = rho_a = rho_b = 1.0
        p_a = -2.0
        p_b = 0.5
        grad_kernel = SVector(1.0, -0.5)
        standard = TrixiParticles.pressure_acceleration_continuity_density(m_a, m_b,
                                                                           rho_a, rho_b,
                                                                           p_a, p_b,
                                                                           grad_kernel)
        controlled = tensile_instability_control(m_a, m_b, rho_a, rho_b,
                                                 p_a, p_b, grad_kernel)
        @test TrixiParticles.interface_aware_tensile_acceleration(m_a, m_b, rho_a,
                                                                  rho_b, p_a, p_b,
                                                                  grad_kernel, 0.0,
                                                                  0.0, 1.0) == controlled
        @test TrixiParticles.interface_aware_tensile_acceleration(m_a, m_b, rho_a,
                                                                  rho_b, p_a, p_b,
                                                                  grad_kernel, 1.0,
                                                                  0.0, 1.0) == standard
        @test TrixiParticles.interface_aware_tensile_acceleration(m_a, m_b, rho_a,
                                                                  rho_b, p_a, p_b,
                                                                  grad_kernel, 0.5,
                                                                  0.0, 1.0) ==
              (standard + controlled) / 2
        @test TrixiParticles.interface_aware_tensile_acceleration(m_a, m_b, rho_a,
                                                                  rho_b, p_a, p_b,
                                                                  grad_kernel, 0.0,
                                                                  0.0, 0.25) ==
              standard + 0.25 * (controlled - standard)
        @test_throws ArgumentError InterfaceAwareTensileInstabilityControl(; strength=0)

        particle_spacing = 0.1
        initial_condition = RectangularShape(particle_spacing, (3, 3), (0.0, 0.0);
                                             density=1000.0)
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = 1.4particle_spacing
        surface_tension = SurfaceTensionMomentumMorris(;
                                                       surface_tension_coefficient=1.0)
        surface_normal_method = ColorfieldSurfaceNormal(; ideal_density_threshold=0.95)
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=7, clip_negative_pressure=false)
        system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                             smoothing_length,
                                             density_calculator=ContinuityDensity(),
                                             state_equation, pressure_acceleration=control,
                                             surface_tension, surface_normal_method,
                                             reference_particle_spacing=particle_spacing)
        @test system.pressure_acceleration_formulation === control

        clipped_state_equation = StateEquationCole(; sound_speed=10.0,
                                                   reference_density=1000.0,
                                                   exponent=7,
                                                   clip_negative_pressure=true)
        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               density_calculator=ContinuityDensity(),
                                                               state_equation=clipped_state_equation,
                                                               pressure_acceleration=control,
                                                               surface_tension,
                                                               surface_normal_method,
                                                               reference_particle_spacing=particle_spacing)
        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               density_calculator=ContinuityDensity(),
                                                               state_equation,
                                                               pressure_acceleration=control)

        # C-CSF provides the interface activity required by the TIC blend.
        ccsf_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                  smoothing_length,
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation,
                                                  pressure_acceleration=control,
                                                  surface_tension=SurfaceTensionMorris(;
                                                                                       surface_tension_coefficient=1.0),
                                                  surface_normal_method=CorrectedCSFSurfaceNormal(),
                                                  reference_particle_spacing=particle_spacing)
        @test ccsf_system.pressure_acceleration_formulation === control

        @test TrixiParticles.supports_interface_aware_tic(CorrectedCSFSurfaceNormal(),
                                                          SurfaceTensionMorris(;
                                                                               surface_tension_coefficient=1.0))
        @test !TrixiParticles.supports_interface_aware_tic(CorrectedCSFSurfaceNormal(),
                                                           SurfaceTensionMomentumMorris(;
                                                                                        surface_tension_coefficient=1.0))
        @test !TrixiParticles.supports_interface_aware_tic(nothing,
                                                           SurfaceTensionMorris(;
                                                                                surface_tension_coefficient=1.0))
    end
end
