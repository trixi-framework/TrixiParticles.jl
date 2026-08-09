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

    @testset verbose=true "Interface-Aware Tensile Instability Control" begin
        control = InterfaceAwareTensileInstabilityControl()
        @test control.strength == 1.0
        @test InterfaceAwareTensileInstabilityControl(; strength=0.25).strength == 0.25
        for strength in (0, -1, 1.1, Inf, NaN, "invalid")
            @test_throws ArgumentError InterfaceAwareTensileInstabilityControl(; strength)
        end

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
        interface_aware(activity_a, activity_b,
                        strength=1.0) = TrixiParticles.interface_aware_tensile_acceleration(m_a,
                                                                                            m_b,
                                                                                            rho_a,
                                                                                            rho_b,
                                                                                            p_a,
                                                                                            p_b,
                                                                                            grad_kernel,
                                                                                            activity_a,
                                                                                            activity_b,
                                                                                            strength)

        @test interface_aware(0.0, 0.0) == controlled
        @test interface_aware(1.0, 0.0) == standard
        @test interface_aware(0.0, 1.0) == standard
        @test interface_aware(0.5, 0.0) == (standard + controlled) / 2
        @test interface_aware(0.0, 0.5) == (standard + controlled) / 2
        @test interface_aware(0.0, 0.0, 0.25) ==
              standard + 0.25 * (controlled - standard)
        @test interface_aware(-1.0, -0.5) == controlled
        @test interface_aware(2.0, 0.0) == standard
        @test interface_aware(NaN, 0.0) == standard
        @test interface_aware(Inf, 0.0) == standard

        colorfield = ColorfieldSurfaceNormal(; ideal_density_threshold=0.95)
        css = SurfaceTensionMomentumMorris(; surface_tension_coefficient=1.0)
        morris = SurfaceTensionMorris(; surface_tension_coefficient=1.0)
        ccsf = CorrectedCSFSurfaceNormal()
        @test TrixiParticles.supports_interface_aware_tic(colorfield, css)
        @test TrixiParticles.supports_interface_aware_tic(colorfield, morris)
        @test TrixiParticles.supports_interface_aware_tic(ccsf, morris)
        @test !TrixiParticles.supports_interface_aware_tic(ccsf, css)
        @test !TrixiParticles.supports_interface_aware_tic(nothing, morris)

        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=1000.0,
                                           exponent=7,
                                           clip_negative_pressure=false)
        clipped_state_equation = StateEquationCole(; sound_speed=10.0,
                                                   reference_density=1000.0,
                                                   exponent=7,
                                                   clip_negative_pressure=true)
        validate(density_calculator, equation, normal_method, surface_tension,
                 correction=nothing) = TrixiParticles.validate_interface_aware_tic(control,
                                                                                   density_calculator,
                                                                                   equation,
                                                                                   normal_method,
                                                                                   surface_tension,
                                                                                   correction)
        @test_nowarn validate(ContinuityDensity(), state_equation, colorfield, css)
        @test_nowarn validate(ContinuityDensity(), state_equation, colorfield, css,
                              AkinciFreeSurfaceCorrection(1000.0))
        @test_throws ArgumentError validate(SummationDensity(), state_equation,
                                            colorfield, css)
        @test_throws ArgumentError validate(ContinuityDensity(), clipped_state_equation,
                                            colorfield, css)
        @test_throws ArgumentError validate(ContinuityDensity(), state_equation,
                                            colorfield, css, KernelCorrection())
        @test_throws ArgumentError validate(ContinuityDensity(), state_equation,
                                            nothing, css)

        @test TrixiParticles.choose_pressure_acceleration_formulation(control,
                                                                      ContinuityDensity(),
                                                                      2, Float64,
                                                                      nothing) === control
        @test_throws ArgumentError TrixiParticles.choose_pressure_acceleration_formulation(control,
                                                                                           SummationDensity(),
                                                                                           2,
                                                                                           Float64,
                                                                                           nothing)
        @test_throws ArgumentError TrixiParticles.choose_pressure_acceleration_formulation(control,
                                                                                           ContinuityDensity(),
                                                                                           2,
                                                                                           Float64,
                                                                                           GradientCorrection())

        particle_spacing = 0.1
        initial_condition = RectangularShape(particle_spacing, (3, 3), (0.0, 0.0);
                                             density=1000.0)
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = 1.4particle_spacing
        system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                             smoothing_length,
                                             density_calculator=ContinuityDensity(),
                                             state_equation,
                                             pressure_acceleration=control,
                                             surface_tension=css,
                                             surface_normal_method=colorfield,
                                             reference_particle_spacing=particle_spacing)
        @test system.pressure_acceleration_formulation === control
        system_data = Dict{String, Any}()
        @test_nowarn TrixiParticles.add_system_data!(system_data, system)
        @test system_data["pressure_acceleration_formulation"] ==
              :InterfaceAwareTensileInstabilityControl
        @test system_data["interface_aware_tic_strength"] == 1.0

        system.cache.interface_activity .= 0.0
        system.cache.interface_activity[2] = 0.5
        actual = TrixiParticles.pressure_acceleration(system, system, 1, 2,
                                                      m_a, m_b, p_a, p_b,
                                                      rho_a, rho_b,
                                                      SVector(0.1, 0.0), 0.1,
                                                      grad_kernel, nothing)
        @test actual == interface_aware(0.0, 0.5)

        boundary_result = TrixiParticles.evaluate_pressure_acceleration(control, system,
                                                                        nothing, 1, 1,
                                                                        m_a, m_b,
                                                                        rho_a, rho_b,
                                                                        p_a, p_b,
                                                                        grad_kernel)
        @test boundary_result == standard

        neighbor_without_interface = WeaklyCompressibleSPHSystem(initial_condition;
                                                                 smoothing_kernel,
                                                                 smoothing_length,
                                                                 density_calculator=ContinuityDensity(),
                                                                 state_equation)
        unsupported_fluid_result = TrixiParticles.evaluate_pressure_acceleration(control,
                                                                                 system,
                                                                                 neighbor_without_interface,
                                                                                 1, 1,
                                                                                 m_a, m_b,
                                                                                 rho_a,
                                                                                 rho_b,
                                                                                 p_a, p_b,
                                                                                 grad_kernel)
        @test unsupported_fluid_result == standard

        matrix_pressure_a = TrixiParticles.SMatrix{2, 2}(1.0, 0.0, 0.0, 2.0)
        matrix_pressure_b = TrixiParticles.SMatrix{2, 2}(0.5, 0.0, 0.0, 1.0)
        matrix_result = TrixiParticles.evaluate_pressure_acceleration(control, system,
                                                                      system, 1, 2,
                                                                      m_a, m_b,
                                                                      rho_a, rho_b,
                                                                      matrix_pressure_a,
                                                                      matrix_pressure_b,
                                                                      grad_kernel)
        @test matrix_result ==
              TrixiParticles.pressure_acceleration_continuity_density(m_a, m_b,
                                                                      rho_a, rho_b,
                                                                      matrix_pressure_a,
                                                                      matrix_pressure_b,
                                                                      grad_kernel)

        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               density_calculator=ContinuityDensity(),
                                                               state_equation,
                                                               pressure_acceleration=control)

        edac = EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                           smoothing_length, sound_speed=10.0,
                                           density_calculator=ContinuityDensity(),
                                           pressure_acceleration=control,
                                           surface_tension=css,
                                           surface_normal_method=colorfield,
                                           reference_particle_spacing=particle_spacing)
        @test edac.pressure_acceleration_formulation === control
        @test_throws ArgumentError EntropicallyDampedSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               sound_speed=10.0,
                                                               pressure_acceleration=control,
                                                               surface_tension=css,
                                                               surface_normal_method=colorfield,
                                                               reference_particle_spacing=particle_spacing)

        ccsf_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                  smoothing_length,
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation,
                                                  pressure_acceleration=control,
                                                  surface_tension=morris,
                                                  surface_normal_method=ccsf,
                                                  reference_particle_spacing=particle_spacing)
        @test ccsf_system.pressure_acceleration_formulation === control
    end
end
