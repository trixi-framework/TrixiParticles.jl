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
        # Use unequal masses, densities, and non-opposite gradients to exercise every branch.
        m_a, m_b = 1.2, 0.8
        rho_a, rho_b = 1000.0, 980.0
        p_a, p_b = 2.0, 3.0
        W_a = SVector(0.2, -0.1)
        W_b = -W_a
        W_b_asymmetric = SVector(-0.13, 0.17)

        summation = TrixiParticles.pressure_acceleration_summation_density
        continuity = TrixiParticles.pressure_acceleration_continuity_density
        interparticle = TrixiParticles.inter_particle_averaged_pressure
        difference = TrixiParticles.pressure_acceleration_difference

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

        # Each asymmetric pair formulation reduces to its symmetric form and conserves momentum.
        for pressure_formulation in (summation, continuity, interparticle)
            # Asymmetric formulations are selected based on the configured correction and
            # must reduce to the symmetric formulation when a pair has `W_b == -W_a`.
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

            acceleration_a = pressure_formulation(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                  W_a, W_b_asymmetric)
            acceleration_b = pressure_formulation(m_b, m_a, rho_b, rho_a, p_b, p_a,
                                                  W_b_asymmetric, W_a)
            @test m_a * acceleration_a + m_b * acceleration_b ≈ zero(W_a) atol = eps()
        end

        # The asymmetric overload preserves Float32 inference and scalar type.
        result32 = @inferred interparticle(1.2f0, 0.8f0, 1000.0f0, 980.0f0,
                                           2.0f0, 3.0f0, SVector(0.2f0, -0.1f0),
                                           SVector(-0.13f0, 0.17f0))
        @test result32 isa SVector{2, Float32}
        @test tensile_instability_control(m_a, m_b, rho_a, rho_b, 0.0, 0.0,
                                          W_a) == zero(W_a)

        difference_acceleration = difference(m_b, rho_a, rho_b, p_a, p_b, W_a)
        conservative_acceleration = continuity(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                               W_a, W_b)
        model = SurfacePressureDifference()
        @test difference_acceleration ≈
              -m_b * (p_b - p_a) / (rho_a * rho_b) * W_a
        @test difference(m_b, rho_a, rho_b, p_a, p_a, W_a) == zero(W_a)
        @test TrixiParticles.blend_surface_pressure(model, conservative_acceleration,
                                                    0.0, true, m_b, rho_a, rho_b, p_a,
                                                    p_b, W_a) == conservative_acceleration
        @test TrixiParticles.blend_surface_pressure(model, conservative_acceleration,
                                                    1.0, true, m_b, rho_a, rho_b, p_a,
                                                    p_b, W_a) == difference_acceleration
        @test TrixiParticles.blend_surface_pressure(model, conservative_acceleration,
                                                    0.5, true, m_b, rho_a, rho_b, p_a,
                                                    p_b, W_a) ≈
              0.5 * (conservative_acceleration + difference_acceleration)
        @test TrixiParticles.blend_surface_pressure(model, conservative_acceleration,
                                                    1.0, false, m_b, rho_a, rho_b, p_a,
                                                    p_b, W_a) == conservative_acceleration
    end

    @testset "Surface pressure configuration" begin
        particle_spacing = 0.1
        initial_condition = RectangularShape(particle_spacing, (4, 4), (0.0, 0.0),
                                             density=1000.0)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        state_equation = StateEquationCole(sound_speed=10.0, reference_density=1000.0,
                                           exponent=1)
        surface_method_ = ColorfieldSurfaceDetection(ideal_density_threshold=0.9)
        model = SurfacePressureDifference()

        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length=0.15,
                                                               density_calculator=SummationDensity(),
                                                               state_equation,
                                                               gradient_correction=GradientCorrection(),
                                                               surface_method=surface_method_,
                                                               surface_pressure=:invalid,
                                                               reference_particle_spacing=particle_spacing)
        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length=0.15,
                                                               density_calculator=SummationDensity(),
                                                               state_equation,
                                                               gradient_correction=GradientCorrection(),
                                                               surface_pressure=model)
        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length=0.15,
                                                               density_calculator=SummationDensity(),
                                                               state_equation,
                                                               gradient_correction=KernelCorrection(),
                                                               surface_method=surface_method_,
                                                               surface_pressure=model,
                                                               reference_particle_spacing=particle_spacing)

        for correction in (GradientCorrection(), MixedKernelGradientCorrection())
            wcsph = WeaklyCompressibleSPHSystem(initial_condition;
                                                smoothing_kernel,
                                                smoothing_length=0.15,
                                                density_calculator=SummationDensity(),
                                                state_equation,
                                                gradient_correction=correction,
                                                surface_method=surface_method_,
                                                surface_pressure=model,
                                                reference_particle_spacing=particle_spacing)
            edac = EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                               smoothing_length=0.15, sound_speed=10.0,
                                               density_calculator=SummationDensity(),
                                               gradient_correction=correction,
                                               surface_method=surface_method_,
                                               surface_pressure=model,
                                               reference_particle_spacing=particle_spacing)
            @test wcsph.surface_pressure === model
            @test edac.surface_pressure === model
        end

        system = WeaklyCompressibleSPHSystem(initial_condition;
                                             smoothing_kernel,
                                             smoothing_length=0.15,
                                             density_calculator=SummationDensity(),
                                             state_equation,
                                             gradient_correction=GradientCorrection(),
                                             surface_method=surface_method_,
                                             surface_pressure=model,
                                             reference_particle_spacing=particle_spacing)
        semi = Semidiscretization(system; parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.01); reset_threads=false)
        v_ode, u_ode = ode.u0.x
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        coordinates = TrixiParticles.current_coordinates(u, system)
        particle, neighbor = 1, 2
        pos_diff = SVector{2}(coordinates[:, particle] - coordinates[:, neighbor])
        distance = norm(pos_diff)
        W_a = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance, particle)
        m_a = TrixiParticles.hydrodynamic_mass(system, particle)
        m_b = TrixiParticles.hydrodynamic_mass(system, neighbor)
        rho_a = TrixiParticles.current_density(v, system, particle)
        rho_b = TrixiParticles.current_density(v, system, neighbor)
        p_a, p_b = 2.0, 3.0
        gradient_correction = TrixiParticles.correction_gradient(system.correction)

        system.cache.surface_activity[particle] = 1
        acceleration = TrixiParticles.pressure_acceleration(system, system, particle,
                                                            neighbor, m_a, m_b, p_a, p_b,
                                                            rho_a, rho_b, pos_diff,
                                                            distance,
                                                            W_a, gradient_correction)
        @test acceleration ≈ TrixiParticles.pressure_acceleration_difference(m_b, rho_a,
                                                              rho_b, p_a,
                                                              p_b, W_a)
        W_b = TrixiParticles.smoothing_kernel_grad(system, -pos_diff, distance, neighbor)
        conservative_acceleration = TrixiParticles.pressure_acceleration(system, system,
                                                                         particle, neighbor,
                                                                         m_a, m_b, p_a, p_b,
                                                                         rho_a, rho_b,
                                                                         pos_diff, distance,
                                                                         W_a,
                                                                         gradient_correction;
                                                                         use_surface_pressure=false)
        @test conservative_acceleration ≈
              system.pressure_acceleration_formulation(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                                       W_a, W_b)
        constant_acceleration = TrixiParticles.pressure_acceleration(system, system,
                                                                     particle, neighbor,
                                                                     m_a, m_b, p_a, p_a,
                                                                     rho_a, rho_b, pos_diff,
                                                                     distance, W_a,
                                                                     gradient_correction)
        @test constant_acceleration == zero(W_a)
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

        # A locally symmetric custom formulation is insufficient if an enabled neighbor
        # requires asymmetric pair gradients.
        initial_condition = InitialCondition(; coordinates=zeros(2, 2),
                                             velocity=zeros(2, 2),
                                             mass=ones(2), density=fill(1000.0, 2))
        kernel = SchoenbergCubicSplineKernel{2}()
        asymmetric_system = EntropicallyDampedSPHSystem(initial_condition;
                                                        smoothing_kernel=kernel,
                                                        smoothing_length=1.0,
                                                        sound_speed=10.0,
                                                        gradient_correction=GradientCorrection())
        symmetric_system = EntropicallyDampedSPHSystem(initial_condition;
                                                       smoothing_kernel=kernel,
                                                       smoothing_length=1.0,
                                                       sound_speed=10.0,
                                                       pressure_acceleration=p_fun_1)
        error_str = "the pressure acceleration formulation of " *
                    "`EntropicallyDampedSPHSystem` must provide " *
                    "`m_a, m_b, rho_a, rho_b, p_a, p_b, W_a, W_b` when " *
                    "interacting with an asymmetric gradient correction"
        @test_throws ArgumentError(error_str) Semidiscretization(symmetric_system,
                                                                 asymmetric_system)
        @test_nowarn Semidiscretization(symmetric_system, asymmetric_system;
                                        interaction_matrix=Bool[true false; false true])
    end
end
