@testset "Gradient and blended corrections" begin
    identity_matrix = Matrix{Float64}(I, 2, 2)
    linear_field(pos) = 2.0 + 3.0 * pos[1] - 2.0 * pos[2]
    exact_gradient = [3.0, -2.0]

    @test_throws ArgumentError BlendedGradientCorrection(-0.1)
    @test_throws ArgumentError BlendedGradientCorrection(1.1)

    for correction in (GradientCorrection(), BlendedGradientCorrection(0.4)),
        edac in (false, true)
        setup = correction_setup(correction; edac, pressure_acceleration=nothing)
        update_correction!(setup)
        @test all(isfinite, setup.system.cache.correction_matrix)
    end

    for perturbation in (false, true)
        raw_setup = update_correction!(correction_setup(nothing; perturbation))
        raw_moments = correction_moments(raw_setup; field=linear_field)

        gradient_setup = update_correction!(correction_setup(GradientCorrection();
                                                             perturbation))
        gradient_moments = correction_moments(gradient_setup; field=linear_field)
        @test maximum(particle -> norm(gradient_moments.first_gradient_moment[:, :,
                                                                              particle] -
                                       identity_matrix),
                      TrixiParticles.eachparticle(gradient_setup.system)) < 2e-12
        @test maximum(particle -> norm(gradient_moments.difference_gradient[:, particle] -
                                       exact_gradient),
                      TrixiParticles.eachparticle(gradient_setup.system)) < 5e-12

        blending_factor = 0.4
        blended_setup = update_correction!(correction_setup(BlendedGradientCorrection(blending_factor);
                                                            perturbation))
        blended_moments = correction_moments(blended_setup; field=linear_field)
        expected = (1 - blending_factor) * raw_moments.first_gradient_moment
        for particle in TrixiParticles.eachparticle(blended_setup.system)
            expected[:, :, particle] .+= blending_factor * identity_matrix
        end
        @test maximum(abs, blended_moments.first_gradient_moment - expected) < 2e-12

        corner = corner_particle(raw_setup.system)
        @test norm(raw_moments.first_gradient_moment[:, :, corner] - identity_matrix) > 1e-2
    end

    density32 = fill(1000.0f0, 4)
    mass32 = fill(10.0f0, 4)
    state_equation32 = StateEquationCole(; sound_speed=10.0f0,
                                         reference_density=1000.0f0, exponent=1)
    for correction in (GradientCorrection(), BlendedGradientCorrection(0.4f0),
         MixedKernelGradientCorrection())
        boundary = BoundaryModelDummyParticles(density32, mass32, SummationDensity(),
                                               WendlandC6Kernel{2}(), 0.2f0;
                                               state_equation=state_equation32, correction)
        @test eltype(boundary.cache.correction_matrix) == Float32
        if hasproperty(boundary.cache, :dw_gamma)
            @test eltype(boundary.cache.dw_gamma) == Float32
        end
    end

    @testset "scale-independent inversion" begin
        cases = ((Float32, 2, 1.0f-30),
                 (Float32, 2, 1.0f20),
                 (Float32, 3, 1.0f-16),
                 (Float32, 3, 1.0f13),
                 (Float64, 2, 1.0e-200),
                 (Float64, 3, 1.0e150))
        for (ELTYPE, NDIMS, scale) in cases
            inverse = invert_scaled_correction_matrix(ELTYPE, Val(NDIMS), scale)
            expected = Matrix{ELTYPE}(I, NDIMS, NDIMS) / scale
            @test all(isfinite, inverse)
            @test inverse ≈ expected rtol = 10eps(ELTYPE)
        end

        # The normalized matrix is valid, but rescaling its inverse overflows.
        inverse = invert_scaled_correction_matrix(Float32, Val(1), 1.0f-39)
        @test inverse == ones(Float32, 1, 1)

        # The entry scale squared is finite, but this matrix's determinant overflows.
        scale = 1.4f19
        matrix = Float32[scale scale; -scale scale]
        inverse = invert_correction_matrix(matrix)
        expected = Float32[0.5 -0.5; 0.5 0.5] / scale
        @test all(isfinite, inverse)
        @test inverse ≈ expected rtol = 10eps(Float32)
    end

    particle_spacing = 0.25
    particles = RectangularShape(particle_spacing, (4, 4, 4), (0.0, 0.0, 0.0);
                                 density=1000.0)
    state_equation = StateEquationCole(; sound_speed=10.0,
                                       reference_density=1000.0, exponent=1)
    system = WeaklyCompressibleSPHSystem(particles;
                                         smoothing_kernel=WendlandC6Kernel{3}(),
                                         smoothing_length=2particle_spacing,
                                         density_calculator=ContinuityDensity(),
                                         state_equation,
                                         correction=GradientCorrection())
    semi = Semidiscretization(system; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    v_ode = Array(ode.u0.x[1])
    u_ode = Array(ode.u0.x[2])
    system = first(ode.p.semi.systems)
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)
    v = TrixiParticles.wrap_v(v_ode, system, ode.p.semi)
    u = TrixiParticles.wrap_u(u_ode, system, ode.p.semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    first_moment = zeros(3, 3)
    GC.@preserve v_ode u_ode begin
        TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates,
                                              ode.p.semi;
                                              points=1:1) do particle,
                                                             neighbor,
                                                             pos_diff,
                                                             distance
            volume = TrixiParticles.hydrodynamic_mass(system, neighbor) /
                     TrixiParticles.current_density(v, system, neighbor)
            gradient = TrixiParticles.smoothing_kernel_grad(system, SVector(pos_diff),
                                                            distance, particle)
            for j in 1:3, i in 1:3
                first_moment[i, j] -= volume * gradient[i] * pos_diff[j]
            end
        end
    end
    @test first_moment ≈ Matrix{Float64}(I, 3, 3) atol = 3e-12

    for y_offset in (0.0, 1.0e-12)
        coordinates = [0.0 0.1 0.2; 0.0 y_offset 0.0]
        initial = InitialCondition(; coordinates, velocity=zeros(2, 3),
                                   density=fill(1000.0, 3), particle_spacing=0.1)
        system = WeaklyCompressibleSPHSystem(initial;
                                             smoothing_kernel=WendlandC6Kernel{2}(),
                                             smoothing_length=0.2,
                                             density_calculator=ContinuityDensity(),
                                             state_equation,
                                             correction=GradientCorrection())
        semi = Semidiscretization(system; parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        v_ode = Array(ode.u0.x[1])
        u_ode = Array(ode.u0.x[2])
        system = first(ode.p.semi.systems)
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)
        for particle in TrixiParticles.eachparticle(system)
            @test TrixiParticles.correction_matrix(system, particle) == I
        end
    end

    analytic_density_rate = -2000.0
    errors = Dict{Any, Float64}()
    for correction in (nothing, GradientCorrection(), BlendedGradientCorrection(0.4))
        setup = correction_setup(correction)
        dv_ode = zero(setup.v_ode)
        TrixiParticles.kick!(dv_ode, setup.v_ode, setup.u_ode,
                             (; semi=setup.semi, split_integration_data=nothing), 0.0)
        dv = TrixiParticles.wrap_v(dv_ode, setup.system, setup.semi)
        error = dv[end, :] .- analytic_density_rate
        errors[correction] = sqrt(sum(abs2, error) / length(error))
    end
    @test errors[GradientCorrection()] < 2e-10
    @test errors[BlendedGradientCorrection(0.4)] < errors[nothing]
    @test errors[nothing] > 1.0

    for correction in (GradientCorrection(), BlendedGradientCorrection(0.4)),
        edac in (false, true),
        density_calculator in (SummationDensity(), ContinuityDensity())
        result = correction_restart_result(correction; edac, density_calculator)
        @test result.state_equal
        @test result.rhs_equal
        @test result.cache_finite
    end
end
