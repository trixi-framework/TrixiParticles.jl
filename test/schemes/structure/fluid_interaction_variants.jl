using Logging: NullLogger, with_logger

@testset verbose=true "Structure-fluid interaction variants" begin
    # Prescribed pair states isolate interaction formulas from density/pressure updates.
    function structure_fluid_pair_state(; fluid_scheme=:wcsph, structure_kind=:tlsph,
                                        fluid_options=(;), distance=1.5, dimensions=2,
                                        boundary_density=AdamiPressureExtrapolation(),
                                        boundary_smoothing_length=1.0,
                                        boundary_correction=nothing,
                                        structure_smoothing_length=1.0,
                                        structure_smoothing_kernel=SchoenbergCubicSplineKernel{dimensions}(),
                                        boundary_viscosity=get(fluid_options, :viscosity,
                                                               nothing),
                                        fluid_pressure=500.0, monaghan_kajtar=false,
                                        coordinates=reshape([distance;
                                                             zeros(dimensions - 1)],
                                                            dimensions, 1),
                                        parallelization_backend=SerialBackend(),
                                        neighborhood_search=GridNeighborhoodSearch{dimensions}(),
                                        neighborhood_search_handler=SharedNHSHandler)
        smoothing_kernel = SchoenbergCubicSplineKernel{dimensions}()
        smoothing_length = particle_spacing = 1.0
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=1.0, clip_negative_pressure=false)
        velocity = reshape([1.0, 0.5, 0.7][1:dimensions], dimensions, 1)
        fluid_ic = InitialCondition(; coordinates=zeros(dimensions, 1), velocity,
                                    mass=[1100.0], density=[1005.0],
                                    pressure=fluid_pressure,
                                    particle_spacing)
        options = (; density_calculator=ContinuityDensity(), reference_particle_spacing=1.0,
                   fluid_options...)
        fluid = if fluid_scheme == :wcsph
            WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                        state_equation, options...)
        elseif fluid_scheme == :edac
            options = (; average_pressure_reduction=false, options...)
            EntropicallyDampedSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                        sound_speed=10.0, options...)
        else
            ImplicitIncompressibleSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                            reference_density=1005.0, time_step=0.001,
                                            viscosity=get(fluid_options, :viscosity,
                                                          nothing))
        end
        n = size(coordinates, 2)
        structure_velocity = repeat(reshape([0.25, -0.4, 0.2][1:dimensions], dimensions, 1),
                                    1,
                                    n)
        structure_ic = InitialCondition(; coordinates, velocity=structure_velocity,
                                        mass=2100.0 .+ 300.0 .* (0:(n - 1)),
                                        density=fill(2000.0, n), particle_spacing)
        hydrodynamic_mass = 700.0 .+ 50.0 .* (0:(n - 1))
        boundary_model = monaghan_kajtar ?
                         BoundaryModelMonaghanKajtar(10.0, 1.0, particle_spacing,
                                                     hydrodynamic_mass) :
                         BoundaryModelDummyParticles(fill(950.0, n), hydrodynamic_mass,
                                                     boundary_density, smoothing_kernel,
                                                     boundary_smoothing_length;
                                                     state_equation,
                                                     viscosity=boundary_viscosity,
                                                     correction=boundary_correction,
                                                     reference_particle_spacing=1.0)
        structure = if structure_kind == :rigid
            RigidBodySystem(structure_ic; boundary_model, adhesion_coefficient=0.25)
        elseif structure_kind == :wall
            WallBoundarySystem(structure_ic, boundary_model)
        else
            TotalLagrangianSPHSystem(structure_ic;
                                     smoothing_kernel=structure_smoothing_kernel,
                                     smoothing_length=structure_smoothing_length,
                                     young_modulus=1.0e5, poisson_ratio=0.3,
                                     boundary_model)
        end
        semi = with_logger(NullLogger()) do
            Semidiscretization(fluid, structure;
                               parallelization_backend, neighborhood_search,
                               neighborhood_search_handler)
        end
        ode = semidiscretize(semi, (0.0, 0.01); reset_threads=false)
        fluid, structure = semi.systems
        v_ode, u_ode = ode.u0.x
        v_fluid = TrixiParticles.wrap_v(v_ode, fluid, semi)
        u_fluid = TrixiParticles.wrap_u(u_ode, fluid, semi)
        v_structure = TrixiParticles.wrap_v(v_ode, structure, semi)
        u_structure = TrixiParticles.wrap_u(u_ode, structure, semi)
        TrixiParticles.current_density(v_fluid, fluid) .= 1005.0
        TrixiParticles.current_pressure(v_fluid, fluid) .= fluid_pressure
        if !monaghan_kajtar
            structure.boundary_model.pressure .= 230.0
            if !isnothing(boundary_viscosity)
                structure.boundary_model.cache.wall_velocity .= 2 .* structure_velocity .-
                                                                velocity
            end
        end
        # Nontrivial cache values also exercise non-odd gradients and tensor pressure terms.
        for (field, value) in ((:delta_v, [0.4, -0.3, 0.2][1:dimensions]),
             (:dw_gamma, [0.1, -0.2, 0.15][1:dimensions]),
             (:kernel_correction_coefficient, 1.3), (:pressure_average, 120.0))
            haskey(fluid.cache, field) && (getproperty(fluid.cache, field) .= value)
        end
        if haskey(fluid.cache, :correction_matrix)
            fluid.cache.correction_matrix[:, :,
                                          1] .= Matrix{Float64}(I, dimensions, dimensions)
            fluid.cache.correction_matrix[1, 2, 1] = 0.3
        end
        if structure isa RigidBodySystem
            TrixiParticles.update_final!(structure, v_structure, u_structure,
                                         v_ode, u_ode, semi, 0.0)
        end
        return (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure, u_structure)
    end

    function test_structure_fluid_pair_balance(state)
        (; fluid, structure, semi, v_fluid, u_fluid, v_structure, u_structure) = state
        dv_fluid, dv_structure = zero(v_fluid), zero(v_structure)
        TrixiParticles.reset_interaction_caches!(structure)
        TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                 fluid, structure, semi)
        TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid, u_fluid,
                                 structure, fluid, semi)
        # Zero shifting velocities isolate physical traction without relying on the
        # shared implementation of that operator. Restore the prescribed shifts before
        # testing the structural load, which must be independent of transport terms.
        dv_physical = dv_fluid
        if haskey(fluid.cache, :delta_v)
            shifting_velocity = copy(fluid.cache.delta_v)
            fluid.cache.delta_v .= 0
            dv_physical = zero(v_fluid)
            TrixiParticles.interact!(dv_physical, v_fluid, u_fluid, v_structure,
                                     u_structure,
                                     fluid, structure, semi)
            fluid.cache.delta_v .= shifting_velocity
        end
        force = -fluid.mass[1] * dv_physical[1:ndims(fluid), 1]
        forces = structure isa RigidBodySystem ? copy(structure.force_per_particle) :
                 dv_structure[1:ndims(structure), :] .* reshape(structure.mass, 1, :)
        @test isapprox(vec(sum(forces; dims=2)), force; rtol=1e-12, atol=1e-12)
        @test all(isfinite, dv_fluid) && all(isfinite, dv_structure)
        return (; dv_fluid, dv_structure, force, forces)
    end

    @testset "Analytical pressure and continuity" begin
        @testset "$scheme, $kind" for scheme in (:wcsph, :edac), kind in (:tlsph, :rigid)
            for density in (SummationDensity(), ContinuityDensity()),
                correction in (nothing, KernelCorrection()),
                boundary in
                (AdamiPressureExtrapolation(), PressureMirroring(), ContinuityDensity())

                kind == :tlsph && boundary isa ContinuityDensity && continue
                # EDAC correction-cache lifecycle is covered by PR #1218.
                scheme == :edac && !isnothing(correction) && continue
                options = (; density_calculator=density, pressure_acceleration=nothing,
                           correction)
                scheme == :edac &&
                    (options = (; options..., average_pressure_reduction=true))
                state = structure_fluid_pair_state(; fluid_scheme=scheme,
                                                   structure_kind=kind,
                                                   fluid_options=options,
                                                   boundary_density=boundary)
                grad_s = [-15 / (56pi), 0.0] # Cubic-spline derivative at r/h = 1.5.
                grad_f = isnothing(correction) ? -grad_s :
                         (-grad_s - 5 / (112pi) * [0.1, -0.2]) / 1.3
                p_avg = scheme == :edac ? 120.0 : 0.0
                p_f = 500.0 - p_avg
                p_s = (boundary isa PressureMirroring ? 500.0 : 230.0) - p_avg
                expected_force = 1100.0 * 700.0 *
                                 (density isa SummationDensity ?
                                  p_f / 1005.0^2 * grad_f - p_s / 950.0^2 * grad_s :
                                  (p_f * grad_f - p_s * grad_s) / (1005.0 * 950.0))
                for direction in (-1, 0, 1)
                    state.v_fluid[1:2,
                                  1] .= state.v_structure[1:2, 1] + direction * [0.75, 0.9]
                    result = test_structure_fluid_pair_balance(state)
                    @test result.forces[:, 1] ≈ expected_force
                    if density isa ContinuityDensity
                        @test result.dv_fluid[end, 1] ≈
                              1005.0 / 950.0 * 700.0 * direction * dot([0.75, 0.9], grad_f)
                    end
                    if boundary isa ContinuityDensity
                        rho_factor = density isa ContinuityDensity ? 950.0 / 1005.0 : 1.0
                        @test result.dv_structure[end, 1] ≈
                              rho_factor * 1100.0 * direction * dot([-0.75, -0.9], grad_s)
                    end
                end
                if scheme == :edac && boundary isa PressureMirroring
                    (; fluid, ode, semi) = state
                    TrixiParticles.update_average_pressure!(fluid,
                                                            fluid.average_pressure_reduction,
                                                            ode.u0.x[1], ode.u0.x[2], semi)
                    @test fluid.cache.pressure_average[1] == 500.0
                end
            end
        end
        # Density uses the boundary kernel even when the fluid skips the pressure pair.
        for (h, distance) in ((1.0, 2.0e-8), (2.0, 2.5))
            state = structure_fluid_pair_state(; structure_kind=:rigid, distance,
                                               boundary_density=ContinuityDensity(),
                                               boundary_smoothing_length=h)
            result = test_structure_fluid_pair_balance(state)
            q = distance / h
            derivative = (q < 1 ? -3q + 2.25q^2 : -0.75 * (2 - q)^2) * 10 / (7pi * h^3)
            @test iszero(result.force)
            @test result.dv_structure[end, 1] ≈
                  950.0 / 1005.0 * 1100.0 * (-0.75) * derivative
        end
    end

    @testset "Additional structure-fluid variants" begin
        variants = NamedTuple[(; boundary_viscosity=viscosity,
                               fluid_options=(; viscosity=ViscosityAdami(nu=0.4)))
                              for viscosity in (nothing, ViscosityAdamiSGS(nu=0.1),
                                   ViscosityMorrisSGS(nu=0.1),
                                   ViscosityCarreauYasuda(nu0=0.1, nu_inf=0.01,
                                                          lambda=1.0, a=2.0, n=0.5))]
        append!(variants,
                [(; fluid_pressure=-500.0, fluid_options=(; pressure_acceleration))
                 for pressure_acceleration in
                     (TrixiParticles.pressure_acceleration_summation_density,
                      TrixiParticles.inter_particle_averaged_pressure,
                      tensile_instability_control)])
        append!(variants,
                [(; boundary_density,
                  fluid_options=(; density_calculator=SummationDensity()))
                 for boundary_density in (BernoulliPressureExtrapolation(),
                      PressureMirroring(), PressureZeroing(),
                      ContinuityDensity())])
        append!(variants,
                [(; boundary_density=SummationDensity(),
                  fluid_options=(; correction, shifting_technique,
                                 density_calculator=SummationDensity(),
                                 pressure_acceleration=TrixiParticles.pressure_acceleration_summation_density))
                 for correction in (ShepardKernelCorrection(), GradientCorrection(),
                      BlendedGradientCorrection(0.5)),
                     shifting_technique in (ConsistentShiftingSun2019(),
                      TransportVelocityAdami(background_pressure=1000.0))])
        @testset "$scheme, $kind" for scheme in (:wcsph, :edac), kind in (:tlsph, :rigid)
            @testset "$variant" for variant in variants
                if kind == :tlsph &&
                   get(variant, :boundary_density, nothing) isa ContinuityDensity
                    @test_throws ArgumentError structure_fluid_pair_state(;
                                                                          fluid_scheme=scheme,
                                                                          structure_kind=kind,
                                                                          variant...)
                    continue
                end
                scheme == :edac &&
                    !isnothing(get(get(variant, :fluid_options, (;)), :correction, nothing)) &&
                    continue
                state = structure_fluid_pair_state(; fluid_scheme=scheme,
                                                   structure_kind=kind,
                                                   variant...)
                result = test_structure_fluid_pair_balance(state)
                if get(variant, :boundary_density, nothing) isa ContinuityDensity
                    @test result.dv_structure[end, 1] ≈
                          1100.0 * 0.75 * 0.75 * 0.5^2 * 10 / (7pi)
                end
            end
        end
        @testset "$scheme" for scheme in (:wcsph, :edac)
            test_structure_fluid_pair_balance(structure_fluid_pair_state(;
                                                                         fluid_scheme=scheme,
                                                                         monaghan_kajtar=true))
            @testset "$surface_tension" for surface_tension in
                                            (CohesionForceAkinci(), SurfaceTensionAkinci(),
                                             SurfaceTensionMorris(),
                                             SurfaceTensionMomentumMorris())
                state = structure_fluid_pair_state(; fluid_scheme=scheme,
                                                   structure_kind=:rigid,
                                                   fluid_options=(; surface_tension))
                test_structure_fluid_pair_balance(state)
            end
        end
    end

    @testset "Shifting momentum and distance regressions" begin
        # Representative momentum operators instead of a Cartesian product of
        # callback and continuity options, which do not affect structural traction.
        shifting_variants = (nothing, ParticleShiftingTechniqueSun2017(),
                             ConsistentShiftingSun2019(),
                             ParticleShiftingTechnique(momentum_equation_term=MomentumEquationTermSun2019()),
                             TransportVelocityAdami(background_pressure=1000.0),
                             TransportVelocityAdami(background_pressure=1000.0,
                                                    modify_continuity_equation=true))
        @testset "$scheme, $kind" for scheme in (:wcsph, :edac), kind in (:tlsph, :rigid)
            @testset "$shifting" for shifting in shifting_variants
                options = (; shifting_technique=shifting)
                state = structure_fluid_pair_state(; fluid_scheme=scheme,
                                                   structure_kind=kind,
                                                   fluid_options=options,
                                                   fluid_pressure=0.0)
                state.structure.boundary_model.pressure .= 0.0
                result = test_structure_fluid_pair_balance(state)
                # Analytical Sun2019/Adami momentum terms for delta_v_structure = 0.
                factor = 0.0
                if shifting isa ParticleShiftingTechnique &&
                   !isnothing(shifting.momentum_equation_term)
                    factor = 2 * 700.0 / 950.0
                elseif shifting isa TransportVelocityAdami
                    factor = scheme == :wcsph ? -700.0 / 950.0 :
                             -((1100.0 / 1005.0)^2 + (700.0 / 950.0)^2) / 1100.0 *
                             1005.0 * 950.0 / (1005.0 + 950.0)
                end
                grad = [0.75 * 0.5^2 * 10 / (7pi), 0.0]
                expected = factor * state.v_fluid[1:2, 1] *
                           dot(TrixiParticles.delta_v(state.fluid, 1), grad)
                @test result.dv_fluid[1:2, 1] ≈ expected
                @test iszero(result.force)
                @test all(iszero, result.forces)
            end
            @testset "$distance" for distance in (0.0, 2.0e-8, 6.0e-8, 2.0, 2.1)
                state = structure_fluid_pair_state(; fluid_scheme=scheme,
                                                   structure_kind=kind,
                                                   distance)
                result = test_structure_fluid_pair_balance(state)
                skipped = distance == 0 || distance >= 2 ||
                          (scheme == :wcsph && distance == 2.0e-8)
                @test iszero(result.force) == skipped
            end
        end
    end
    @testset "3D structure-fluid forces and rigid torque" begin
        @testset "$scheme, $kind, $backend" for scheme in (:wcsph, :edac),
                                                kind in (:tlsph, :rigid),
                                                backend in
                                                (SerialBackend(), PolyesterBackend())

            options = (; viscosity=ViscosityAdami(nu=0.1),
                       shifting_technique=ConsistentShiftingSun2019())
            state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                               dimensions=3,
                                               parallelization_backend=backend,
                                               coordinates=[1.2 1.4 1.6; -0.4 0.5 0.1;
                                                            0.2 -0.3 0.6],
                                               fluid_options=options)
            result = test_structure_fluid_pair_balance(state)
            if kind == :rigid
                (; structure, semi) = state
                expected_torque = sum(cross(state.u_structure[:, i] -
                                            structure.center_of_mass[],
                                            result.forces[:, i])
                                      for i in eachparticle(structure))
                TrixiParticles.apply_resultant_force_and_torque!(result.dv_structure,
                                                                 structure,
                                                                 semi)
                @test structure.resultant_force[] ≈ result.force
                @test structure.resultant_torque[] ≈ expected_torque
            end
        end
    end

    @testset "Unequal fluid and boundary supports" begin
        @testset "$kind, $handler, $h" for kind in (:tlsph, :rigid),
                                           handler in (PairsNHSHandler, SharedNHSHandler),
                                           h in (0.5, 1.0, 2.0)
            state = structure_fluid_pair_state(; structure_kind=kind,
                                               boundary_smoothing_length=h,
                                               neighborhood_search_handler=handler)
            result = test_structure_fluid_pair_balance(state)
            @test !iszero(result.force)
            @test TrixiParticles.compact_support(state.structure, state.fluid) ==
                  max(2h, 2.0)
        end
        # The enlarged reaction search must not extend the boundary continuity kernel.
        state = structure_fluid_pair_state(; structure_kind=:rigid,
                                           boundary_smoothing_length=0.5,
                                           boundary_density=ContinuityDensity())
        result = test_structure_fluid_pair_balance(state)
        @test !iszero(result.force)
        @test iszero(result.dv_structure[end, 1])
    end

    @testset "Hydrodynamic versus elastic TLSPH kernels" begin
        for correction in (KernelCorrection(), MixedKernelGradientCorrection())
            options = (; correction, pressure_acceleration=nothing)
            state = structure_fluid_pair_state(; fluid_options=options,
                                               structure_smoothing_length=0.4,
                                               structure_smoothing_kernel=WendlandC2Kernel{2}())
            result = test_structure_fluid_pair_balance(state)
            grad_s = [-15 / (56pi), 0.0]
            grad_f = (-grad_s - 5 / (112pi) * [0.1, -0.2]) / 1.3
            # Mixed correction applies the prescribed, non-diagonal fluid matrix.
            correction isa MixedKernelGradientCorrection &&
                (grad_f = [1.0 0.3; 0.0 1.0] * grad_f)
            expected = 1100.0 * 700.0 / (1005.0 * 950.0) * (500.0 * grad_f - 230.0 * grad_s)
            @test result.forces[:, 1] ≈ expected
            @test iszero(TrixiParticles.smoothing_kernel_grad(state.structure,
                                                              SVector(1.5, 0.0), 1.5, 1))
            @test TrixiParticles.hydrodynamic_kernel_grad(state.structure,
                                                          SVector(1.5, 0.0), 1.5, 1) ≈
                  grad_s
        end
    end

    @testset "TVF does not duplicate Monaghan-Kajtar repulsion" begin
        for (scheme, kind) in ((:wcsph, :tlsph), (:wcsph, :wall), (:edac, :tlsph))
            tvf = TransportVelocityAdami(background_pressure=1000.0)
            state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                               monaghan_kajtar=true,
                                               fluid_options=(; shifting_technique=tvf))
            (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure,
             u_structure) = state
            fluid.cache.delta_v .= 0
            dv_fluid = zero(v_fluid)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            # Independent Monaghan-Kajtar repulsion at q = 1.5, K = 10, beta = 1,
            # boundary spacing = 1. Even with zero shifting, the old TVF doubled it.
            repulsion = -10.0 / 0.5 * (1.77 / 32) *
                        (1 + 2.5 * 1.5 + 2 * 1.5^2) * 0.5^5
            @test dv_fluid[1:2, 1] ≈ [repulsion, 0.0]

            TrixiParticles.update_shifting!(fluid, tvf, v_fluid, u_fluid,
                                            ode.u0.x[1], ode.u0.x[2], semi)
            grad_f = [15 / (56pi), 0.0]
            volume_term = ((1100.0 / 1005.0)^2 + 1.0) / 1100.0
            expected_shift = -1000.0 / (8 * 10.0) *
                             (scheme == :wcsph ? 2 / 1005.0 : volume_term) * grad_f
            @test TrixiParticles.delta_v(fluid, 1) ≈ expected_shift

            # The TVF tensor contribution stays in the fluid and does not change the
            # structural repulsion. This also exercises nonzero transport velocities.
            if kind == :tlsph
                result = test_structure_fluid_pair_balance(state)
                @test result.forces[:, 1] ≈ -1100.0 * [repulsion, 0.0]
                @test result.dv_fluid[1, 1] != repulsion
            end
        end
    end

    @testset "Boundary-model correction context" begin
        for kind in (:tlsph, :rigid),
            correction in (KernelCorrection(),
             GradientCorrection(),
             MixedKernelGradientCorrection())

            state = structure_fluid_pair_state(; structure_kind=kind,
                                               structure_smoothing_length=0.4,
                                               boundary_density=PressureMirroring(),
                                               boundary_correction=correction,
                                               fluid_options=(;
                                                              correction=KernelCorrection(),
                                                              pressure_acceleration=nothing))
            (; structure, v_structure, u_structure, ode, semi) = state
            model = structure.boundary_model
            for field in (:dw_gamma, :kernel_correction_coefficient, :correction_matrix)
                haskey(model.cache, field) && fill!(getproperty(model.cache, field), NaN)
            end
            TrixiParticles.update_boundary_interpolation!(structure, v_structure,
                                                          u_structure,
                                                          ode.u0.x[1], ode.u0.x[2], semi,
                                                          0.0)
            for field in (:dw_gamma, :kernel_correction_coefficient, :correction_matrix)
                haskey(model.cache, field) &&
                    @test all(isfinite, getproperty(model.cache, field))
            end

            grad_s = [-15 / (56pi), 0.0]
            if correction isa KernelCorrection ||
               correction isa MixedKernelGradientCorrection
                # Use the boundary kernel for every visited neighbor. TLSPH's self
                # search includes the particle itself; a rigid body without a
                # contact model has an empty rigid-to-rigid search.
                gamma = 1100.0 / 1005.0 * 5 / (112pi)
                kind == :tlsph && (gamma += 700.0 / 950.0 * 10 / (7pi))
                dw_gamma = 1100.0 / 1005.0 * grad_s / gamma
                @test model.cache.kernel_correction_coefficient[1] ≈ gamma
                @test model.cache.dw_gamma[:, 1] ≈ dw_gamma
                grad_s = (grad_s - 5 / (112pi) * dw_gamma) / gamma
            end
            # A one-neighbor layout is rank deficient, so gradient correction uses I.
            grad_f = ([15 / (56pi), 0.0] - 5 / (112pi) * [0.1, -0.2]) / 1.3
            expected = 1100.0 * 700.0 * 500.0 / (1005.0 * 950.0) * (grad_f - grad_s)
            result = test_structure_fluid_pair_balance(state)
            @test result.forces[:, 1] ≈ expected
        end
    end

    @testset "EDAC pressure mirroring at walls" begin
        for average_pressure_reduction in (false, true)
            state = structure_fluid_pair_state(; fluid_scheme=:edac, structure_kind=:wall,
                                               boundary_density=PressureMirroring(),
                                               fluid_options=(; average_pressure_reduction))
            (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure,
             u_structure) = state
            TrixiParticles.update_average_pressure!(fluid, fluid.average_pressure_reduction,
                                                    ode.u0.x[1], ode.u0.x[2], semi)
            p_avg = average_pressure_reduction ? 500.0 : 0.0
            @test TrixiParticles.average_pressure(fluid, 1) == p_avg
            dv_fluid = zero(v_fluid)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                                     fluid, structure, semi)
            grad_f = [15 / (56pi), 0.0]
            volume_term = ((1100.0 / 1005.0)^2 + (700.0 / 950.0)^2) / 1100.0
            @test dv_fluid[1:2, 1] ≈ -volume_term * (500.0 - p_avg) * grad_f
            # Mirroring eliminates pressure diffusion, leaving only the artificial EOS.
            @test dv_fluid[3, 1] ≈ 700.0 * 1005.0 / 950.0 * 10.0^2 * dot([1.0, 0.5], grad_f)
        end
    end

    @testset "IISPH pair pressure and viscosity" begin
        # Test the pair RHS with prescribed pressure. The IISPH pressure solver does
        # not yet support structural neighbors, so no coupled kick/solve is attempted.
        for kind in (:tlsph, :rigid), boundary in (PressureMirroring(), PressureZeroing())
            state = structure_fluid_pair_state(; fluid_scheme=:iisph, structure_kind=kind,
                                               boundary_density=boundary,
                                               fluid_options=(;
                                                              viscosity=ViscosityAdami(nu=0.1)))
            result = test_structure_fluid_pair_balance(state)
            @test !iszero(result.force)
        end
    end
end # Structure-fluid interaction variants
