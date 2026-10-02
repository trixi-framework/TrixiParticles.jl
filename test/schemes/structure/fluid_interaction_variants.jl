# Prescribed pair states isolate interaction formulas from density/pressure updates.
function structure_fluid_pair_state(; fluid_scheme=:wcsph, structure_kind=:tlsph,
                                    fluid_options=(;), distance=1.5, dimensions=2,
                                    boundary_density=AdamiPressureExtrapolation(),
                                    boundary_smoothing_length=1.0,
                                    boundary_viscosity=get(fluid_options, :viscosity,
                                                           nothing),
                                    fluid_pressure=500.0, monaghan_kajtar=false,
                                    coordinates=reshape([distance; zeros(dimensions - 1)],
                                                        dimensions, 1),
                                    parallelization_backend=SerialBackend())
    smoothing_kernel = SchoenbergCubicSplineKernel{dimensions}()
    smoothing_length = particle_spacing = 1.0
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1.0, clip_negative_pressure=false)
    velocity = reshape([1.0, 0.5, 0.7][1:dimensions], dimensions, 1)
    fluid_ic = InitialCondition(; coordinates=zeros(dimensions, 1), velocity,
                                mass=[1100.0], density=[1005.0], pressure=fluid_pressure,
                                particle_spacing)
    options = (; density_calculator=ContinuityDensity(), reference_particle_spacing=1.0,
               fluid_options...)
    fluid = if fluid_scheme == :wcsph
        WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                    state_equation, options...)
    else
        options = (; average_pressure_reduction=false, options...)
        EntropicallyDampedSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                    sound_speed=10.0, options...)
    end
    n = size(coordinates, 2)
    structure_velocity = repeat(reshape([0.25, -0.4, 0.2][1:dimensions], dimensions, 1), 1,
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
                                                 boundary_smoothing_length; state_equation,
                                                 viscosity=boundary_viscosity,
                                                 reference_particle_spacing=1.0)
    structure = structure_kind == :rigid ?
                RigidBodySystem(structure_ic; boundary_model, adhesion_coefficient=0.25) :
                TotalLagrangianSPHSystem(structure_ic; smoothing_kernel, smoothing_length,
                                         young_modulus=1.0e5, poisson_ratio=0.3,
                                         boundary_model)
    semi = Semidiscretization(fluid, structure; parallelization_backend)
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
        fluid.cache.correction_matrix[:, :, 1] .= Matrix{Float64}(I, dimensions, dimensions)
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
    force = -fluid.mass[1] * dv_fluid[1:ndims(fluid), 1]
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
            options = (; density_calculator=density, pressure_acceleration=nothing,
                       correction)
            scheme == :edac && (options = (; options..., average_pressure_reduction=true))
            state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
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
                state.v_fluid[1:2, 1] .= state.v_structure[1:2, 1] + direction * [0.75, 0.9]
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
        @test result.dv_structure[end, 1] ≈ 950.0 / 1005.0 * 1100.0 * (-0.75) * derivative
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
                @test_throws ArgumentError structure_fluid_pair_state(; fluid_scheme=scheme,
                                                                      structure_kind=kind,
                                                                      variant...)
                continue
            end
            state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                               variant...)
            result = test_structure_fluid_pair_balance(state)
            if get(variant, :boundary_density, nothing) isa ContinuityDensity
                @test result.dv_structure[end, 1] ≈
                      1100.0 * 0.75 * 0.75 * 0.5^2 * 10 / (7pi)
            end
        end
    end
    @testset "$scheme" for scheme in (:wcsph, :edac)
        test_structure_fluid_pair_balance(structure_fluid_pair_state(; fluid_scheme=scheme,
                                                                     monaghan_kajtar=true))
        @testset "$surface_tension" for surface_tension in
                                        (CohesionForceAkinci(), SurfaceTensionAkinci(),
                                         SurfaceTensionMorris(),
                                         SurfaceTensionMomentumMorris())
            state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=:rigid,
                                               fluid_options=(; surface_tension))
            test_structure_fluid_pair_balance(state)
        end
    end
end

@testset "Shifting momentum and distance regressions" begin
    shifting_variants = Any[nothing, ParticleShiftingTechniqueSun2017(),
                            TransportVelocityAdami(background_pressure=1000.0),
                            TransportVelocityAdami(background_pressure=1000.0,
                                                   modify_continuity_equation=true)]
    for update_everystage in (false, true),
        (modify_continuity_equation, second_continuity_equation_term) in
        ((false, nothing), (true, nothing), (true, ContinuityEquationTermSun2019())),
        momentum_equation_term in (nothing, MomentumEquationTermSun2019())

        push!(shifting_variants,
              ParticleShiftingTechnique(; update_everystage,
                                        modify_continuity_equation,
                                        second_continuity_equation_term,
                                        momentum_equation_term))
    end
    @testset "$scheme, $kind" for scheme in (:wcsph, :edac), kind in (:tlsph, :rigid)
        @testset "$shifting" for shifting in shifting_variants
            options = (; shifting_technique=shifting)
            state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
                                               fluid_options=options, fluid_pressure=0.0)
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
        end
        @testset "$distance" for distance in (0.0, 2.0e-8, 6.0e-8, 2.0, 2.1)
            state = structure_fluid_pair_state(; fluid_scheme=scheme, structure_kind=kind,
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
                                           dimensions=3, parallelization_backend=backend,
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
            TrixiParticles.apply_resultant_force_and_torque!(result.dv_structure, structure,
                                                             semi)
            @test structure.resultant_force[] ≈ result.force
            @test structure.resultant_torque[] ≈ expected_torque
        end
    end
end
