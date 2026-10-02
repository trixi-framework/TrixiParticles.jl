@testset verbose=true "Structure-fluid force balance" begin
    particle_spacing = 1.0
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.0
    reference_density = 1000.0
    structure_density = 2000.0
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density, exponent=1.0)
    viscosities = (nothing, ViscosityAdami(nu=0.1), ViscosityMorris(nu=0.1),
                   ArtificialViscosityMonaghan(alpha=0.1, beta=0.2))
    shifting_techniques = (nothing, ConsistentShiftingSun2019(),
                           TransportVelocityAdami(background_pressure=1000.0))
    configurations = ((:wcsph, nothing, AdamiPressureExtrapolation()),
                      (:wcsph, nothing, ContinuityDensity()),
                      (:wcsph, AkinciFreeSurfaceCorrection(reference_density),
                       AdamiPressureExtrapolation()),
                      (:wcsph, KernelCorrection(), SummationDensity()),
                      (:wcsph, MixedKernelGradientCorrection(), SummationDensity()),
                      (:edac, nothing, AdamiPressureExtrapolation()),
                      (:edac_pressure_reduction, nothing, AdamiPressureExtrapolation()))
    # Zero pressure isolates viscosity and shifting for approaching and receding particles.
    # Nonzero pressure checks that all momentum terms accumulate consistently.
    fluid_states = ((velocity=(1.0, 0.5), density=reference_density),
                    (velocity=(-1.0, -0.5), density=reference_density),
                    (velocity=(1.0, 0.5), density=1005.0))

    @testset "$fluid_scheme, $viscosity, $correction, $boundary_density, $shifting_technique" for shifting_technique in
                                                                                                  shifting_techniques,
                                                                                                  viscosity in
                                                                                                  viscosities,
                                                                                                  (fluid_scheme,
                                                                                                   correction,
                                                                                                   boundary_density) in
                                                                                                  configurations

        @testset "$structure_kind, $fluid_state" for structure_kind in
                                                     (:tlsph, :clamped_tlsph, :rigid),
                                                     fluid_state in fluid_states

            # `ContinuityDensity` is currently supported only for rigid structures.
            if structure_kind != :rigid && boundary_density isa ContinuityDensity
                continue
            end

            fluid_velocity = reshape(collect(fluid_state.velocity), 2, 1)
            fluid_ic = InitialCondition(; coordinates=reshape([0.0, 0.0], 2, 1),
                                        velocity=fluid_velocity,
                                        mass=[reference_density],
                                        density=[fluid_state.density],
                                        pressure=state_equation(fluid_state.density),
                                        particle_spacing)
            fluid_system = if fluid_scheme == :wcsph
                WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                            viscosity, correction, shifting_technique,
                                            density_calculator=ContinuityDensity(),
                                            state_equation)
            else
                EntropicallyDampedSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                            sound_speed=10.0, viscosity, correction,
                                            shifting_technique,
                                            density_calculator=ContinuityDensity(),
                                            average_pressure_reduction=fluid_scheme ==
                                                                       :edac_pressure_reduction)
            end

            structure_ic = InitialCondition(; coordinates=reshape([1.5, 0.0], 2, 1),
                                            velocity=zeros(2, 1), mass=[structure_density],
                                            density=[structure_density], particle_spacing)
            boundary_model = BoundaryModelDummyParticles([reference_density],
                                                         [reference_density],
                                                         boundary_density,
                                                         smoothing_kernel, smoothing_length;
                                                         state_equation, viscosity)
            structure_system = if structure_kind == :rigid
                RigidBodySystem(structure_ic; boundary_model)
            else
                clamped_particles = structure_kind == :clamped_tlsph ? (1:1) : (1:0)
                TotalLagrangianSPHSystem(structure_ic; smoothing_kernel, smoothing_length,
                                         young_modulus=1.0e5, poisson_ratio=0.3,
                                         clamped_particles, boundary_model)
            end

            semi = Semidiscretization(fluid_system, structure_system;
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.01))
            fluid, structure = ode.p.semi.systems
            v_ode, u_ode = ode.u0.x
            dv_ode = zero(v_ode)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
            v_fluid = TrixiParticles.wrap_v(v_ode, fluid, semi)
            u_fluid = TrixiParticles.wrap_u(u_ode, fluid, semi)
            v_structure = TrixiParticles.wrap_v(v_ode, structure, semi)
            u_structure = TrixiParticles.wrap_u(u_ode, structure, semi)
            # Isolate this pair from fluid self-interaction terms, which can be nonzero
            # with shifting and corrected kernel gradients.
            dv_fluid = zero(v_fluid)
            TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid,
                                     v_structure, u_structure, fluid, structure, semi)
            expected_force = -fluid.mass[1] * dv_fluid[1:2, 1]

            if !isnothing(shifting_technique)
                @test !iszero(TrixiParticles.delta_v(fluid, 1))
            end

            if correction isa KernelCorrection ||
               correction isa MixedKernelGradientCorrection
                # Ensure that this fixture exercises a gradient that is not odd.
                pos_diff = SVector(1.5, 0.0)
                gradient = TrixiParticles.smoothing_kernel_grad_unsafe(fluid, pos_diff,
                                                                       1.5, 1)
                reverse_gradient = TrixiParticles.smoothing_kernel_grad_unsafe(fluid,
                                                                               -pos_diff,
                                                                               1.5, 1)
                @test !isapprox(gradient, -reverse_gradient)
            end

            if fluid_state.density == reference_density
                @test iszero(TrixiParticles.current_pressure(v_fluid, fluid, 1))
                @test iszero(structure.boundary_model.pressure[1])
                if isnothing(shifting_technique)
                    if isnothing(viscosity) ||
                       (viscosity isa ArtificialViscosityMonaghan &&
                        fluid_state.velocity[1] < 0)
                        @test iszero(expected_force)
                    else
                        # The viscous reaction on the structure follows the fluid motion.
                        @test expected_force[1] * fluid_state.velocity[1] > 0
                    end
                elseif isnothing(viscosity)
                    # A nonzero reaction with no pressure or viscosity exercises shifting.
                    @test !iszero(expected_force)
                end
            elseif fluid_scheme == :edac_pressure_reduction && isnothing(viscosity)
                # Uniform nonzero pressure produces no force after pressure reduction.
                @test TrixiParticles.current_pressure(v_fluid, fluid, 1) ≈
                      structure.boundary_model.pressure[1]
                if isnothing(shifting_technique)
                    @test isapprox(expected_force, zeros(2); atol=sqrt(eps()))
                else
                    @test !iszero(expected_force)
                end
            else
                @test !iszero(expected_force)
            end

            if boundary_density isa ContinuityDensity
                # Independently evaluate d(rho_s)/dt = (rho_s/rho_f) m_f (v_s-v_f) ⋅ ∇_s W.
                # Cubic-spline derivative at r/h = 1.5 in 2D, with h = 1.
                kernel_derivative = -0.75 * (2 - 1.5)^2 * 10 / (7pi)
                expected_density_rate = -reference_density / fluid_state.density *
                                        fluid.mass[1] * fluid_state.velocity[1] *
                                        kernel_derivative
                dv_structure = TrixiParticles.wrap_v(dv_ode, structure, semi)
                @test dv_structure[end, 1] ≈ expected_density_rate
            end

            if structure isa RigidBodySystem
                @test isapprox(structure.resultant_force[], expected_force;
                               rtol=sqrt(eps()), atol=sqrt(eps()))
            else
                # Include clamped particles when checking the pair reaction directly.
                dv_structure_fluid = zeros(eltype(structure), ndims(structure),
                                           nparticles(structure))
                TrixiParticles.interact!(dv_structure_fluid, v_structure, u_structure,
                                         v_fluid, u_fluid, structure, fluid, semi;
                                         eachparticle=eachparticle(structure))
                @test isapprox(structure.mass[1] * dv_structure_fluid[:, 1], expected_force;
                               rtol=sqrt(eps()), atol=sqrt(eps()))

                if structure_kind == :tlsph
                    dv_structure = TrixiParticles.wrap_v(dv_ode, structure, semi)
                    @test isapprox(structure.mass[1] * dv_structure[1:2, 1], expected_force;
                                   rtol=sqrt(eps()), atol=sqrt(eps()))
                end
            end
        end
    end
end
