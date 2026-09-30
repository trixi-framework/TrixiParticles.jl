@testset verbose=true "Structure-fluid force balance" begin
    particle_spacing = 1.0
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.0
    reference_density = 1000.0
    structure_density = 2000.0
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density, exponent=1.0)
    viscosities = (nothing, ViscosityAdami(nu=0.1), ViscosityMorris(nu=0.1),
                   ArtificialViscosityMonaghan(alpha=0.1))
    configurations = ((nothing, AdamiPressureExtrapolation()),
                      (nothing, ContinuityDensity()),
                      (KernelCorrection(), SummationDensity()),
                      (MixedKernelGradientCorrection(), SummationDensity()))
    # Zero pressure isolates viscosity for approaching and receding particles.
    # Nonzero pressure checks that pressure and viscous reactions accumulate consistently.
    fluid_states = ((velocity=(1.0, 0.5), density=reference_density),
                    (velocity=(-1.0, -0.5), density=reference_density),
                    (velocity=(1.0, 0.5), density=1005.0))

    @testset "$viscosity, $correction, $boundary_density" for viscosity in viscosities,
                                                              (correction,
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
                                        density=[fluid_state.density], particle_spacing)
            fluid_system = WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel,
                                                       smoothing_length, viscosity,
                                                       correction,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation)

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
            dv_fluid = TrixiParticles.wrap_v(dv_ode, fluid, semi)
            expected_force = -fluid.mass[1] * dv_fluid[1:2, 1]

            if !isnothing(correction)
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
                @test iszero(fluid.pressure[1])
                @test iszero(structure.boundary_model.pressure[1])
                if isnothing(viscosity) ||
                   (viscosity isa ArtificialViscosityMonaghan &&
                    fluid_state.velocity[1] < 0)
                    @test iszero(expected_force)
                else
                    # The viscous reaction on the structure follows the fluid motion.
                    @test expected_force[1] * fluid_state.velocity[1] > 0
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
                v_structure = TrixiParticles.wrap_v(v_ode, structure, semi)
                u_structure = TrixiParticles.wrap_u(u_ode, structure, semi)
                v_fluid = TrixiParticles.wrap_v(v_ode, fluid, semi)
                u_fluid = TrixiParticles.wrap_u(u_ode, fluid, semi)
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
