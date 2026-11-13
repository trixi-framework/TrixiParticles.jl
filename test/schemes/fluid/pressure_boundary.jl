@testset verbose=true "WCSPH Boundary Pressure Consistency" begin
    # A fluid at uniform, nonzero pressure must remain at rest in a closed vessel.
    # Different fluid and boundary corrections expose spurious pressure forces caused
    # by using the boundary's corrected gradient for the extrapolated pressure.
    function test_constant_pressure(; density_calculator=SummationDensity(),
                                    fluid_correction, boundary_correction,
                                    boundary_type=WallBoundarySystem,
                                    pressure_extrapolation=AdamiPressureExtrapolation())
        particle_spacing = 0.1
        fluid_size = (5, 5)
        boundary_layers = 3
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = sqrt(2) * particle_spacing
        density = 1000.0
        background_pressure = 10_000.0
        state_equation = StateEquationCole(; sound_speed=50.0, reference_density=density,
                                           exponent=1, background_pressure)

        # Surround the fluid patch on all sides. Three boundary layers fill the
        # kernel support, so missing neighbors cannot create a free surface.
        fluid_initial_condition = RectangularShape(particle_spacing, fluid_size,
                                                   (0.0, 0.0); density)
        outer_size = fluid_size .+ 2 * boundary_layers
        boundary_origin = -boundary_layers * particle_spacing
        outer_initial_condition = RectangularShape(particle_spacing, outer_size,
                                                   (boundary_origin, boundary_origin);
                                                   density)
        boundary_initial_condition = setdiff(outer_initial_condition,
                                             fluid_initial_condition)

        fluid = WeaklyCompressibleSPHSystem(fluid_initial_condition; smoothing_kernel,
                                            smoothing_length, state_equation,
                                            density_calculator,
                                            gradient_correction=fluid_correction)
        boundary_model = BoundaryModelDummyParticles(boundary_initial_condition.density,
                                                     boundary_initial_condition.mass,
                                                     pressure_extrapolation,
                                                     smoothing_kernel, smoothing_length;
                                                     state_equation,
                                                     gradient_correction=boundary_correction)

        if boundary_type === WallBoundarySystem
            boundary = WallBoundarySystem(boundary_initial_condition, boundary_model)
        elseif boundary_type === RigidBodySystem
            boundary = RigidBodySystem(boundary_initial_condition; boundary_model,
                                       particle_spacing)
        else
            # Disable elastic forces to isolate the hydrodynamic reaction force.
            boundary = TotalLagrangianSPHSystem(boundary_initial_condition;
                                                smoothing_kernel, smoothing_length,
                                                young_modulus=0.0, poisson_ratio=0.0,
                                                boundary_model)
        end

        semi = Semidiscretization(fluid, boundary; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)

        # The semidiscretization may copy the TLSPH system for its reference neighbor
        # search. Read and update the systems owned by the ODE.
        semi = ode.p.semi
        fluid, boundary = semi.systems
        v, u = ode.u0.x

        # Fluid density, extrapolated boundary density and correction moments depend
        # on one another. Repeat the initialization at fixed coordinates so updates
        # propagate between them before evaluating a single RHS, without time stepping.
        for _ in 1:3
            TrixiParticles.update_systems_and_nhs(v, u, semi, 0.0)
        end

        dv = zero(v)
        TrixiParticles.kick!(dv, v, u, ode.p, 0.0)
        dv_fluid = TrixiParticles.wrap_v(dv, fluid, semi)
        acceleration_fluid = view(dv_fluid, 1:2, :)

        # First verify that pressure is uniform and nonzero. Zero pressure would
        # trivially produce zero pressure forces even with an incorrect gradient.
        pressure_min, pressure_max = extrema(fluid.pressure)
        @test pressure_min >= background_pressure - 1.0
        @test pressure_max - pressure_min < 1e-6

        # Constant pressure must produce no fluid acceleration, independently of the
        # boundary correction. The full RHS, including density evolution, must be finite.
        @test maximum(abs, acceleration_fluid) < 1e-8
        @test all(isfinite, dv)

        if boundary isa TrixiParticles.AbstractStructureSystem
            force_fluid = vec(sum(fluid.mass' .* acceleration_fluid; dims=2))

            # Rigid bodies store per-particle forces; TLSPH stores accelerations,
            # which must be multiplied by the material mass to obtain forces.
            if boundary isa RigidBodySystem
                forces_boundary = boundary.force_per_particle
            else
                forces_boundary = boundary.mass' .*
                                  TrixiParticles.wrap_v(dv, boundary, semi)
            end
            force_boundary = vec(sum(forces_boundary; dims=2))

            # Newton's third law requires equal and opposite total forces. Scale by
            # per-particle force magnitudes since their resultant can be nearly zero.
            force_scale = max(sum(norm, eachcol(forces_boundary)), 1.0)
            @test norm(force_fluid + force_boundary) / force_scale < 1e-12
        end

        return nothing
    end

    @testset "Boundary Systems" begin
        density_calculators = (SummationDensity(), ContinuityDensity())
        pressure_extrapolations = (AdamiPressureExtrapolation(),
                                   BernoulliPressureExtrapolation())

        # Use a different correction for walls and elastic structures; uncorrected
        # rigid-body particles also cover the absence of a boundary-side correction.
        boundary_configurations = ((WallBoundarySystem, KernelCorrection()),
                                   (RigidBodySystem, nothing),
                                   (TotalLagrangianSPHSystem, KernelCorrection()))

        # Exercise both density formulations and extrapolation methods for every
        # boundary system.
        for density_calculator in density_calculators,
            pressure_extrapolation in pressure_extrapolations,
            (boundary_type, boundary_correction) in boundary_configurations
            density_name = nameof(typeof(density_calculator))
            pressure_name = nameof(typeof(pressure_extrapolation))

            @testset "$boundary_type, $density_name, $pressure_name" begin
                test_constant_pressure(; density_calculator,
                                       fluid_correction=MixedKernelGradientCorrection(),
                                       boundary_correction, boundary_type,
                                       pressure_extrapolation)
            end
        end
    end

    @testset "Fluid Corrections" begin
        # A corrected wall must not disturb equilibrium for any fluid correction.
        # Zero blending must also recover the uncorrected fluid behavior.
        fluid_corrections = (nothing, KernelCorrection(), GradientCorrection(),
                             BlendedGradientCorrection(0.0),
                             MixedKernelGradientCorrection())
        @testset "$fluid_correction" for fluid_correction in fluid_corrections
            test_constant_pressure(; fluid_correction,
                                   boundary_correction=MixedKernelGradientCorrection())
        end
    end

    @testset "Uncorrected Boundary" begin
        # Conversely, corrected fluid must remain at rest next to an uncorrected wall.
        test_constant_pressure(; fluid_correction=KernelCorrection(),
                               boundary_correction=nothing)
    end
end
