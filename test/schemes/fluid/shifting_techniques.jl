@testset verbose=true "Shifting Techniques" begin
    @testset "Constructors" begin
        @test_nowarn TransportVelocityAdami(background_pressure=1.0)
        @test_nowarn ParticleShiftingTechniqueSun2017()
        pst = @test_nowarn ParticleShiftingTechniqueSun2017(v_max_factor=1.2)
        @test pst.v_factor == 1.2
        @test_nowarn ConsistentShiftingSun2019()
        pst = @test_nowarn ConsistentShiftingSun2019(sound_speed_factor=0.2)
        @test pst.v_factor == 0.2

        # Can't use both `v_max_factor` and `sound_speed_factor`
        @test_throws ArgumentError ParticleShiftingTechnique(v_max_factor=1.0,
                                                             sound_speed_factor=0.5)
        # At least one of `v_max_factor` and `sound_speed_factor` must be positive
        @test_throws ArgumentError ParticleShiftingTechnique(v_max_factor=0.0,
                                                             sound_speed_factor=0.0)
        # Can't update every stage if not integrating shifting velocity
        @test_throws ArgumentError ParticleShiftingTechnique(integrate_shifting_velocity=false,
                                                             update_everystage=true)
        # Can't modify continuity equation if not integrating shifting velocity
        @test_throws ArgumentError ParticleShiftingTechnique(integrate_shifting_velocity=false,
                                                             modify_continuity_equation=true)
        # Can't modify momentum equation if not integrating shifting velocity
        @test_throws ArgumentError ParticleShiftingTechnique(integrate_shifting_velocity=false,
                                                             momentum_equation_term=MomentumEquationTermSun2019())
        # Can't use second continuity equation term if not modifying continuity equation
        @test_throws ArgumentError ParticleShiftingTechnique(integrate_shifting_velocity=true,
                                                             modify_continuity_equation=false,
                                                             second_continuity_equation_term=ContinuityEquationTermSun2019())
    end

    @testset "EDAC at a refined repulsive wall" begin
        # Both fluid particles are at x=h=1, so their mutual shift has no x component.
        # Integrate the normal derivative of the 2D cubic spline along x=0; only
        # the outer branch contributes (1 <= r/h <= 2).
        grad_normal(y) = 15 / (14pi) * (2 - hypot(1, y))^2 / hypot(1, y)
        density = 1000.0
        sound_speed = 10.0
        background_pressure = 1000.0
        shifting_technique = TransportVelocityAdami(; background_pressure)
        integral, _ = quadgk(grad_normal, -sqrt(3.0), sqrt(3.0))
        expected_shift = background_pressure / (8 * sound_speed) * 2 / density * integral

        for beta in (4, 8, 16)
            d = 1.0 / beta
            y = -4.0:d:4.0
            wall_ic = InitialCondition(coordinates=permutedims(hcat(zeros(length(y)), y)),
                                       density=density, particle_spacing=d)
            model = BoundaryModelMonaghanKajtar(10.0, beta, d, wall_ic.mass)
            wall = WallBoundarySystem(wall_ic, model)
            fluid_ic = InitialCondition(coordinates=[1.0 1.0; 0.0 1.5], density=density,
                                        pressure=[10.0, 30.0], particle_spacing=1.0)
            fluid = EntropicallyDampedSPHSystem(fluid_ic;
                                                smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                                smoothing_length=1.0, sound_speed,
                                                density_calculator=ContinuityDensity(),
                                                shifting_technique)
            semi = Semidiscretization(fluid, wall; parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.01))
            semi = ode.p.semi
            v_ode, u_ode = ode.u0.x
            vf = TrixiParticles.wrap_v(v_ode, fluid, semi)
            uf = TrixiParticles.wrap_u(u_ode, fluid, semi)
            vw = TrixiParticles.wrap_v(v_ode, wall, semi)
            uw = TrixiParticles.wrap_u(u_ode, wall, semi)

            function rhs_at_offset(offset)
                vf[3, :] .= fluid_ic.pressure .+ offset
                TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
                @test all(==(2), fluid.cache.neighbor_counter)
                @test all(==(20 + offset), fluid.cache.pressure_average)
                @test isapprox(TrixiParticles.delta_v(fluid, 1)[1], expected_shift;
                               rtol=5e-4)
                dv = zeros(size(vf))
                TrixiParticles.interact!(dv, vf, uf, vf, uf, fluid, fluid, semi)
                TrixiParticles.interact!(dv, vf, uf, vw, uw, fluid, wall, semi)
                return dv
            end

            @test rhs_at_offset(0.0) ≈ rhs_at_offset(300.0)

            # Only pressureless wall samples remain if fluid self-interaction is disabled.
            semi.interaction_matrix[1, 1] = false
            TrixiParticles.update_average_pressure!(fluid, fluid.average_pressure_reduction,
                                                    v_ode, u_ode, semi)
            @test all(iszero, fluid.cache.neighbor_counter)
            @test fluid.cache.pressure_average == vf[3, :]
        end
    end
end
