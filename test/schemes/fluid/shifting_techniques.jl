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
        # Can't correct properties if integrating shifting velocity
        @test_throws ArgumentError ParticleShiftingTechnique(correct_properties=true)
        @test_throws ArgumentError ConsistentShiftingSun2019(correct_properties=true)
        @test_nowarn ParticleShiftingTechniqueSun2017(correct_properties=true)
    end

    @testset verbose=true "Property Correction" begin
        particle_spacing = 0.1
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = 1.5 * particle_spacing
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=1)
        dt = 0.01

        function create_fluid(velocity, shifting_technique; edac=false,
                              density_calculator=ContinuityDensity())
            fluid = RectangularShape(particle_spacing, (15, 15), (0.0, 0.0);
                                     density=1000.0, velocity)
            if edac
                return EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length, sound_speed=10.0,
                                                   density_calculator,
                                                   average_pressure_reduction=false,
                                                   shifting_technique)
            end

            return WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                               density_calculator, state_equation,
                                               shifting_technique)
        end

        # Return `v` and `u` before and after applying the shifting in the callback
        function apply_shifting(system, systems...; delta_v=nothing)
            semi = Semidiscretization(system, systems...)
            ode = semidiscretize(semi, (0.0, 1.0))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

            # `wrap_v` and `wrap_u` return unsafe pointer arrays
            GC.@preserve v_ode u_ode begin
                v0 = copy(TrixiParticles.wrap_v(v_ode, system, semi))
                u0 = copy(TrixiParticles.wrap_u(u_ode, system, semi))
                shifting = system.shifting_technique

                if isnothing(delta_v)
                    TrixiParticles.update_shifting_from_callback!(system, shifting, v_ode,
                                                                  u_ode, semi)
                else
                    # Prescribe a constant shifting velocity
                    system.cache.delta_v .= delta_v
                end
                TrixiParticles.correct_properties!(v_ode, u_ode, shifting, system, semi, dt)
                TrixiParticles.apply_particle_shifting!(u_ode, shifting, system, semi, dt)

                v = copy(TrixiParticles.wrap_v(v_ode, system, semi))
                u = copy(TrixiParticles.wrap_u(u_ode, system, semi))
            end

            return v0, u0, v, u
        end

        pst = ParticleShiftingTechniqueSun2017(correct_properties=true)
        gradient = [1.0 2.0; -3.0 -1.0]
        velocity(x) = gradient * x
        delta_v = [0.3, -0.2]

        # Particles with full kernel support
        is_inside(x) = all(coord -> 0.35 < coord < 1.15, x)

        density_calculators = (ContinuityDensity(), SummationDensity())
        @testset "Linear Velocity Field, EDAC: $edac" for edac in (false, true)
            @testset "$density_calculator" for density_calculator in density_calculators
                system = create_fluid(velocity, pst; edac, density_calculator)
                v0, u0, v, u = apply_shifting(system; delta_v)
                inside = findall(is_inside, eachcol(u0))
                @test length(inside) == 64

                # The corrected velocity is the velocity field at the shifted positions
                change = v[1:2, inside] - v0[1:2, inside]
                expected = dt * gradient * delta_v
                @test isapprox(change, repeat(expected, 1, 64), rtol=0.02)
                @test isapprox(v[1:2, inside], gradient * u[:, inside], rtol=1e-3)

                # All other properties are unchanged
                @test v[3:end, :] == v0[3:end, :]
            end
        end

        @testset "Computed Shifting Velocity" begin
            # The shifting velocity is nonzero at the surface of the fluid
            system = create_fluid(velocity, pst)
            v0, u0, v, u = apply_shifting(system)
            @test !iszero(system.cache.delta_v)
            @test v[1:2, :] != v0[1:2, :]
            @test u ≈ u0 + dt * system.cache.delta_v

            # Without the correction, the positions are the same, but the velocity is kept
            system = create_fluid(velocity, ParticleShiftingTechniqueSun2017())
            v0_, u0_, v_, u_ = apply_shifting(system)
            @test !haskey(system.cache, :velocity_correction)
            @test v_ == v0_
            @test u_ == u
        end

        @testset "Walls" begin
            function shift_towards_wall(viscosity)
                system = create_fluid((1.0, 0.0), pst)
                # Wall below the fluid
                wall = RectangularShape(particle_spacing, (15, 4), (0.0, -0.4);
                                        density=1000.0)
                boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                             AdamiPressureExtrapolation(),
                                                             smoothing_kernel,
                                                             smoothing_length;
                                                             state_equation, viscosity)
                boundary = WallBoundarySystem(wall, boundary_model)

                return apply_shifting(system, boundary; delta_v=[0.0, -0.3])
            end

            # Free-slip walls don't change the uniform velocity
            v0, u0, v, u = shift_towards_wall(nothing)
            @test v[1:2, :] == v0[1:2, :]

            # No-slip walls slow down the particles that are shifted towards the wall
            v0, u0, v, u = shift_towards_wall(ViscosityAdami(nu=0.01))
            first_row = findall(y -> y < 0.1, u0[2, :])
            @test length(first_row) == 15
            @test all(v[1, first_row] .< v0[1, first_row])
            # Particles that don't see the wall are unchanged
            far = findall(y -> y > 0.4, u0[2, :])
            @test v[1:2, far] == v0[1:2, far]
        end
    end
end
