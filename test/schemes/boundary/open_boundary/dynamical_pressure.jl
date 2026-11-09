@testset verbose=true "Dynamical Pressure" begin
    # The following tests for linear and angular momentum and total energy conservation
    # are based on Sections 3.3.4 and 3.4.2 of
    # Daniel J. Price. "Smoothed Particle Hydrodynamics and Magnetohydrodynamics."
    # In: Journal of Computational Physics 231.3 (2012), pages 759–94.
    # https://doi.org/10.1016/j.jcp.2010.12.011
    @testset verbose=true "Momentum and Total Energy Conservation" begin
        # We are testing the momentum conservation of SPH with random initial configurations
        density_calculator = ContinuityDensity()

        particle_spacing = 0.1

        # The state equation is only needed to unpack `sound_speed`, so we can mock
        # it by using a `NamedTuple`.
        state_equation = (; sound_speed=0.0)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2 * particle_spacing
        search_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)

        # Run three times with different seed for the random initial condition
        for seed in 1:3
            # A larger number of particles will increase accumulated errors in the
            # summation. A larger tolerance has to be used for the tests below.
            ic = rectangular_patch(particle_spacing, (3, 3); seed)

            min_coords = vec(minimum(ic.coordinates, dims=2)) .- particle_spacing
            max_coords = vec(maximum(ic.coordinates, dims=2)) .+ particle_spacing
            bz = BoundaryZone(; boundary_face=(min_coords, [min_coords[1], max_coords[2]]),
                              boundary_type=OutFlow(), face_normal=[-1.0, 0.0],
                              open_boundary_layers=10, initial_condition=ic,
                              density=1.0, particle_spacing)
            bz.initial_condition.mass .= ic.mass

            system_wcsph = WeaklyCompressibleSPHSystem(ic; smoothing_kernel,
                                                       smoothing_length,
                                                       density_calculator,
                                                       state_equation)

            system_edac = EntropicallyDampedSPHSystem(ic; smoothing_kernel,
                                                      smoothing_length,
                                                      sound_speed=0.0,
                                                      pressure_acceleration=nothing,
                                                      density_calculator)

            open_boundary_wcsph = OpenBoundarySystem(bz; fluid_system=system_wcsph,
                                                     buffer_size=0,
                                                     boundary_model=BoundaryModelDynamicalPressureZhang())

            open_boundary_edac = OpenBoundarySystem(bz; fluid_system=system_edac,
                                                    buffer_size=0,
                                                    boundary_model=BoundaryModelDynamicalPressureZhang())

            n_particles = TrixiParticles.nparticles(open_boundary_edac)

            # Overwrite `system.pressure` because we skip the update step
            open_boundary_wcsph.cache.pressure .= ic.pressure

            # Since there is only one boundary zone, calling `update_boundary_zone_indices!` is unnecessary
            open_boundary_edac.boundary_zone_indices .= 1
            open_boundary_wcsph.boundary_zone_indices .= 1

            @testset "`OpenBoundarySystem` with `$(nameof(typeof(system.fluid_system)))`" for system in
                                                                                              (open_boundary_wcsph,
                                                                                               open_boundary_edac)
                u = ic.coordinates

                # Density is integrated with `ContinuityDensity`
                if system.fluid_system isa EntropicallyDampedSPHSystem
                    v = vcat(ic.velocity, ic.pressure', ic.density')
                else
                    v = vcat(ic.velocity, ic.density')
                end

                semi = DummySemidiscretization()

                # Result
                dv = zero(v)
                TrixiParticles.interact!(dv, v, u, v, u, system, system, semi)

                # Linear momentum conservation
                # ∑ m_a dv_a
                deriv_linear_momentum = sum(ic.mass' .* view(dv, 1:2, :), dims=2)

                @test isapprox(deriv_linear_momentum, zeros(2, 1), atol=5e-14)

                # Angular momentum conservation
                # m_a (r_a × dv_a)
                function deriv_angular_momentum(particle)
                    r_a = SVector(u[1, particle], u[2, particle], 0.0)
                    dv_a = SVector(dv[1, particle], dv[2, particle], 0.0)

                    return ic.mass[particle] * cross(r_a, dv_a)
                end

                # ∑ m_a (r_a × dv_a)
                deriv_angular_momentum_ = sum(deriv_angular_momentum, 1:n_particles)

                # Cross product is always 3-dimensional
                @test isapprox(deriv_angular_momentum_, zeros(3), atol=4e-15)

                # m_a (v_a ⋅ dv_a + dte_a),
                # where `te` is the thermal energy, called `u` in the Price paper.
                function deriv_energy(particle)
                    p_a = ic.pressure[particle]
                    rho_a = ic.density[particle]
                    dte_a = p_a / rho_a^2 * dv[end, particle]
                    v_a = TrixiParticles.extract_svector(v, system, particle)
                    dv_a = TrixiParticles.extract_svector(dv, system, particle)

                    return ic.mass[particle] * (dot(v_a, dv_a) + dte_a)
                end

                # ∑ m_a (v_a ⋅ dv_a + dte_a)
                deriv_total_energy = sum(deriv_energy, 1:n_particles)

                @test isapprox(deriv_total_energy, 0.0, atol=6e-15)
            end
        end
    end

    @testset verbose=true "integrated variables $(n_dims)D" for n_dims in (2, 3)
        particle_spacing = 0.1
        initial_condition = rectangular_patch(particle_spacing, ntuple(_ -> 2, n_dims))

        boundary_face = n_dims == 2 ? ([0.0, 0.0], [0.0, 1.0]) :
                        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0])
        face_normal = n_dims == 2 ? [1.0, 0.0] : [1.0, 0.0, 0.0]
        inflow = BoundaryZone(; boundary_face, boundary_type=InFlow(), face_normal,
                              open_boundary_layers=10, density=1.0, particle_spacing)

        system_wcsph = WeaklyCompressibleSPHSystem(initial_condition;
                                                   smoothing_kernel=SchoenbergCubicSplineKernel{n_dims}(),
                                                   smoothing_length=1,
                                                   density_calculator=ContinuityDensity(),
                                                   state_equation=nothing)

        open_boundary_wcsph = OpenBoundarySystem(inflow; fluid_system=system_wcsph,
                                                 buffer_size=0,
                                                 boundary_model=BoundaryModelDynamicalPressureZhang())

        # Integrated: velocity + density
        @test TrixiParticles.v_nvariables(open_boundary_wcsph) == n_dims + 1

        # EDAC with `SummationDensity`
        system_edac_1 = EntropicallyDampedSPHSystem(initial_condition;
                                                    smoothing_kernel=SchoenbergCubicSplineKernel{n_dims}(),
                                                    smoothing_length=1.0,
                                                    sound_speed=1.0)

        open_boundary_edac_1 = OpenBoundarySystem(inflow; fluid_system=system_edac_1,
                                                  buffer_size=0,
                                                  boundary_model=BoundaryModelDynamicalPressureZhang())

        # Integrated: velocity + pressure + density
        @test TrixiParticles.v_nvariables(open_boundary_edac_1) == n_dims + 2

        # EDAC with `ContinuityDensity`
        system_edac_2 = EntropicallyDampedSPHSystem(initial_condition;
                                                    smoothing_kernel=SchoenbergCubicSplineKernel{n_dims}(),
                                                    smoothing_length=1.0,
                                                    sound_speed=1.0,
                                                    density_calculator=ContinuityDensity())

        open_boundary_edac_2 = OpenBoundarySystem(inflow; fluid_system=system_edac_2,
                                                  buffer_size=0,
                                                  boundary_model=BoundaryModelDynamicalPressureZhang())

        # Integrated: velocity + pressure + density (`BoundaryModelDynamicalPressureZhang` always uses `ContinuityDensity`)
        @test TrixiParticles.v_nvariables(open_boundary_edac_2) == n_dims + 2
    end

    # A wall with `PressureMirroring` has no pressure of its own. Instead, each fluid
    # particle sees its own pressure at the wall. The buffer particles of the open
    # boundary are part of the fluid and must do the same, i.e., they must not read the
    # (unused) pressure stored in the wall.
    @testset verbose=true "Mirrored Wall Neighbor: $scheme" for scheme in ("WCSPH", "EDAC")
        particle_spacing = 0.1
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2 * particle_spacing
        sound_speed = 20.0
        fluid_density = 1020.0
        wall_density = 1000.0
        pressure = 8000.0

        # Fluid, buffer and wall particle, all within the kernel support of each other
        fluid_ic = InitialCondition(; coordinates=reshape([particle_spacing, 0.0], 2, 1),
                                    density=fluid_density, particle_spacing)
        buffer_ic = InitialCondition(; coordinates=zeros(2, 1), density=fluid_density,
                                     particle_spacing)
        wall_ic = InitialCondition(; coordinates=reshape([0.0, -particle_spacing], 2, 1),
                                   density=wall_density, particle_spacing)

        if scheme == "WCSPH"
            # The pressure of the fluid is `sound_speed^2 * (ρ - ρ_0) = pressure`
            state_equation = StateEquationCole(; sound_speed, exponent=1,
                                               reference_density=wall_density)
            fluid = WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel,
                                                smoothing_length, state_equation,
                                                density_calculator=ContinuityDensity())
        else
            state_equation = nothing
            fluid = EntropicallyDampedSPHSystem(fluid_ic; smoothing_kernel,
                                                smoothing_length, sound_speed,
                                                density_calculator=ContinuityDensity())
        end

        wall_model = BoundaryModelDummyParticles(wall_ic.density, wall_ic.mass,
                                                 PressureMirroring(), smoothing_kernel,
                                                 smoothing_length; state_equation)
        wall_system = WallBoundarySystem(wall_ic, wall_model)

        zone = BoundaryZone(; initial_condition=buffer_ic, boundary_type=InFlow(),
                            boundary_face=([0.5particle_spacing, -1.0],
                                           [0.5particle_spacing, 1.0]),
                            face_normal=[1.0, 0.0], open_boundary_layers=3,
                            density=fluid_density, particle_spacing,
                            reference_velocity=[0.0, 0.0])
        open_boundary_system = OpenBoundarySystem(zone; fluid_system=fluid, buffer_size=0,
                                                  boundary_model=BoundaryModelDynamicalPressureZhang())

        semi = Semidiscretization(fluid, wall_system, open_boundary_system;
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0))
        v_ode, u_ode = ode.u0.x
        _, wall, open_boundary = semi.systems
        v_open = TrixiParticles.wrap_v(v_ode, open_boundary, semi)
        u_open = TrixiParticles.wrap_u(u_ode, open_boundary, semi)
        v_wall = TrixiParticles.wrap_v(v_ode, wall, semi)
        u_wall = TrixiParticles.wrap_u(u_ode, wall, semi)

        # The pressure of the EDAC buffer particle is an integrated variable
        scheme == "EDAC" &&
            (TrixiParticles.current_pressure(v_open, open_boundary) .= pressure)
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

        p_a = TrixiParticles.current_pressure(v_open, open_boundary, 1)
        @test p_a ≈ pressure

        # Interaction of the buffer particle with the wall particle
        function interact_buffer_wall(wall_pressure)
            # The pressure stored in a mirrored wall is not used
            wall.boundary_model.pressure .= wall_pressure

            dv_buffer = zero(v_open)
            TrixiParticles.interact!(dv_buffer, v_open, u_open, v_wall, u_wall,
                                     open_boundary, wall, semi)

            return dv_buffer
        end

        dv = interact_buffer_wall(0.0)
        @test dv ≈ interact_buffer_wall(1234.0)
        @test dv ≈ interact_buffer_wall(pressure)

        # The pressure force equals the force of a wall particle with pressure `p_a`.
        # The buffer particle is at rest, so there is no viscosity and no continuity term.
        m_a = TrixiParticles.hydrodynamic_mass(open_boundary, 1)
        m_b = TrixiParticles.hydrodynamic_mass(wall, 1)
        pos_diff = SVector(0.0, particle_spacing)
        grad_kernel = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff,
                                                 particle_spacing, smoothing_length)
        if scheme == "WCSPH"
            # -m_b * (p_a + p_b) / (ρ_a * ρ_b) * ∇W_ab with `p_b = p_a`
            expected = -m_b * (p_a + p_a) / (fluid_density * wall_density) * grad_kernel
        else
            # -(V_a^2 + V_b^2) / m_a * p̃ * ∇W_ab, where `p̃ = p_a` for `p_b = p_a`
            volume_term = ((m_a / fluid_density)^2 + (m_b / wall_density)^2) / m_a
            expected = -volume_term * p_a * grad_kernel

            # The pressure difference between the buffer particle and the wall is zero,
            # so there is no pressure diffusion.
            @test dv[3, 1] == 0
        end
        @test dv[1:2, 1] ≈ expected
        @test !iszero(expected)
    end
end
