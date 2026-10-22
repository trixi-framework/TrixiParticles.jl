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

            # The boundary pressure term does not conserve momentum, so we set the pressure
            # of the boundary zone to zero (it is otherwise taken from `ic`).
            bz.initial_condition.pressure .= 0

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

    @testset verbose=true "Rest Density of Recycled Particles" begin
        particle_spacing = 0.1
        fluid_ic = rectangular_patch(particle_spacing, (4, 4), perturbation_factor=0.0,
                                     offset=(0.5, 0.5))

        # Two boundary zones with different rest densities.
        inflow = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 1.0]),
                              face_normal=(1.0, 0.0), open_boundary_layers=2,
                              density=1.0, particle_spacing)
        outflow = BoundaryZone(; boundary_face=([1.0, 0.0], [1.0, 1.0]),
                               face_normal=(-1.0, 0.0), open_boundary_layers=2,
                               density=2.0, particle_spacing)

        # Without an equation of state, recycled particles get the rest density of the zone.
        fluid_system = EntropicallyDampedSPHSystem(fluid_ic;
                                                   smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                                   smoothing_length=1.2 * particle_spacing,
                                                   sound_speed=1.0, buffer_size=2)
        system = OpenBoundarySystem(inflow, outflow; fluid_system, buffer_size=1,
                                    boundary_model=BoundaryModelDynamicalPressureZhang())

        n_inflow = nparticles(inflow.initial_condition)
        system.boundary_zone_indices[1:n_inflow] .= 1
        system.boundary_zone_indices[(n_inflow + 1):(end - 1)] .= 2

        # The first particle of each zone has crossed the boundary face into the fluid domain.
        crossed_particles = ((zone_id=1, particle=1, x_new=0.05),
                             (zone_id=2, particle=n_inflow + 1, x_new=0.95))
        @testset "Zone $zone_id" for (; zone_id, particle, x_new) in crossed_particles
            boundary_zone = system.boundary_zones[zone_id]
            particle_new = nparticles(fluid_ic) + zone_id

            u = copy(system.initial_condition.coordinates)
            v = fill(5.0, TrixiParticles.v_nvariables(system), nparticles(system))
            u_fluid = copy(fluid_system.initial_condition.coordinates)
            v_fluid = zeros(TrixiParticles.v_nvariables(fluid_system),
                            nparticles(fluid_system))
            u[1, particle] = x_new

            TrixiParticles.convert_particle!(system, fluid_system, boundary_zone,
                                             particle, particle_new, v, u, v_fluid,
                                             u_fluid, nothing)

            # The particle has been transferred to the fluid and recycled in its zone.
            @test fluid_system.buffer.active_particle[particle_new]
            @test TrixiParticles.is_in_boundary_zone(boundary_zone,
                                                     TrixiParticles.current_coords(u,
                                                                                   system,
                                                                                   particle))
            @test TrixiParticles.current_density(v, system, particle) ==
                  boundary_zone.rest_density
        end
    end
end
