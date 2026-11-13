@testset verbose=true "Merging" begin
    smoothing_kernel = WendlandC2Kernel{2}()
    sound_speed = 10.0
    reference_density = 1000.0
    state_equation = StateEquationCole(; sound_speed, reference_density, exponent=7)
    particle_spacing = 0.1
    smoothing_length = 1.2 * particle_spacing

    # The default update strategy does not support inactive particles
    neighborhood_search = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())

    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2, min_spacing=0.05,
                                                   max_spacing=0.1, reference_density)

    # The reference mass of all particles is `reference_density * particle_spacing^2 = 10`,
    # so particles below 5 can merge after splitting and below 10.5 before splitting.
    tank = RectangularTank(particle_spacing, (0.9, 0.9), (1.1, 1.1), reference_density;
                           n_layers=3)
    boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length;
                                                 state_equation)
    coordinates = tank.fluid.coordinates
    nearest_particle(x) = argmin(vec(sum(abs2, coordinates .- x, dims=1)))

    # Two neighbors in the middle of the tank
    particle_a = nearest_particle([0.45, 0.45])
    particle_b = nearest_particle([0.55, 0.45])

    function wcsph(; density_calculator=ContinuityDensity(), buffer_size=10,
                   particle_refinement=refinement, kwargs...)
        return WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel, smoothing_length,
                                           state_equation, density_calculator,
                                           particle_refinement, buffer_size,
                                           kwargs...)
    end

    function edac(; density_calculator=SummationDensity(), buffer_size=10,
                  particle_refinement=refinement, kwargs...)
        return EntropicallyDampedSPHSystem(tank.fluid; smoothing_kernel, smoothing_length,
                                           sound_speed, density_calculator,
                                           particle_refinement, buffer_size,
                                           kwargs...)
    end

    function semidiscretize_systems(fluid_system)
        semi = Semidiscretization(fluid_system,
                                  WallBoundarySystem(tank.boundary, boundary_model);
                                  neighborhood_search,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x

        return v_ode, u_ode, ode.p
    end

    function rhs(v_ode, u_ode, p)
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, p, 0.0)

        return dv_ode
    end

    W(r, h) = TrixiParticles.kernel(smoothing_kernel, r, h)

    setups = [
        ("WCSPH with `ContinuityDensity`", () -> wcsph()),
        ("WCSPH with `SummationDensity`",
         () -> wcsph(density_calculator=SummationDensity())),
        ("EDAC with `ContinuityDensity`",
         () -> edac(density_calculator=ContinuityDensity())),
        ("EDAC with `SummationDensity`", () -> edac())
    ]

    @testset verbose=true "$name" for (name, create_fluid) in setups
        v_ode, u_ode, p = semidiscretize_systems(create_fluid())
        (; semi) = p
        fluid_system = semi.systems[1]
        (; buffer) = fluid_system

        v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
        u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        n_active = buffer.active_particle_count[]

        # Use a non-trivial state
        for particle in TrixiParticles.each_active_particle(fluid_system)
            TrixiParticles.set_particle_velocity!(v, fluid_system, particle,
                                                  [sin(particle), cos(particle)])
            TrixiParticles.set_particle_density!(v, fluid_system, particle,
                                                 reference_density + particle)
            TrixiParticles.set_particle_pressure!(v, fluid_system, particle,
                                                  10.0 * particle)
        end

        # Two light particles with different masses
        m_a, m_b = 3.0, 4.0
        TrixiParticles.set_particle_mass!(fluid_system, particle_a, m_a)
        TrixiParticles.set_particle_mass!(fluid_system, particle_b, m_b)

        mass = copy(fluid_system.mass)
        active = collect(TrixiParticles.each_active_particle(fluid_system))
        momentum = sum(particle -> mass[particle] *
                                   TrixiParticles.current_velocity(v, fluid_system,
                                                                   particle), active)
        pos_a = TrixiParticles.current_coords(u, fluid_system, particle_a)
        pos_b = TrixiParticles.current_coords(u, fluid_system, particle_b)
        vel_a = TrixiParticles.current_velocity(v, fluid_system, particle_a)
        vel_b = TrixiParticles.current_velocity(v, fluid_system, particle_b)
        p_a = TrixiParticles.current_pressure(v, fluid_system, particle_a)
        p_b = TrixiParticles.current_pressure(v, fluid_system, particle_b)
        density_a = TrixiParticles.current_density(v, fluid_system, particle_a)

        # Only the two light particles are candidates
        TrixiParticles.merge_particles!(semi, v_ode, u_ode)

        # The particle with the smaller index takes the merged properties
        merged, deleted = minmax(particle_a, particle_b)
        m = m_a + m_b
        position = (m_a * pos_a + m_b * pos_b) / m

        @testset "Only the Light Pair is Merged" begin
            @test buffer.active_particle_count[] == n_active - 1
            @test !buffer.active_particle[deleted]
            @test all(==(1e16), u[:, deleted])
            @test fluid_system.mass[merged] == m
        end

        @testset "Mass and Momentum are Conserved" begin
            new_active = TrixiParticles.each_active_particle(fluid_system)
            new_momentum = sum(particle -> fluid_system.mass[particle] *
                                           TrixiParticles.current_velocity(v,
                                                                           fluid_system,
                                                                           particle),
                               new_active)

            @test sum(fluid_system.mass[new_active]) ≈ sum(mass[active])
            @test new_momentum ≈ momentum
        end

        @testset "Merged Properties" begin
            @test TrixiParticles.current_coords(u, fluid_system, merged) ≈ position
            @test TrixiParticles.current_velocity(v, fluid_system, merged) ≈
                  (m_a * vel_a + m_b * vel_b) / m
            @test TrixiParticles.current_pressure(v, fluid_system, merged) ≈
                  (m_a * p_a + m_b * p_b) / m
            @test TrixiParticles.current_density(v, fluid_system, merged) == density_a

            # Eq. 34
            h = smoothing_length
            h_expected = sqrt(m * W(0.0, 1.0) /
                              (m_a * W(norm(position - pos_a), h) +
                               m_b * W(norm(position - pos_b), h)))
            @test TrixiParticles.smoothing_length(fluid_system, merged) ≈ h_expected

            # The target spacing is not changed
            @test fluid_system.cache.target_spacing[merged] == particle_spacing
        end

        @testset "Other Particles are Unchanged" begin
            others = setdiff(active, [particle_a, particle_b])

            @test fluid_system.mass[others] == mass[others]
            @test all(==(smoothing_length),
                      TrixiParticles.smoothing_length(fluid_system)[others])
        end

        @testset "Deleted Particle Does Not Interact" begin
            u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
            coords = TrixiParticles.current_coordinates(u, fluid_system)
            neighbors = Int[]
            TrixiParticles.foreach_point_neighbor(fluid_system, fluid_system, coords,
                                                  coords, semi) do particle, neighbor,
                                                                   _, _
                push!(neighbors, particle, neighbor)
            end

            @test !(deleted in neighbors)
            @test merged in neighbors

            dv_ode = rhs(v_ode, u_ode, p)
            dv = TrixiParticles.wrap_v(dv_ode, fluid_system, semi)
            @test any(!iszero, dv[:, merged])
        end
    end

    @testset verbose=true "Mass Limits Before and After Splitting" begin
        # Both particles are below `m_max` and their merged mass is below `m_max`,
        # but `particle_a` is not below `m_min`.
        function setup()
            v_ode, u_ode, p = semidiscretize_systems(wcsph())
            fluid_system = p.semi.systems[1]
            TrixiParticles.set_particle_mass!(fluid_system, particle_a, 6.0)
            TrixiParticles.set_particle_mass!(fluid_system, particle_b, 4.0)

            return v_ode, u_ode, p.semi, fluid_system
        end

        v_ode, u_ode, semi, fluid_system = setup()
        n_active = fluid_system.buffer.active_particle_count[]
        TrixiParticles.merge_particles!(semi, v_ode, u_ode)
        @test fluid_system.buffer.active_particle_count[] == n_active

        v_ode, u_ode, semi, fluid_system = setup()
        TrixiParticles.merge_particles!(semi, v_ode, u_ode, before_splitting=true)
        @test fluid_system.buffer.active_particle_count[] == n_active - 1
        @test fluid_system.mass[min(particle_a, particle_b)] == 10.0
    end

    @testset verbose=true "No Merging Above the Mass Limit" begin
        v_ode, u_ode, p = semidiscretize_systems(wcsph())
        (; semi) = p
        fluid_system = semi.systems[1]
        n_active = fluid_system.buffer.active_particle_count[]

        # The merged mass would be above `m_max`
        TrixiParticles.set_particle_mass!(fluid_system, particle_a, 6.0)
        TrixiParticles.set_particle_mass!(fluid_system, particle_b, 5.0)
        TrixiParticles.merge_particles!(semi, v_ode, u_ode, before_splitting=true)

        @test fluid_system.buffer.active_particle_count[] == n_active

        # Particles with the reference mass are never merged
        v_ode, u_ode, p = semidiscretize_systems(wcsph())
        (; semi) = p
        fluid_system = semi.systems[1]
        TrixiParticles.merge_particles!(semi, v_ode, u_ode, before_splitting=true)
        TrixiParticles.merge_particles!(semi, v_ode, u_ode)

        @test fluid_system.buffer.active_particle_count[] == n_active
    end

    @testset verbose=true "Pairs Must Choose Each Other" begin
        # `particle_c` is closer to `particle_b` than `particle_a`, so `particle_a`
        # does not find a partner in the first round. In the second round,
        # it merges with the merged particle.
        particle_c = nearest_particle([0.65, 0.45])

        function setup()
            v_ode, u_ode, p = semidiscretize_systems(wcsph())
            (; semi) = p
            fluid_system = semi.systems[1]
            u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
            TrixiParticles.set_particle_position!(u, fluid_system, particle_c, [0.58, 0.45])
            TrixiParticles.update_nhs!(semi, u_ode)

            for particle in (particle_a, particle_b, particle_c)
                TrixiParticles.set_particle_mass!(fluid_system, particle, 2.0)
            end

            return v_ode, u_ode, semi, fluid_system
        end

        # Only one round before splitting
        v_ode, u_ode, semi, fluid_system = setup()
        n_active = fluid_system.buffer.active_particle_count[]
        TrixiParticles.merge_particles!(semi, v_ode, u_ode, before_splitting=true)

        @test fluid_system.buffer.active_particle_count[] == n_active - 1
        @test fluid_system.mass[particle_a] == 2.0
        @test fluid_system.mass[min(particle_b, particle_c)] == 4.0

        # Up to three rounds after splitting
        v_ode, u_ode, semi, fluid_system = setup()
        TrixiParticles.merge_particles!(semi, v_ode, u_ode)

        @test fluid_system.buffer.active_particle_count[] == n_active - 2
        @test fluid_system.mass[min(particle_a, particle_b, particle_c)] == 6.0
    end

    @testset verbose=true "Periodic Boundaries" begin
        fluid = RectangularShape(particle_spacing, (9, 9), (0.0, 0.0);
                                 density=reference_density)
        periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[0.9, 0.9])
        periodic_nhs = GridNeighborhoodSearch{2}(; periodic_box,
                                                 update_strategy=SerialUpdate())
        fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length, state_equation,
                                                   density_calculator=ContinuityDensity(),
                                                   particle_refinement=refinement,
                                                   buffer_size=10)
        semi = Semidiscretization(fluid_system; neighborhood_search=periodic_nhs,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x
        semi = ode.p.semi
        fluid_system = semi.systems[1]

        # Two particles on opposite sides of the periodic box
        nearest_fluid(x) = argmin(vec(sum(abs2, fluid.coordinates .- x, dims=1)))
        left = nearest_fluid([0.05, 0.45])
        right = nearest_fluid([0.85, 0.45])
        TrixiParticles.set_particle_mass!(fluid_system, left, 3.0)
        TrixiParticles.set_particle_mass!(fluid_system, right, 1.0)

        TrixiParticles.merge_particles!(semi, v_ode, u_ode)

        # The center of mass across the periodic boundary
        u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        merged = min(left, right)
        h_expected = sqrt(4 * W(0.0, 1.0) /
                          (3 * W(0.025, smoothing_length) +
                           1 * W(0.075, smoothing_length)))

        @test fluid_system.buffer.active_particle_count[] == nparticles(fluid) - 1
        @test TrixiParticles.current_coords(u, fluid_system, merged) ≈ [0.025, 0.45]
        @test TrixiParticles.smoothing_length(fluid_system, merged) ≈ h_expected
    end

    @testset verbose=true "Splitting Reuses the Free Slot" begin
        v_ode, u_ode, p = semidiscretize_systems(wcsph(buffer_size=6))
        (; semi) = p
        fluid_system = semi.systems[1]
        (; buffer) = fluid_system
        n_active = buffer.active_particle_count[]

        TrixiParticles.set_particle_mass!(fluid_system, particle_a, 3.0)
        TrixiParticles.set_particle_mass!(fluid_system, particle_b, 4.0)
        TrixiParticles.merge_particles!(semi, v_ode, u_ode)
        deleted = max(particle_a, particle_b)

        # Split a particle far away from the merged pair. The slot of the deleted particle
        # is the first free slot, so it is reused for the first new particle.
        heavy = nearest_particle([0.15, 0.15])
        fluid_system.cache.target_spacing[heavy] = particle_spacing / 2
        TrixiParticles.split_particles!(semi, v_ode, u_ode)

        @test buffer.active_particle[deleted]
        @test buffer.active_particle_count[] == n_active - 1 + 6
    end

    @testset verbose=true "Merging Requires a Buffer" begin
        v_ode, u_ode, p = semidiscretize_systems(wcsph(buffer_size=nothing))
        (; semi) = p
        fluid_system = semi.systems[1]

        # Without candidates, nothing happens
        TrixiParticles.merge_particles!(semi, v_ode, u_ode)

        TrixiParticles.set_particle_mass!(fluid_system, particle_a, 3.0)
        TrixiParticles.set_particle_mass!(fluid_system, particle_b, 4.0)

        @test_throws "merging particles requires a `SystemBuffer`" TrixiParticles.merge_particles!(semi,
                                                                                                   v_ode,
                                                                                                   u_ode)
    end
end
