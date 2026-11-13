@testset verbose=true "Splitting" begin
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

    @testset verbose=true "`HexagonalSplitting`" begin
        pattern = TrixiParticles.HexagonalSplitting()
        directions = TrixiParticles.split_directions(pattern, Val(2))

        # The values of the code of Haftu et al.
        @test pattern.epsilon == 0.4
        @test pattern.alpha == 0.9
        @test TrixiParticles.n_split_particles(pattern, Val(2)) == 7

        # Unit vectors at angles `j * π / 3`
        @test all(direction -> norm(direction) ≈ 1, directions)
        @test directions[1] ≈ [1.0, 0.0]
        @test directions[2] ≈ [0.5, sqrt(3) / 2]
        @test isapprox(sum(directions), zeros(2), atol=1e-14)

        @test refinement.splitting_pattern isa TrixiParticles.HexagonalSplitting
        @test refinement.max_mass_factor == 1.05

        # TODO: Splitting patterns in 3D
        @test_throws "`HexagonalSplitting` is not implemented in 3D" TrixiParticles.split_directions(pattern,
                                                                                                     Val(3))
    end

    @testset verbose=true "`TriangularSplitting`" begin
        pattern = TrixiParticles.TriangularSplitting()
        directions = TrixiParticles.split_directions(pattern, Val(2))

        # The values of the code of Haftu et al.
        @test pattern.epsilon == 0.4
        @test pattern.alpha == 0.92
        @test TrixiParticles.n_split_particles(pattern, Val(2)) == 4

        # Unit vectors at angles `π / 2 * (j + 1 / 2)` for `j = 0, 1, 2`
        @test all(direction -> norm(direction) ≈ 1, directions)
        @test directions[1] ≈ [1, 1] / sqrt(2)
        @test directions[2] ≈ [-1, 1] / sqrt(2)
        @test directions[3] ≈ [-1, -1] / sqrt(2)

        @test_throws "`TriangularSplitting` is not implemented in 3D" TrixiParticles.split_directions(pattern,
                                                                                                      Val(3))
    end

    # A block of fluid particles in a tank. The target spacing of the particle in the
    # center is set to half the particle spacing, so that it is the only particle
    # that is heavier than `max_mass_factor` times its reference mass.
    tank = RectangularTank(particle_spacing, (0.9, 0.9), (1.1, 1.1), reference_density;
                           n_layers=3)
    boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length;
                                                 state_equation)
    coordinates = tank.fluid.coordinates
    center = argmin(vec(sum(abs2, coordinates .- [0.45, 0.45], dims=1)))

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

        mass = copy(fluid_system.mass)
        momentum = sum(particle -> mass[particle] *
                                   TrixiParticles.current_velocity(v, fluid_system,
                                                                   particle),
                       TrixiParticles.each_active_particle(fluid_system))
        position = TrixiParticles.current_coords(u, fluid_system, center)
        velocity = TrixiParticles.current_velocity(v, fluid_system, center)
        density = TrixiParticles.current_density(v, fluid_system, center)
        pressure = TrixiParticles.current_pressure(v, fluid_system, center)
        h = TrixiParticles.smoothing_length(fluid_system, center)

        fluid_system.cache.target_spacing[center] = particle_spacing / 2
        TrixiParticles.split_particles!(semi, v_ode, u_ode)

        v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
        u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        new_particles = setdiff(TrixiParticles.each_active_particle(fluid_system),
                                1:n_active)

        @testset "Only the Heavy Particle is Split" begin
            @test buffer.active_particle_count[] == n_active + 6
            @test length(new_particles) == 6

            # The new particles take the first free slots
            @test new_particles == (n_active + 1):(n_active + 6)
        end

        @testset "Mass and Momentum are Conserved" begin
            new_mass = fluid_system.mass
            active = TrixiParticles.each_active_particle(fluid_system)
            new_momentum = sum(particle -> new_mass[particle] *
                                           TrixiParticles.current_velocity(v,
                                                                           fluid_system,
                                                                           particle),
                               active)

            @test sum(new_mass[active]) ≈ sum(mass[1:n_active])
            @test new_momentum ≈ momentum
            @test all(≈(mass[center] / 7), new_mass[[center; new_particles]])
        end

        @testset "Positions" begin
            for (j, particle) in enumerate(new_particles)
                angle = (j - 1) * pi / 3
                expected = position + 0.4 * h * [cos(angle), sin(angle)]

                @test TrixiParticles.current_coords(u, fluid_system, particle) ≈ expected
            end

            # The particle itself stays in the center
            @test TrixiParticles.current_coords(u, fluid_system, center) == position
        end

        @testset "Properties are Copied" begin
            for particle in [center; new_particles]
                @test TrixiParticles.current_velocity(v, fluid_system, particle) ==
                      velocity
                @test TrixiParticles.current_density(v, fluid_system, particle) == density
                @test TrixiParticles.current_pressure(v, fluid_system, particle) ==
                      pressure
                @test TrixiParticles.smoothing_length(fluid_system, particle) ≈ 0.9 * h
                @test fluid_system.cache.target_spacing[particle] == particle_spacing / 2
            end
        end

        @testset "Other Particles are Unchanged" begin
            others = setdiff(1:n_active, center)

            @test fluid_system.mass[others] == mass[others]
            @test all(==(smoothing_length),
                      TrixiParticles.smoothing_length(fluid_system)[others])
        end

        @testset "New Particles Interact" begin
            dv_ode = rhs(v_ode, u_ode, p)
            dv = TrixiParticles.wrap_v(dv_ode, fluid_system, semi)

            @test all(particle -> any(!iszero, dv[:, particle]), new_particles)
            @test !any(fluid_system.cache.split_candidate)
        end
    end

    @testset verbose=true "Split With `TriangularSplitting`" begin
        triangular = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2,
                                                       min_spacing=0.05, max_spacing=0.1,
                                                       reference_density,
                                                       splitting_pattern=TrixiParticles.TriangularSplitting())
        v_ode, u_ode, p = semidiscretize_systems(wcsph(particle_refinement=triangular))
        (; semi) = p
        fluid_system = semi.systems[1]
        n_active = fluid_system.buffer.active_particle_count[]
        h = TrixiParticles.smoothing_length(fluid_system, center)
        position = tank.fluid.coordinates[:, center]

        fluid_system.cache.target_spacing[center] = particle_spacing / 2
        TrixiParticles.split_particles!(semi, v_ode, u_ode)

        u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        new_particles = (n_active + 1):(n_active + 3)

        @test fluid_system.buffer.active_particle_count[] == n_active + 3
        @test all(≈(tank.fluid.mass[center] / 4),
                  fluid_system.mass[[center; new_particles]])
        @test all(≈(0.92 * h),
                  TrixiParticles.smoothing_length(fluid_system)[[center; new_particles]])

        for (j, particle) in enumerate(new_particles)
            angle = pi / 2 * (j - 1 / 2)
            expected = position + 0.4 * h * [cos(angle), sin(angle)]

            @test TrixiParticles.current_coords(u, fluid_system, particle) ≈ expected
        end
    end

    @testset verbose=true "No Splitting Below the Mass Limit" begin
        v_ode, u_ode, p = semidiscretize_systems(wcsph())
        (; semi) = p
        fluid_system = semi.systems[1]
        n_active = fluid_system.buffer.active_particle_count[]
        length_v = length(v_ode)

        # Slightly lighter than `max_mass_factor * m_ref`
        fluid_system.cache.target_spacing .= particle_spacing / sqrt(1.04)
        TrixiParticles.split_particles!(semi, v_ode, u_ode)

        @test fluid_system.buffer.active_particle_count[] == n_active
        @test length(v_ode) == length_v
    end

    @testset verbose=true "Free Slots Between Active Particles" begin
        v_ode, u_ode, p = semidiscretize_systems(wcsph())
        (; semi) = p
        fluid_system = semi.systems[1]
        (; buffer) = fluid_system
        n_active = buffer.active_particle_count[]

        # Deactivate a particle far away from the center
        v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
        u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        TrixiParticles.deactivate_particle!(fluid_system, 1, v, u)
        TrixiParticles.update_system_buffer!(buffer)

        fluid_system.cache.target_spacing[center] = particle_spacing / 2
        TrixiParticles.split_particles!(semi, v_ode, u_ode)

        # The first new particle is activated in the free slot
        @test buffer.active_particle[1]
        @test buffer.active_particle_count[] == n_active - 1 + 6
        @test count(buffer.active_particle[(n_active + 1):end]) == 5
    end

    @testset verbose=true "Resize" begin
        # Four free slots are not enough for six new particles
        v_ode, u_ode, p = semidiscretize_systems(wcsph(buffer_size=4))
        (; semi) = p
        fluid_system = semi.systems[1]
        capacity = nparticles(fluid_system)
        n_active = fluid_system.buffer.active_particle_count[]

        fluid_system.cache.target_spacing[center] = particle_spacing / 2
        TrixiParticles.split_particles!(semi, v_ode, u_ode)

        @test nparticles(fluid_system) ==
              max(round(Int, 1.25 * capacity), n_active + 6 + 4)
        @test fluid_system.buffer.active_particle_count[] == n_active + 6

        dv_ode = rhs(v_ode, u_ode, p)
        @test length(dv_ode) == length(v_ode)
    end

    @testset verbose=true "No Splitting Near Boundaries" begin
        v_ode, u_ode, p = semidiscretize_systems(wcsph())
        (; semi) = p
        fluid_system = semi.systems[1]
        n_active = fluid_system.buffer.active_particle_count[]

        # Move a particle next to the left wall to `x = 0.01`. The wall particles are at
        # `x = -0.05`, so the wall surface is at `x = 0`. The new particle in the
        # direction of the wall at `0.4 * h = 0.048` from the particle would be behind
        # the wall surface.
        near_wall = argmin(vec(sum(abs2, coordinates .- [0.05, 0.45], dims=1)))
        u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        TrixiParticles.set_particle_position!(u, fluid_system, near_wall, [0.01, 0.45])
        TrixiParticles.update_nhs!(semi, u_ode)

        fluid_system.cache.target_spacing[[center, near_wall]] .= particle_spacing / 2
        TrixiParticles.split_particles!(semi, v_ode, u_ode)

        @test fluid_system.buffer.active_particle_count[] == n_active + 6
        @test fluid_system.mass[near_wall] == tank.fluid.mass[near_wall]
        @test fluid_system.mass[center] ≈ tank.fluid.mass[center] / 7

        # A particle at its regular position next to the wall is split, since the new
        # particles are in front of the wall surface
        v_ode, u_ode, p = semidiscretize_systems(wcsph())
        (; semi) = p
        fluid_system = semi.systems[1]
        fluid_system.cache.target_spacing[near_wall] = particle_spacing / 2
        TrixiParticles.split_particles!(semi, v_ode, u_ode)

        @test fluid_system.buffer.active_particle_count[] == n_active + 6
    end

    @testset verbose=true "No Splitting Into Open Boundary Zones" begin
        fluid_system = edac(density_calculator=ContinuityDensity())

        # The inflow zone is left of the face at `x = 0`
        boundary_zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 0.9]),
                                     face_normal=[1.0, 0.0], boundary_type=InFlow(),
                                     open_boundary_layers=2, density=reference_density,
                                     particle_spacing)
        open_boundary = OpenBoundarySystem(boundary_zone; fluid_system, buffer_size=4,
                                           boundary_model=BoundaryModelDynamicalPressureZhang())

        semi = Semidiscretization(fluid_system, open_boundary; neighborhood_search,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x
        semi = ode.p.semi
        fluid_system = semi.systems[1]
        n_active = fluid_system.buffer.active_particle_count[]

        # Move a particle to `x = 0.03`. The new particle in the direction of the face
        # at `0.4 * h = 0.048` from the particle would be in the zone.
        near_face = argmin(vec(sum(abs2, coordinates .- [0.05, 0.45], dims=1)))
        u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        TrixiParticles.set_particle_position!(u, fluid_system, near_face, [0.03, 0.45])
        TrixiParticles.update_nhs!(semi, u_ode)

        fluid_system.cache.target_spacing[[center, near_face]] .= particle_spacing / 2
        TrixiParticles.split_particles!(semi, v_ode, u_ode)

        @test fluid_system.buffer.active_particle_count[] == n_active + 6
        @test fluid_system.mass[near_face] == tank.fluid.mass[near_face]
        @test fluid_system.mass[center] ≈ tank.fluid.mass[center] / 7
    end

    @testset verbose=true "Splitting Requires a Buffer" begin
        v_ode, u_ode, p = semidiscretize_systems(wcsph(buffer_size=nothing))
        (; semi) = p
        fluid_system = semi.systems[1]
        fluid_system.cache.target_spacing[center] = particle_spacing / 2

        @test_throws "splitting particles requires a `SystemBuffer`" TrixiParticles.split_particles!(semi,
                                                                                                     v_ode,
                                                                                                     u_ode)
    end
end
