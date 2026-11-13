@testset verbose=true "Refinement Callback" begin
    @testset "`ParticleRefinement` Startup Rounds" begin
        # `ceil(log(0.1 / 0.05) / log(1.2)) = 4` levels, three rounds per level
        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2,
                                                       min_spacing=0.05, max_spacing=0.1,
                                                       reference_density=1000.0)
        @test refinement.n_startup_rounds == 12

        # At least three rounds without a spacing band
        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2,
                                                       min_spacing=0.1, max_spacing=0.1,
                                                       reference_density=1000.0)
        @test refinement.n_startup_rounds == 3

        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2,
                                                       min_spacing=0.05, max_spacing=0.1,
                                                       reference_density=1000.0,
                                                       n_startup_rounds=2)
        @test refinement.n_startup_rounds == 2
    end

    @testset verbose=true "Show" begin
        callback = TrixiParticles.ParticleRefinementCallback()
        @test repr(callback) == "ParticleRefinementCallback(interval=1)"

        show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ ParticleRefinementCallback                                                                       │
            │ ══════════════════════════                                                                       │
            │ interval: ……………………………………………………… 1                                                                │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = TrixiParticles.ParticleRefinementCallback(dt=0.1)
        @test repr(callback) == "ParticleRefinementCallback(dt=0.1)"

        show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ ParticleRefinementCallback                                                                       │
            │ ══════════════════════════                                                                       │
            │ dt: ……………………………………………………………………… 0.1                                                              │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        error_str = "setting both `interval` and `dt` is not supported"
        @test_throws ArgumentError(error_str) TrixiParticles.ParticleRefinementCallback(interval=2,
                                                                                        dt=0.1)
    end

    # A fluid at `max_spacing` in a tank with walls at `min_spacing`, so the startup
    # refines the fluid near the walls
    particle_spacing = 0.1
    reference_density = 1000.0
    sound_speed = 10.0
    smoothing_kernel = WendlandC2Kernel{2}()
    smoothing_length = 1.2 * particle_spacing
    state_equation = StateEquationCole(; sound_speed, reference_density, exponent=7)

    # The wall spacing is `particle_spacing / 2 = 0.05`
    tank = RectangularTank(particle_spacing, (0.6, 0.6), (0.6, 0.8), reference_density;
                           n_layers=4, spacing_ratio=2)
    wall_spacing = tank.boundary.particle_spacing
    boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, 1.2 * wall_spacing;
                                                 state_equation)

    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2,
                                                   min_spacing=wall_spacing,
                                                   max_spacing=particle_spacing,
                                                   reference_density)

    # The default update strategy does not support inactive particles
    neighborhood_search = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())

    function create_integrator(callbacks...; particle_refinement=refinement,
                               buffer_size=10, kwargs...)
        fluid_system = EntropicallyDampedSPHSystem(tank.fluid; smoothing_kernel,
                                                   smoothing_length, sound_speed,
                                                   particle_refinement,
                                                   buffer_size, kwargs...)
        semi = Semidiscretization(fluid_system,
                                  WallBoundarySystem(tank.boundary, boundary_model);
                                  neighborhood_search,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0))

        return init(ode, SymplecticPositionVerlet(), dt=1e-4, save_everystep=false,
                    callback=CallbackSet(callbacks...))
    end

    n_initial = nparticles(tank.fluid)
    count_active(system) = length(TrixiParticles.each_active_particle(system))

    callbacks = [
        ("`interval`", TrixiParticles.ParticleRefinementCallback()),
        # The `PeriodicCallback` passes its own `initialize` to the startup
        ("`dt`", TrixiParticles.ParticleRefinementCallback(dt=0.01))
    ]

    @testset verbose=true "Startup With $name" for (name, callback) in callbacks
        integrator = create_integrator(callback)
        (; semi) = integrator.p
        fluid_system = semi.systems[1]
        v_ode, u_ode = integrator.u.x

        n_active = count_active(fluid_system)

        # The startup splits the particles near the walls
        @test n_active > n_initial

        # The integrator has the size of the resized systems
        sizes_v, sizes_u = TrixiParticles.sizes_vu(semi.systems)
        @test length(v_ode) == sum(sizes_v)
        @test length(u_ode) == sum(sizes_u)

        # The particles next to the walls have the wall spacing, the others are coarser
        u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        target_spacing = fluid_system.cache.target_spacing
        for particle in TrixiParticles.each_integrated_particle(fluid_system)
            @test wall_spacing - eps() <= target_spacing[particle] <=
                  particle_spacing + eps()
        end

        # Time integration works with the resized integrator
        for _ in 1:3
            step!(integrator)
        end
        @test integrator.t > 0
        @test all(isfinite, TrixiParticles.wrap_v(v_ode, fluid_system, semi))
        @test all(isfinite, TrixiParticles.wrap_u(u_ode, fluid_system, semi))
    end

    @testset verbose=true "Missing `SystemBuffer` With $name" for (name,
                                                                   callback) in callbacks
        # Refinement would only fail at the first split, so the startup throws before
        error_str = "particle refinement requires a `SystemBuffer`. Pass `buffer_size` " *
                    "to the `EntropicallyDampedSPHSystem` with `particle_refinement`."
        @test_throws ArgumentError(error_str) create_integrator(callback;
                                                                buffer_size=nothing)
    end

    @testset verbose=true "Order of Callbacks" begin
        # Without a startup, the refinement in the first time step splits the particles
        no_startup = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2,
                                                       min_spacing=wall_spacing,
                                                       max_spacing=particle_spacing,
                                                       reference_density,
                                                       n_startup_rounds=0)

        # The `UpdateCallback` comes first, so the shifting velocity of the particles
        # that the refinement adds or moves would be stale without the update
        # in the refinement callback. The shifting velocity is computed from the speed
        # of sound, since the fluid is at rest.
        integrator = create_integrator(UpdateCallback(),
                                       TrixiParticles.ParticleRefinementCallback();
                                       particle_refinement=no_startup,
                                       shifting_technique=ParticleShiftingTechnique(v_max_factor=0,
                                                                                    sound_speed_factor=0.1))
        (; semi) = integrator.p
        fluid_system = semi.systems[1]
        v_ode, u_ode = integrator.u.x

        @test count_active(fluid_system) == n_initial

        step!(integrator)

        @test count_active(fluid_system) > n_initial

        # Recompute the shifting velocity like the `UpdateCallback`
        delta_v = copy(fluid_system.cache.delta_v)
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, integrator.t)
        TrixiParticles.update_shifting_from_callback!(fluid_system,
                                                      TrixiParticles.shifting_technique(fluid_system),
                                                      v_ode, u_ode, semi)

        @test fluid_system.cache.delta_v ≈ delta_v
    end
end
