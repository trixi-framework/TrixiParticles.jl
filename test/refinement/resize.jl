@testset verbose=true "Resize" begin
    smoothing_kernel = WendlandC2Kernel{2}()
    sound_speed = 10.0
    reference_density = 1000.0
    state_equation = StateEquationCole(; sound_speed, reference_density, exponent=7)
    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2, min_spacing=0.05,
                                                   max_spacing=0.1, reference_density)
    particle_spacing = 0.1
    smoothing_length = 1.2 * particle_spacing

    tank = RectangularTank(particle_spacing, (0.5, 0.5), (0.7, 0.7), reference_density;
                           n_layers=3)
    boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length;
                                                 state_equation)

    # A second fluid without refinement next to the tank, which is not resized
    other_fluid = RectangularShape(particle_spacing, (3, 3), (0.8, 0.1);
                                   density=reference_density)

    function wcsph(fluid; density_calculator=ContinuityDensity(), kwargs...)
        return WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           state_equation, density_calculator, kwargs...)
    end

    function edac(fluid; kwargs...)
        return EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           sound_speed, kwargs...)
    end

    # The default update strategy does not support inactive particles
    grid_nhs = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())

    function semidiscretize_systems(systems...; neighborhood_search=grid_nhs,
                                    neighborhood_search_handler=SharedNHSHandler)
        semi = Semidiscretization(systems...; neighborhood_search,
                                  neighborhood_search_handler,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x

        # The shifting velocity is set manually instead of in the `UpdateCallback`
        ode.p.semi.update_callback_used[] = true

        return v_ode, u_ode, ode.p
    end

    function rhs(v_ode, u_ode, p)
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, p, 0.0)

        return dv_ode
    end

    setups = [
        ("WCSPH with `ContinuityDensity`",
         () -> wcsph(tank.fluid; density_calculator=ContinuityDensity(),
                     particle_refinement=refinement, buffer_size=4)),
        ("WCSPH with `SummationDensity`",
         () -> wcsph(tank.fluid; density_calculator=SummationDensity(),
                     particle_refinement=refinement, buffer_size=4)),
        ("EDAC with `ContinuityDensity`",
         () -> edac(tank.fluid; density_calculator=ContinuityDensity(),
                    particle_refinement=refinement, buffer_size=4)),
        ("EDAC with `SummationDensity` and Average Pressure Reduction",
         () -> edac(tank.fluid; average_pressure_reduction=true,
                    particle_refinement=refinement, buffer_size=4)),
        # The shifting velocity is only updated in the `UpdateCallback`,
        # so the right-hand side uses the values in the cache.
        ("EDAC with Shifting",
         () -> edac(tank.fluid; shifting_technique=ParticleShiftingTechnique(),
                    particle_refinement=refinement, buffer_size=4))
    ]

    full_grid_nhs = GridNeighborhoodSearch{2}(cell_list=FullGridCellList(min_corner=(-0.5,
                                                                                     -0.5),
                                                                         max_corner=(1.5,
                                                                                     1.5)),
                                              update_strategy=ParallelUpdate())

    # This neighborhood search loops over the number of points it was created with,
    # so it only finds the new particles after it has been recreated.
    trivial_nhs = TrivialNeighborhoodSearch{2}()

    neighborhood_searches = [
        ("`GridNeighborhoodSearch`, `SharedNHSHandler`", grid_nhs, SharedNHSHandler),
        ("`GridNeighborhoodSearch`, `PairsNHSHandler`", grid_nhs, PairsNHSHandler),
        ("`FullGridCellList`, `SharedNHSHandler`", full_grid_nhs, SharedNHSHandler),
        ("`TrivialNeighborhoodSearch`, `SharedNHSHandler`", trivial_nhs,
         SharedNHSHandler),
        ("`TrivialNeighborhoodSearch`, `PairsNHSHandler`", trivial_nhs, PairsNHSHandler)
    ]

    @testset verbose=true "$name, $nhs_name" for (name, create_fluid) in setups,
                                                 (nhs_name, neighborhood_search,
                                                  neighborhood_search_handler) in
                                                 neighborhood_searches

        v_ode, u_ode,
        p = semidiscretize_systems(create_fluid(),
                                   WallBoundarySystem(tank.boundary, boundary_model),
                                   wcsph(other_fluid;
                                         density_calculator=ContinuityDensity());
                                   neighborhood_search, neighborhood_search_handler)
        (; semi) = p
        fluid_system, boundary_system, other_system = semi.systems

        # Use a non-trivial state for the active particles
        n_old = nparticles(fluid_system)
        active = collect(TrixiParticles.each_active_particle(fluid_system))
        v_fluid = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
        u_fluid = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
        for particle in active
            v_fluid[1, particle] = sin(particle)
            v_fluid[2, particle] = cos(particle)
            u_fluid[1, particle] += 0.01 * sin(2 * particle)
        end

        has_shifting = !isnothing(TrixiParticles.shifting_technique(fluid_system))
        if has_shifting
            fluid_system.cache.delta_v .= 0.1 .* cos.(eachindex(fluid_system.cache.delta_v))
        end
        delta_v_old = has_shifting ? copy(fluid_system.cache.delta_v) : nothing

        v_fluid_old = copy(v_fluid)
        u_fluid_old = copy(u_fluid)
        v_other_old = copy(TrixiParticles.wrap_v(v_ode, other_system, semi))
        u_other_old = copy(TrixiParticles.wrap_u(u_ode, other_system, semi))
        dv_ode_old = rhs(v_ode, u_ode, p)
        dv_fluid_old = copy(TrixiParticles.wrap_v(dv_ode_old, fluid_system, semi))
        dv_other_old = copy(TrixiParticles.wrap_v(dv_ode_old, other_system, semi))

        n_new = n_old + 7
        TrixiParticles.resize_systems!(v_ode, u_ode, semi,
                                       [n_new, nparticles(boundary_system),
                                           nparticles(other_system)])

        v_fluid = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
        u_fluid = TrixiParticles.wrap_u(u_ode, fluid_system, semi)

        @testset "Number of Particles" begin
            (; buffer) = fluid_system

            @test nparticles(fluid_system) == n_new
            @test length(buffer.active_particle) == n_new
            @test length(buffer.eachparticle) == n_new
            @test TrixiParticles.each_active_particle(fluid_system) == active
            @test TrixiParticles.smoothing_length(fluid_system) isa Vector
            @test length(TrixiParticles.smoothing_length(fluid_system)) == n_new
        end

        @testset "Ranges" begin
            n_v = TrixiParticles.v_nvariables(fluid_system)
            n_u = TrixiParticles.u_nvariables(fluid_system)

            @test semi.ranges_v[1] == 1:(n_v * n_new)
            @test semi.ranges_u[1] == 1:(n_u * n_new)
            @test first(semi.ranges_v[3]) == last(semi.ranges_v[1]) + 1
            @test first(semi.ranges_u[3]) == last(semi.ranges_u[1]) + 1
            @test length(v_ode) == last(semi.ranges_v[3])
            @test length(u_ode) == last(semi.ranges_u[3])
        end

        @testset "Data is Kept" begin
            @test v_fluid[:, 1:n_old] == v_fluid_old
            @test u_fluid[:, 1:n_old] == u_fluid_old
            @test TrixiParticles.wrap_v(v_ode, other_system, semi) == v_other_old
            @test TrixiParticles.wrap_u(u_ode, other_system, semi) == u_other_old

            if has_shifting
                (; delta_v) = fluid_system.cache
                @test length(delta_v) == 2 * n_new
                @test delta_v[1:(2 * n_old)] == delta_v_old
                @test iszero(delta_v[(2 * n_old + 1):end])
            end
        end

        @testset "New Particles are Inactive" begin
            new_particles = (n_old + 1):n_new
            spacing = tank.fluid.particle_spacing

            @test !any(fluid_system.buffer.active_particle[new_particles])
            @test all(==(1e16), u_fluid[:, new_particles])
            @test iszero(TrixiParticles.current_velocity(v_fluid, fluid_system)[:,
                                                                                new_particles])
            @test all(==(reference_density),
                      TrixiParticles.current_density(v_fluid, fluid_system)[new_particles])
            @test iszero(TrixiParticles.current_pressure(v_fluid,
                                                         fluid_system)[new_particles])
            @test all(≈(reference_density * spacing^2), fluid_system.mass[new_particles])
            @test all(==(smoothing_length),
                      TrixiParticles.smoothing_length(fluid_system)[new_particles])
        end

        @testset "Right-Hand Side is Unchanged" begin
            dv_ode = rhs(v_ode, u_ode, p)
            dv_fluid = TrixiParticles.wrap_v(dv_ode, fluid_system, semi)

            @test any(!iszero, dv_fluid_old)
            @test isapprox(dv_fluid[:, 1:n_old], dv_fluid_old, rtol=1e-12)
            @test iszero(dv_fluid[:, (n_old + 1):n_new])
            @test isapprox(TrixiParticles.wrap_v(dv_ode, other_system, semi),
                           dv_other_old, rtol=1e-12)
        end

        @testset "Resize Again" begin
            TrixiParticles.resize_systems!(v_ode, u_ode, semi,
                                           [n_new + 3, nparticles(boundary_system),
                                               nparticles(other_system)])

            v_fluid = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
            dv_ode = rhs(v_ode, u_ode, p)
            dv_fluid = TrixiParticles.wrap_v(dv_ode, fluid_system, semi)

            @test nparticles(fluid_system) == n_new + 3
            @test v_fluid[:, 1:n_old] == v_fluid_old
            @test isapprox(dv_fluid[:, 1:n_old], dv_fluid_old, rtol=1e-12)
        end

        @testset "Activated New Particle Interacts" begin
            (; buffer) = fluid_system
            v_fluid = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
            u_fluid = TrixiParticles.wrap_u(u_ode, fluid_system, semi)

            # Activate the last new particle next to the first particle
            particle = nparticles(fluid_system)
            position = u_fluid[:, 1] + [0.03, 0.04]
            TrixiParticles.set_particle_position!(u_fluid, fluid_system, particle, position)
            TrixiParticles.set_particle_velocity!(v_fluid, fluid_system, particle,
                                                  [1.0, 0.0])
            buffer.active_particle[particle] = true
            TrixiParticles.update_system_buffer!(buffer)

            dv_ode = rhs(v_ode, u_ode, p)
            dv_fluid = TrixiParticles.wrap_v(dv_ode, fluid_system, semi)

            @test any(!iszero, dv_fluid[:, particle])

            # The first particle finds the new particle as a neighbor
            @test !isapprox(dv_fluid[:, 1], dv_fluid_old[:, 1])
        end
    end

    @testset verbose=true "Unchanged Number of Particles" begin
        v_ode, u_ode,
        p = semidiscretize_systems(wcsph(tank.fluid; particle_refinement=refinement,
                                         buffer_size=4))
        length_v = length(v_ode)

        TrixiParticles.resize_systems!(v_ode, u_ode, p.semi,
                                       [nparticles(p.semi.systems[1])])

        @test length(v_ode) == length_v
    end

    @testset verbose=true "Rejected" begin
        function test_rejected(error_str, create_fluid, n_extra=1)
            v_ode, u_ode,
            p = semidiscretize_systems(create_fluid(),
                                       WallBoundarySystem(tank.boundary, boundary_model))
            (; semi) = p
            fluid_system, boundary_system = semi.systems
            length_v, length_u = length(v_ode), length(u_ode)
            n_fluid = nparticles(fluid_system)

            @test_throws error_str TrixiParticles.resize_systems!(v_ode, u_ode, semi,
                                                                  [n_fluid + n_extra,
                                                                      nparticles(boundary_system)
                                                                  ])

            # Nothing was changed
            @test length(v_ode) == length_v
            @test length(u_ode) == length_u
            @test nparticles(fluid_system) == n_fluid
        end

        @testset "Without Refinement" begin
            test_rejected("only fluid systems with particle refinement can be resized",
                          () -> wcsph(tank.fluid; buffer_size=4))
        end

        @testset "Without Buffer" begin
            test_rejected("resizing a system requires a `SystemBuffer`",
                          () -> wcsph(tank.fluid; particle_refinement=refinement))
        end

        @testset "Shrinking" begin
            test_rejected("systems can only grow",
                          () -> wcsph(tank.fluid; particle_refinement=refinement,
                                      buffer_size=4), -1)
        end

        @testset "Surface Normals" begin
            test_rejected("resizing a system with surface normals or surface tension",
                          () -> wcsph(tank.fluid; particle_refinement=refinement,
                                      buffer_size=4,
                                      surface_normal_method=ColorfieldSurfaceNormal(),
                                      reference_particle_spacing=particle_spacing))
        end

        @testset "`DensityDiffusionAntuono`" begin
            density_diffusion = DensityDiffusionAntuono(; delta=0.1)
            test_rejected("resizing a system with `DensityDiffusionAntuono`",
                          () -> wcsph(tank.fluid; particle_refinement=refinement,
                                      buffer_size=4, density_diffusion))
        end

        @testset "Boundary System" begin
            v_ode, u_ode,
            p = semidiscretize_systems(wcsph(tank.fluid; particle_refinement=refinement,
                                             buffer_size=4),
                                       WallBoundarySystem(tank.boundary, boundary_model))
            fluid_system, boundary_system = p.semi.systems

            @test_throws "only fluid systems with particle refinement can be resized" TrixiParticles.resize_systems!(v_ode,
                                                                                                                     u_ode,
                                                                                                                     p.semi,
                                                                                                                     [
                                                                                                                         nparticles(fluid_system),
                                                                                                                         nparticles(boundary_system) +
                                                                                                                         1
                                                                                                                     ])
        end

        @testset "Wrong Number of Entries" begin
            v_ode, u_ode,
            p = semidiscretize_systems(wcsph(tank.fluid; particle_refinement=refinement,
                                             buffer_size=4))

            @test_throws "`n_particles` must have one entry per system" TrixiParticles.resize_systems!(v_ode,
                                                                                                       u_ode,
                                                                                                       p.semi,
                                                                                                       [
                                                                                                           1,
                                                                                                           2
                                                                                                       ])
        end
    end

    @testset verbose=true "Shifting and Open Boundary" begin
        # With shifting, the zone must be at least two compact supports wide
        fluid_system = edac(tank.fluid; particle_refinement=refinement, buffer_size=4,
                            shifting_technique=ParticleShiftingTechnique())
        boundary_zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 0.5]),
                                     face_normal=[1.0, 0.0], boundary_type=InFlow(),
                                     open_boundary_layers=5, density=reference_density,
                                     particle_spacing)
        open_boundary = OpenBoundarySystem(boundary_zone; fluid_system, buffer_size=4,
                                           boundary_model=BoundaryModelDynamicalPressureZhang())

        v_ode, u_ode, p = semidiscretize_systems(fluid_system, open_boundary)
        fluid_system, open_boundary = p.semi.systems

        n_old = nparticles(fluid_system)
        fluid_system.cache.delta_v .= 1:length(fluid_system.cache.delta_v)
        delta_v_old = copy(fluid_system.cache.delta_v)

        n_new = n_old + 7
        TrixiParticles.resize_systems!(v_ode, u_ode, p.semi,
                                       [n_new, nparticles(open_boundary)])

        @test length(fluid_system.cache.delta_v) == 2 * n_new
        @test fluid_system.cache.delta_v[1:(2 * n_old)] == delta_v_old
        @test iszero(fluid_system.cache.delta_v[(2 * n_old + 1):end])
        @test length(open_boundary.fluid_candidates) == n_new
    end
end
