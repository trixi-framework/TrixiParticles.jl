using OrdinaryDiffEqLowStorageRK

@testset verbose=true "Particle Sorting" begin
    # A small tank with shuffled fluid particles, so that sorting actually reorders them.
    function sorting_setup(; sort_particles, neighborhood_search_handler=nothing,
                           buffer_size=nothing, moving_wall=false,
                           boundary_model_type=:adami, fluid_system_type=:wcsph,
                           shifting_technique=nothing)
        particle_spacing = 0.05
        tank = RectangularTank(particle_spacing, (1.0, 0.6), (2.0, 1.0), 1000.0,
                               n_layers=3)

        n_fluid = nparticles(tank.fluid)
        perm = Random.randperm(Random.Xoshiro(1), n_fluid)
        perturbation = 0.01 * particle_spacing * randn(Random.Xoshiro(2), 2, n_fluid)
        coordinates = tank.fluid.coordinates[:, perm] .+ perturbation
        velocity = 0.1 * randn(Random.Xoshiro(3), 2, n_fluid)
        if isnothing(buffer_size)
            density = tank.fluid.density[perm] .+ randn(Random.Xoshiro(4), n_fluid)
        else
            # The density must be constant with a buffer.
            density = tank.fluid.density

            # Let some particles leave the domain, so that they are deactivated.
            # They are placed to the right of the tank, outside of the compact support
            # of the wall particles, and quickly leave the grid of the neighborhood search.
            coordinates[:, 1:3] .= [2.25 2.3 2.35
                                    0.8 0.8 0.8]
            velocity[:, 1:3] .= [30.0, 0.0]
        end
        mass = tank.fluid.mass[perm] .* (1 .+ 0.01 * rand(Random.Xoshiro(5), n_fluid))
        fluid = InitialCondition(; coordinates, velocity, density, mass, particle_spacing)

        smoothing_length = 1.2 * particle_spacing
        smoothing_kernel = WendlandC2Kernel{2}()
        state_equation = StateEquationCole(; sound_speed=20.0, reference_density=1000.0,
                                           exponent=7)
        viscosity = ArtificialViscosityMonaghan(alpha=0.02)

        if fluid_system_type === :wcsph
            density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
            fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                       smoothing_length,
                                                       density_calculator=ContinuityDensity(),
                                                       state_equation, viscosity,
                                                       density_diffusion, buffer_size,
                                                       shifting_technique,
                                                       acceleration=(0.0, -9.81))
        else
            fluid_system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                       smoothing_length, sound_speed=20.0,
                                                       viscosity, acceleration=(0.0, -9.81))
        end

        if boundary_model_type === :adami
            boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                         tank.boundary.mass,
                                                         AdamiPressureExtrapolation(),
                                                         smoothing_kernel,
                                                         smoothing_length;
                                                         state_equation)
        elseif boundary_model_type === :summation
            boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                         tank.boundary.mass,
                                                         SummationDensity(),
                                                         smoothing_kernel,
                                                         smoothing_length;
                                                         state_equation)
        else
            boundary_model = BoundaryModelMonaghanKajtar(10.0, 1.0, particle_spacing,
                                                         tank.boundary.mass)
        end

        if moving_wall !== false
            # Only move every other particle with `moving_wall=:partial`, so that
            # the indices of the moving particles have to be updated when sorting.
            n_boundary = nparticles(tank.boundary)
            moving_particles = moving_wall === :partial ? collect(1:2:n_boundary) : nothing
            prescribed_motion = PrescribedMotion((x,
                                                  t) -> x +
                                                        SVector(0.0, 0.01 * sin(10t)),
                                                 t -> true; moving_particles)
        else
            prescribed_motion = nothing
        end
        boundary_system = WallBoundarySystem(tank.boundary, boundary_model;
                                             prescribed_motion)

        min_corner = (-0.5, -0.5)
        max_corner = (2.5, 1.5)
        if sort_particles
            neighborhood_search = SortingGridNeighborhoodSearch{2}(; min_corner, max_corner)
        else
            cell_list = FullGridCellList(; min_corner, max_corner)
            update_strategy = ParallelUpdate()
            neighborhood_search = GridNeighborhoodSearch{2}(; cell_list, update_strategy)
        end
        semi = Semidiscretization(fluid_system, boundary_system; neighborhood_search,
                                  neighborhood_search_handler)

        return semi
    end

    function solve_setup(ode; kwargs...)
        return solve(ode, CarpenterKennedy2N54(williamson_condition=false); dt=1e-4,
                     adaptive=false, save_everystep=false, kwargs...)
    end

    setups = [
        (;),
        (; buffer_size=10),
        (; moving_wall=true),
        (; moving_wall=:partial),
        (; boundary_model_type=:summation),
        (; boundary_model_type=:monaghan_kajtar),
        (; shifting_technique=ConsistentShiftingSun2019())
    ]
    names = [
        "Default",
        "Buffer",
        "Moving Wall",
        "Partially Moving Wall",
        "`SummationDensity` Boundary",
        "`BoundaryModelMonaghanKajtar`",
        "Shifting Velocity"
    ]

    @testset verbose=true "$(names[i])" for i in eachindex(setups)
        semi_reference = sorting_setup(; sort_particles=false, setups[i]...)
        semi = sorting_setup(; sort_particles=true, setups[i]...)
        mass_before = copy(semi.systems[1].mass)

        ode_reference = semidiscretize(semi_reference, (0.0, 0.05))
        ode = @test_logs (:info, r"To sort particles") semidiscretize(semi, (0.0, 0.05))
        @test TrixiParticles.particle_sorting(ode.p.semi) isa
              TrixiParticles.ParticleSorting
        @test isnothing(TrixiParticles.particle_sorting(ode_reference.p.semi))

        # The passed systems are not modified.
        @test ode.p.semi.systems[1] !== semi.systems[1]

        # The particles are actually reordered.
        @test ode.p.semi.systems[1].mass != mass_before
        @test sort(ode.p.semi.systems[1].mass) == sort(mass_before)

        @testset "`kick!` and `drift!`" begin
            v_ode, u_ode = ode.u0.x
            dv_ode = similar(v_ode)
            du_ode = similar(u_ode)
            dv_ode_reference = similar(v_ode)
            du_ode_reference = similar(u_ode)

            TrixiParticles.kick!(dv_ode_reference, ode_reference.u0.x...,
                                 ode_reference.p, 0.0)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

            # The result is in the order of the integration arrays.
            @test isapprox(dv_ode, dv_ode_reference, rtol=1e-14)

            TrixiParticles.drift!(du_ode_reference, ode_reference.u0.x...,
                                  ode_reference.p, 0.0)
            TrixiParticles.drift!(du_ode, v_ode, u_ode, ode.p, 0.0)

            # Only the active particles are written in `drift!`.
            fluid_system = ode.p.semi.systems[1]
            n_active = length(TrixiParticles.each_active_particle(fluid_system))
            du = TrixiParticles.wrap_u_unchecked(du_ode, fluid_system,
                                                 ode.p.semi)
            du_reference = TrixiParticles.wrap_u(du_ode_reference,
                                                 ode_reference.p.semi.systems[1],
                                                 ode_reference.p.semi)
            active_particles = TrixiParticles.each_active_particle(ode_reference.p.semi.systems[1])
            # The shifting velocity is only equal up to round-off errors.
            @test isapprox(du[:, active_particles], du_reference[:, active_particles],
                           rtol=1e-14)
        end

        @testset "Time Integration" begin
            sol_reference = solve_setup(ode_reference)
            sol = solve_setup(ode)

            # Only the fluid system has integrated particles.
            # We only compare the active particles. Particles are deactivated in the RHS,
            # which is fine because the active state is stored in the system,
            # and the data of inactive particles is not used.
            # However, `deactivate_particle!` also resets `v` and `u` of the deactivated
            # particle in the arrays passed to the RHS. When particles are sorted,
            # these are the sorted copies, so the integration arrays still contain the
            # old data of inactive particles.
            # Therefore, comparing the whole integration arrays would fail.
            semi_sorted = sol.prob.p.semi
            fluid_system = semi_sorted.systems[1]
            semi_reference = sol_reference.prob.p.semi
            fluid_reference = semi_reference.systems[1]
            particles = TrixiParticles.each_active_particle(fluid_reference)

            v_ode, u_ode = sol.u[end].x
            v_ode_reference, u_ode_reference = sol_reference.u[end].x

            v = TrixiParticles.wrap_v_unchecked(v_ode, fluid_system, semi_sorted)
            u = TrixiParticles.wrap_u_unchecked(u_ode, fluid_system, semi_sorted)
            v_reference = TrixiParticles.wrap_v(v_ode_reference, fluid_reference, semi_reference)
            u_reference = TrixiParticles.wrap_u(u_ode_reference, fluid_reference, semi_reference)

            @test isapprox(v[:, particles], v_reference[:, particles], rtol=1e-12)
            @test isapprox(u[:, particles], u_reference[:, particles], rtol=1e-12)

            # The integration arrays must be copied to system order before wrapping them.
            @test_throws ArgumentError TrixiParticles.wrap_u(sol.u[end].x[2], fluid_system,
                                                             semi_sorted)
            @test_throws ArgumentError TrixiParticles.wrap_v(sol.u[end].x[1], fluid_system,
                                                             semi_sorted)

            # The sorted copies are in the sorted order of the system arrays.
            _, u_ode_sorted = TrixiParticles.sorted_ode_arrays(sol.u[end].x...,
                                                               semi_sorted)
            u = TrixiParticles.wrap_u(u_ode_sorted, fluid_system, semi_sorted)
            u_unsorted = TrixiParticles.wrap_u_unchecked(sol.u[end].x[2],
                                                         fluid_system,
                                                         semi_sorted)
            original_index = semi_sorted.particle_sorting.original_index[1]
            @test u == u_unsorted[:, original_index]
        end
    end

    @testset verbose=true "VTK Output ($(names[i]))" for i in 1:2
        # The second setup uses a buffer, where some particles are deactivated
        # during the simulation.
        semi_reference = sorting_setup(; sort_particles=false, setups[i]...)
        semi = sorting_setup(; sort_particles=true, setups[i]...)
        ode_reference = semidiscretize(semi_reference, (0.0, 0.05))
        ode = semidiscretize(semi, (0.0, 0.05))

        mktempdir() do tmp_dir
            sol_reference = solve_setup(ode_reference)
            sol = solve_setup(ode)

            if haskey(setups[i], :buffer_size)
                # Make sure that particles have actually been deactivated.
                # The buffer particles are inactive from the start.
                fluid_system = sol.prob.p.semi.systems[1]
                n_active = length(TrixiParticles.each_active_particle(fluid_system))
                @test n_active < nparticles(fluid_system) - setups[i].buffer_size
            end

            trixi2vtk(sol_reference.u[end], sol_reference.prob.p.semi, 0.0,
                      output_directory=tmp_dir, prefix="reference")
            trixi2vtk(sol.u[end], sol.prob.p.semi, 0.0, output_directory=tmp_dir,
                      prefix="sorted")

            # The particles are written in the same order as without sorting.
            for system_name in ("fluid_1", "boundary_1")
                ic_reference = vtk2trixi(joinpath(tmp_dir,
                                                  "reference_$(system_name)_current.vtu"))
                ic = vtk2trixi(joinpath(tmp_dir, "sorted_$(system_name)_current.vtu"))

                @test isapprox(ic.coordinates, ic_reference.coordinates, rtol=1e-12)
                @test isapprox(ic.velocity, ic_reference.velocity, rtol=1e-12)
                @test isapprox(ic.density, ic_reference.density, rtol=1e-12)
                @test isapprox(ic.pressure, ic_reference.pressure, rtol=1e-12)
            end
        end
    end

    @testset verbose=true "Reorder Integration Arrays" begin
        semi = sorting_setup(; sort_particles=true)
        ode = semidiscretize(semi, (0.0, 0.05))
        sol = solve_setup(ode)
        semi_sorted = sol.prob.p.semi
        fluid_system = semi_sorted.systems[1]

        v_ode, u_ode = copy(sol.u[end]).x
        dv_ode_reference = similar(v_ode)
        TrixiParticles.kick!(dv_ode_reference, v_ode, u_ode, sol.prob.p, 0.0)

        # `dv` and `u` in system order.
        original_index = semi_sorted.particle_sorting.original_index[1]
        dv_reference = TrixiParticles.wrap_v_unchecked(dv_ode_reference,
                                                       fluid_system,
                                                       semi_sorted)[:, original_index]
        u_reference = TrixiParticles.wrap_u_unchecked(u_ode, fluid_system,
                                                      semi_sorted)[:, original_index]

        # This is called by the `SortingCallback`.
        TrixiParticles.apply_system_order!(v_ode, u_ode, semi_sorted)

        # The integration arrays are now in the order of the system arrays.
        @test semi_sorted.particle_sorting.original_index[1] == eachparticle(fluid_system)
        @test TrixiParticles.wrap_u_unchecked(u_ode, fluid_system, semi_sorted) ==
              u_reference

        dv_ode = similar(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, sol.prob.p, 0.0)
        @test isapprox(TrixiParticles.wrap_v_unchecked(dv_ode, fluid_system,
                                                       semi_sorted),
                       dv_reference, rtol=1e-14)
    end

    # Particle shifting in the `UpdateCallback` and the density reinitialization
    # modify the integration arrays, which have to be copied back
    # from the sorted copies.
    # With `ParticleShiftingTechnique()`, the shifting velocity is computed in the
    # `UpdateCallback` and used in all stages of the next time step, so it has to be
    # reordered when the particles are sorted.
    shifting_techniques = (ParticleShiftingTechniqueSun2017(), ParticleShiftingTechnique())
    shifting_names = ("Position Correction", "Integrated Shifting Velocity")
    @testset verbose=true "Callbacks with $(shifting_names[i])" for i in eachindex(shifting_techniques)
        shifting_technique = shifting_techniques[i]
        semi_reference = sorting_setup(; sort_particles=false, shifting_technique)
        semi = sorting_setup(; sort_particles=true, shifting_technique)
        ode_reference = semidiscretize(semi_reference, (0.0, 0.02))
        ode = semidiscretize(semi, (0.0, 0.02))

        # Pairs the acceleration with the masses stored in the system to test
        # that `dv_ode` is passed in system order.
        function mass_weighted_acceleration(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
            dv = TrixiParticles.wrap_v(dv_ode, system, semi)
            particles = TrixiParticles.each_active_particle(system)

            return sum(system.mass[particle] * dv[2, particle] for particle in particles)
        end

        function callbacks(semi)
            fluid_system = semi.systems[1]
            postprocess = PostprocessCallback(; interval=5, write_csv=false,
                                              write_json=false, kinetic_energy,
                                              avg_pressure, mass_weighted_acceleration)
            density_reinit = DensityReinitializationCallback(fluid_system, semi;
                                                             interval=7)
            stepsize = StepsizeCallback(cfl=0.05)

            return postprocess,
                   CallbackSet(UpdateCallback(), postprocess, density_reinit, stepsize)
        end

        pp_reference, callbacks_reference = callbacks(ode_reference.p.semi)
        pp, callbacks_sorted = callbacks(ode.p.semi)

        sol_reference = solve_setup(ode_reference; callback=callbacks_reference)
        sol = solve_setup(ode; callback=callbacks_sorted)

        @test isapprox(sol.u[end], sol_reference.u[end], rtol=1e-12)

        data = pp.affect!.data
        data_reference = pp_reference.affect!.data
        @test !isempty(data)
        @test keys(data) == keys(data_reference)
        for key in keys(data)
            @test isapprox(data[key], data_reference[key], rtol=1e-12)
        end
    end

    @testset verbose=true "`SortingCallback`" begin
        semi_reference = sorting_setup(; sort_particles=false)
        semi = sorting_setup(; sort_particles=true)
        ode_reference = semidiscretize(semi_reference, (0.0, 0.05))
        ode = semidiscretize(semi, (0.0, 0.05))

        sol_reference = solve_setup(ode_reference)
        sol = solve_setup(ode; callback=SortingCallback(interval=10))

        # The integration arrays are reordered, so we compare the sorted coordinates.
        fluid_system = sol.prob.p.semi.systems[1]
        u = TrixiParticles.wrap_u_unchecked(sol.u[end].x[2], fluid_system,
                                            sol.prob.p.semi)
        u_reference = TrixiParticles.wrap_u(sol_reference.u[end].x[2],
                                            sol_reference.prob.p.semi.systems[1],
                                            sol_reference.prob.p.semi)
        @test isapprox(sortslices(u, dims=2), sortslices(u_reference, dims=2), rtol=1e-6)
    end

    @testset verbose=true "Other Neighborhood Search Handlers" begin
        # Particles can only be sorted with the `CombinedNHSHandler`.
        error_str = "a neighborhood search that sorts points, like the " *
                    "`SortingGridNeighborhoodSearch`, can only be used with the " *
                    "`CombinedNHSHandler`, which requires all pairs of systems " *
                    "to use the same search radius"
        @testset "$handler" for handler in (SharedNHSHandler, PairsNHSHandler)
            @test_throws ArgumentError(error_str) begin
                sorting_setup(; sort_particles=true, neighborhood_search_handler=handler)
            end
        end
    end

    @testset "Unsupported System" begin
        semi = sorting_setup(; sort_particles=true, fluid_system_type=:edac)
        error_str = "sorting particles with a neighborhood search that sorts points is " *
                    "not (yet) supported for systems of type EntropicallyDampedSPHSystem"
        @test_throws ArgumentError(error_str) semidiscretize(semi, (0.0, 0.05))
    end
end
