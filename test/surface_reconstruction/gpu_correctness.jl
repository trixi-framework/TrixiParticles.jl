@trixi_testset "surface reconstruction GPU cache correctness" begin
    # Compare a fresh CPU transfer to the callback's cached-neighbor-search path after
    # moving Float32 particles across cell boundaries and back. Merely comparing the
    # initial state would not detect a stale CPU neighborhood-search handler.
    using OrdinaryDiffEqLowStorageRK
    TP = TrixiParticles
    backend = Main.parallelization_backend
    spacing = 0.05f0
    shape = RectangularShape(spacing, (6, 5, 4), (0.1f0, 0.1f0, 0.1f0);
                             density=1000.0f0, coordinates_eltype=Float32)
    fluid = WeaklyCompressibleSPHSystem(shape; smoothing_kernel=WendlandC2Kernel{3}(),
                                        smoothing_length=1.5f0 * spacing,
                                        density_calculator=SummationDensity(),
                                        state_equation=StateEquationCole(;
                                                                         sound_speed=10.0f0,
                                                                         reference_density=1000.0f0,
                                                                         exponent=7))
    cells = FullGridCellList(; min_corner=(-0.5f0, -0.5f0, -0.5f0),
                             max_corner=(1.5f0, 1.5f0, 1.5f0))
    semi = Semidiscretization(fluid; parallelization_backend=backend,
                              neighborhood_search=GridNeighborhoodSearch{3}(;
                                                                            cell_list=cells))
    ode = semidiscretize(semi, (0.0f0, 0.001f0))
    device_semi = ode.p.semi
    v_device, u_device = ode.u0.x
    @test eltype(v_device) == eltype(u_device) == Float32
    initial_u = Array(u_device)
    state = reshape(collect(Float32, 1:length(v_device)), size(v_device))
    v_device .= TP.Adapt.adapt(backend, state .* 0.0001f0)
    # The GPU simulation updates its caches before the CPU adaptation performed by a
    # reconstruction event; use the same ordering here.
    TP.update_systems_and_nhs(v_device, u_device, device_semi, 0.0f0)
    _, _, first_cpu = TP.transfer2cpu(v_device, u_device, device_semi)
    cached_handler = first_cpu.neighborhood_search_handler

    # Move more than a cell width, and then move back. A same-state comparison would
    # not exercise updates of the cached CPU search.
    for shift in (0.0f0, 0.2f0, -0.05f0)
        coordinates = reshape(copy(initial_u), 3, :)
        coordinates[1, :] .+= shift
        u_device .= TP.Adapt.adapt(backend, vec(coordinates))
        TP.update_systems_and_nhs(v_device, u_device, device_semi, 0.0f0)
        v_full, u_full, full_cpu = TP.transfer2cpu(v_device, u_device, device_semi)
        v_cached, u_cached,
        cached_cpu = TP.transfer2cpu_system_state(v_device, u_device,
                                                  device_semi)
        @test isnothing(cached_cpu.neighborhood_search_handler)
        cached_cpu = TrixiParticles.@set cached_cpu.neighborhood_search_handler = cached_handler
        @test cached_handler !== full_cpu.neighborhood_search_handler
        @test v_full == v_cached && u_full == u_cached
        full_fluid, cached_fluid = full_cpu.systems[1], cached_cpu.systems[1]
        points = TP.active_surface_points(full_fluid,
                                          TP.wrap_u(u_full, full_fluid, full_cpu))
        @test points == TP.active_surface_points(cached_fluid,
                                       TP.wrap_u(u_cached, cached_fluid, cached_cpu))
        @test TP.particle_volumes(full_fluid, TP.wrap_v(v_full, full_fluid, full_cpu)) ==
              TP.particle_volumes(cached_fluid,
                                  TP.wrap_v(v_cached, cached_fluid, cached_cpu))
        queries = points[:, 1:7:end] .+ 0.002
        full = interpolate_points(queries, full_cpu, full_fluid, v_full, u_full;
                                  cut_off_bnd=false)
        cached = interpolate_points(queries, cached_cpu, cached_fluid, v_cached, u_cached;
                                    cut_off_bnd=false)
        @test full.neighbor_count == cached.neighbor_count
        @test all(quantity -> all(isfinite, getproperty(full, quantity)) &&
                              all(isfinite, getproperty(cached, quantity)),
                  (:velocity, :pressure, :density))
        for quantity in (:velocity, :pressure, :density)
            expected, actual = getproperty(full, quantity), getproperty(cached, quantity)
            # GPU and CPU cell updates may enumerate neighbors differently. Float32
            # interpolation is compared at a summation-error scale, not bitwise.
            @test actual≈expected rtol=128eps(Float32) atol=128eps(Float32)
        end
    end
end
