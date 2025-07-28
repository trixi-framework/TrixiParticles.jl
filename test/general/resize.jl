@testset verbose=true "Resize Semidiscretization" begin
    ics = [RectangularShape(0.1, (10, 10), (x, y); density=1)
           for x in [0.0, 1.0], y in [0.0, 1.0, 2.0]]

    systems = WeaklyCompressibleSPHSystem.(vec(ics), [SummationDensity()], nothing,
                                           [SchoenbergCubicSplineKernel{ndims(ics)}()], 1.0)

    semi = Semidiscretization(systems...)

    ode = semidiscretize(semi, (0, 0))

    v_ode, u_ode = ode.u0.x

    origin_nparticles = [nparticles(system) for system in systems]

    @testset verbose=true "`deleteat!`" begin
        delete_candidates = [45:49, 0:0, 78:100, 0:0, 1:15, 1:99]
        resized_nparticles = origin_nparticles .- [5, 0, 23, 0, 15, 99]

        for (i, system) in enumerate(systems)
            delete_candidates[i] == 0:0 && continue

            resize!(system.cache.delete_candidates, nparticles(system))
            system.cache.delete_candidates .= false
            system.cache.delete_candidates[delete_candidates[i]] .= true
        end

        deleteat!(semi, v_ode, u_ode, copy(v_ode), copy(u_ode))

        # We can't compare the coordinates, since the order of the particle IDs has changed.
        # However, plotting both shows that they coincide:
        #
        # origin_coords = [system.initial_condition.coordinates for system in systems]
        # iterators = [setdiff(1:100, i) for i in delete_candidates]
        # test_coords = copy.(view.(origin_coords, :, iterators))
        # test_ics = InitialCondition{ndims(ics)}.(test_coords, zero.(test_coords), [1], [1], [1],
        #                                          [0.1])
        # p2 = plot(v_ode, u_ode, semi, size=(800, 800), legend=nothing)
        # plot!(p2, test_ics..., size=(800, 800), legend=nothing)

        coords = [
            [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.95 0.85 0.75 0.65 0.55 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45;
             0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.45 0.45 0.45 0.45 0.95 0.95 0.95 0.95 0.95 0.45 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.85 0.85 0.85 0.85 0.85 0.85 0.85 0.85 0.85 0.85 0.95 0.95 0.95 0.95 0.95],
            systems[2].initial_condition.coordinates,
            [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65;
             1.05 1.05 1.05 1.05 1.05 1.05 1.05 1.05 1.05 1.05 1.15 1.15 1.15 1.15 1.15 1.15 1.15 1.15 1.15 1.15 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.25 1.35 1.35 1.35 1.35 1.35 1.35 1.35 1.35 1.35 1.35 1.45 1.45 1.45 1.45 1.45 1.45 1.45 1.45 1.45 1.45 1.55 1.55 1.55 1.55 1.55 1.55 1.55 1.55 1.55 1.55 1.65 1.65 1.65 1.65 1.65 1.65 1.65 1.65 1.65 1.65 1.75 1.75 1.75 1.75 1.75 1.75 1.75],
            systems[4].initial_condition.coordinates,
            [0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05 0.95 0.85 0.75 0.65 0.55 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.05 0.15 0.25 0.35 0.45;
             2.95 2.95 2.95 2.95 2.95 2.95 2.95 2.95 2.95 2.95 2.85 2.85 2.85 2.85 2.85 2.15 2.15 2.15 2.15 2.15 2.25 2.25 2.25 2.25 2.25 2.25 2.25 2.25 2.25 2.25 2.35 2.35 2.35 2.35 2.35 2.35 2.35 2.35 2.35 2.35 2.45 2.45 2.45 2.45 2.45 2.45 2.45 2.45 2.45 2.45 2.55 2.55 2.55 2.55 2.55 2.55 2.55 2.55 2.55 2.55 2.65 2.65 2.65 2.65 2.65 2.65 2.65 2.65 2.65 2.65 2.75 2.75 2.75 2.75 2.75 2.75 2.75 2.75 2.75 2.75 2.85 2.85 2.85 2.85 2.85],
            [1.95;
             2.95]
        ]

        @testset verbose=true "`system $i" for i in eachindex(systems)
            @test nparticles(systems[i]) == resized_nparticles[i]
            @test isapprox(coords[i], TrixiParticles.wrap_u(u_ode, systems[i], semi))
        end
    end

    @testset verbose=true "`resize!`" begin
        additional_capacities = [5, 0, 23, 0, 15, 99]
        @testset verbose=true "`system $i" for i in eachindex(systems)
            systems[i].cache.additional_capacity[] = additional_capacities[i]
            resize!(semi, v_ode, u_ode, copy(v_ode), copy(u_ode))

            @test nparticles(systems[i]) == origin_nparticles[i]
        end
    end
end
