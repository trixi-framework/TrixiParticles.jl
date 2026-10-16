@trixi_testset "PST does not regularize a particle against itself" begin
    TP = TrixiParticles
    n = 12
    dx = 1 / n
    kernel = WendlandC6Kernel{2}()
    particles = RectangularShape(dx, (n, n), (0.0, 0.0); density=1.0, velocity=(1.0, 0.0))
    coordinates = particles.coordinates .+
                  0.15dx .*
                  (2rand(Random.MersenneTwister(11), size(particles.coordinates)...) .- 1)
    ic = InitialCondition(; coordinates, density=ones(n^2), mass=fill(dx^2, n^2),
                          velocity=repeat([1.0, 0.0], 1, n^2), particle_spacing=dx)

    function cloud_quality(system, semi, u)
        raw_density = zeros(nparticles(system))
        separation = fill(Inf, nparticles(system))
        x = TP.current_coordinates(u, system)
        TP.foreach_point_neighbor(system, system, x, x, semi) do a, b, diff, distance
            raw_density[a] += system.mass[b] * TP.kernel(kernel, distance, 2dx)
            a == b || (separation[a] = min(separation[a], distance))
        end
        return sqrt(sum(abs2, raw_density .- 1) / length(raw_density)),
               minimum(separation) / dx
    end

    for correction in (KernelCorrection(), MixedKernelGradientCorrection()),
        density_calculator in (SummationDensity(), ContinuityDensity())
        shifting = ParticleShiftingTechniqueSun2017()
        system = EntropicallyDampedSPHSystem(ic; smoothing_kernel=kernel,
                                             smoothing_length=2dx, sound_speed=15.0,
                                             density_calculator, correction,
                                             pressure_acceleration=nothing,
                                             average_pressure_reduction=false,
                                             shifting_technique=shifting)
        box = PeriodicBox(; min_corner=[0.0, 0.0], max_corner=[1.0, 1.0])
        search = GridNeighborhoodSearch{2}(; periodic_box=box,
                                           update_strategy=SerialUpdate())
        semi = Semidiscretization(system; neighborhood_search=search,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.5); reset_threads=false)
        semi = ode.p.semi
        system = first(semi.systems)
        v, u = Array(ode.u0.x[1]), Array(ode.u0.x[2])
        TP.update_systems_and_nhs(v, u, semi, 0.0)
        maximum_shift = 0.0
        dt = 0.2 * 2dx / 16
        GC.@preserve v u begin
            uf = TP.wrap_u(u, system, semi)
            initial_error, _ = cloud_quality(system, semi, uf)
            # The physical consistency operator still needs its nonzero self-gradient.
            self_gradients = [norm(TP.smoothing_kernel_grad(system, SVector(0.0, 0.0), 0.0,
                                                            a))
                              for a in eachparticle(system)]
            @test maximum(self_gradients) > 1e-6
            for step in 1:120
                TP.update_systems_and_nhs(v, u, semi, (step - 1) * dt)
                TP.update_shifting_from_callback!(system, shifting, v, u, semi)
                maximum_shift = max(maximum_shift,
                                    dt / dx * maximum(norm, eachcol(system.cache.delta_v)))
                TP.apply_particle_shifting!(u, shifting, system, semi, dt)
                TP.update_systems_and_nhs(v, u, semi, step * dt)
                _, separation = cloud_quality(system, semi, uf)
                separation < 0.05 && break
            end
            final_error, separation = cloud_quality(system, semi, uf)
            @test final_error < initial_error
            @test separation > 0.5
            @test maximum_shift < 0.05
        end
    end
end
