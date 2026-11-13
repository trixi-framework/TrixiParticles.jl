@testset verbose=true "Target Spacing" begin
    particle_spacing = 0.1
    min_spacing = 0.05
    n_particles = 10
    width = particle_spacing * n_particles
    height = particle_spacing * n_particles
    density = 257.0
    spacing_ratio = 1.05

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.5 * particle_spacing
    state_equation = StateEquationCole(sound_speed=10, reference_density=density,
                                       exponent=7)

    tank = RectangularTank(particle_spacing, (width, height), (width, height),
                           density, n_layers=1, faces=(true, true, true, false))

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                 tank.boundary.mass,
                                                 state_equation=state_equation,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length)
    boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio, min_spacing,
                                                   max_spacing=particle_spacing,
                                                   reference_density=density,
                                                   smoothing_length_factor=1.5)

    function create_fluid_system(; kwargs...)
        return WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                           smoothing_length, state_equation,
                                           density_calculator=SummationDensity(),
                                           particle_refinement=refinement, kwargs...)
    end

    # The default update strategy does not support inactive particles
    neighborhood_search = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())

    function semidiscretize_systems(fluid_system)
        semi = Semidiscretization(fluid_system, boundary_system; neighborhood_search)
        ode = semidiscretize(semi, (0, 0))
        v_ode, u_ode = ode.u0.x

        return ode.p.semi, v_ode, u_ode
    end

    fluid_system = create_fluid_system()
    semi, v_ode, u_ode = semidiscretize_systems(fluid_system)

    (; target_spacing, is_fixed) = fluid_system.cache
    center = argmin(vec(sum(abs2, tank.fluid.coordinates .- [width / 2, height / 2],
                            dims=1)))
    n_fluid = nparticles(fluid_system)

    @testset "Initialization From Mass" begin
        @test all(isapprox.(target_spacing, particle_spacing))

        # Without refinement, there is no refinement cache
        @test isempty(TrixiParticles.create_cache_refinement(tank.fluid, nothing))
    end

    @testset "Uniform Spacing" begin
        fill!(is_fixed, false)
        target_spacing .= particle_spacing

        TrixiParticles.update_target_spacing!(fluid_system, v_ode, u_ode, semi)

        # min(max_spacing, C_r * Δs) = max_spacing
        @test all(isapprox.(target_spacing, particle_spacing))
    end

    @testset "Coarsening of an Unfixed Region" begin
        fill!(is_fixed, false)
        target_spacing .= min_spacing

        TrixiParticles.update_target_spacing!(fluid_system, v_ode, u_ode, semi)

        # The global `max_spacing` caps the coarsening, not the neighborhood maximum
        @test all(isapprox.(target_spacing, spacing_ratio * min_spacing))
    end

    @testset "Persistent Graded Band" begin
        # Fix a strip of fine particles at the left, like the particles next to a boundary
        fixed_strip = tank.fluid.coordinates[1, :] .< 0.3
        n_fixed = count(fixed_strip)

        fill!(is_fixed, false)
        target_spacing .= particle_spacing
        target_spacing[fixed_strip] .= min_spacing
        is_fixed[fixed_strip] .= true

        TrixiParticles.update_target_spacing!(fluid_system, v_ode, u_ode, semi)

        # The fixed particles keep their spacing
        @test all(target_spacing[fixed_strip] .== min_spacing)

        n_refined_1 = count(<(particle_spacing), target_spacing)
        @test n_fixed < n_refined_1 < n_fluid

        # All refined particles are between the fixed spacing and `max_spacing`
        @test all(min_spacing .<= target_spacing .<= particle_spacing)

        # The spacing of the previous call is used, so the band grows
        TrixiParticles.update_target_spacing!(fluid_system, v_ode, u_ode, semi)
        n_refined_2 = count(<(particle_spacing), target_spacing)
        @test n_refined_2 > n_refined_1
        @test all(target_spacing[fixed_strip] .== min_spacing)
    end

    @testset "Geometric Mean" begin
        # Neighbors with 0.1 and the center with 0.05: ratio 2 > C_r^3,
        # so the geometric mean of the neighborhood is used.
        fill!(is_fixed, false)
        target_spacing .= particle_spacing
        target_spacing[center] = min_spacing

        _, _, spacing_avg,
        n = TrixiParticles.neighborhood_spacing(fluid_system, tank.fluid.coordinates,
                                                semi, center)

        TrixiParticles.update_target_spacing!(fluid_system, v_ode, u_ode, semi)

        @test target_spacing[center] ≈ spacing_avg
        @test min_spacing < spacing_avg < particle_spacing

        # With `n` neighbors (including the center), the geometric mean is
        # `0.1 * 0.5^(1 / n)`, which is smaller than the arithmetic mean `0.1 - 0.05 / n`
        @test spacing_avg ≈ particle_spacing * (min_spacing / particle_spacing)^(1 / n)
        @test spacing_avg < particle_spacing - (particle_spacing - min_spacing) / n

        # Only the neighbors within the compact support of the center are counted
        radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)
        distances = vec(sqrt.(sum(abs2,
                                  tank.fluid.coordinates .-
                                  tank.fluid.coordinates[:, center], dims=1)))
        @test n == count(<=(radius), distances)
    end

    @testset "Inactive Particles Are Skipped" begin
        fluid_system = create_fluid_system(buffer_size=2)
        semi, v_ode, u_ode = semidiscretize_systems(fluid_system)
        (; target_spacing, is_fixed) = fluid_system.cache
        inactive = (n_fluid + 1):(n_fluid + 2)

        target_spacing .= min_spacing
        TrixiParticles.update_target_spacing!(fluid_system, v_ode, u_ode, semi)

        # Only the active particles coarsen
        @test all(isapprox.(target_spacing[1:n_fluid], spacing_ratio * min_spacing))
        @test all(==(min_spacing), target_spacing[inactive])
    end

    @testset "Sorting Keeps the Target Spacing" begin
        fluid_system = create_fluid_system()
        (; target_spacing) = fluid_system.cache
        target_spacing .= 1:n_fluid
        perm = reverse(1:n_fluid)

        TrixiParticles.sort_variable_particle_data!(fluid_system, perm)

        @test target_spacing == reverse(1:n_fluid)
    end

    @testset "Resizing Keeps the Target Spacing" begin
        fluid_system = create_fluid_system(buffer_size=2)
        semi, v_ode, u_ode = semidiscretize_systems(fluid_system)
        (; target_spacing, target_spacing_tmp, is_fixed) = fluid_system.cache
        target_spacing .= min_spacing
        n_old = nparticles(fluid_system)

        TrixiParticles.resize_systems!(v_ode, u_ode, semi,
                                       [n_old + 3, nparticles(semi.systems[2])])

        @test length(target_spacing) == length(target_spacing_tmp) ==
              length(is_fixed) == n_old + 3
        @test all(==(min_spacing), target_spacing[1:n_old])

        # The new particles get the target spacing that corresponds to their mass
        @test all(==(particle_spacing), target_spacing[(n_old + 1):end])
        @test !any(is_fixed[(n_old + 1):end])
    end
end
