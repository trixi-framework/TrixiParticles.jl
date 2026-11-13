@testset verbose=true "Refinement Criteria" begin
    particle_spacing = 0.1
    n_particles = 10
    width = particle_spacing * n_particles
    height = particle_spacing * n_particles
    density = 257.0

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.5 * particle_spacing
    state_equation = StateEquationCole(sound_speed=10, reference_density=density,
                                       exponent=7)

    # The wall spacing is `particle_spacing / 2 = 0.05`
    tank = RectangularTank(particle_spacing, (width, height), (width, height),
                           density, n_layers=2, faces=(true, true, true, false),
                           spacing_ratio=2)
    wall_spacing = tank.boundary.particle_spacing

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                 tank.boundary.mass,
                                                 state_equation=state_equation,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel, smoothing_length)
    boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

    # The default update strategy does not support inactive particles
    neighborhood_search = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())

    function setup(refinement_criteria; kwargs...)
        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.05,
                                                       min_spacing=wall_spacing,
                                                       max_spacing=particle_spacing,
                                                       reference_density=density,
                                                       smoothing_length_factor=1.5,
                                                       refinement_criteria)

        fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                   smoothing_length, state_equation,
                                                   density_calculator=SummationDensity(),
                                                   particle_refinement=refinement,
                                                   kwargs...)

        semi = Semidiscretization(fluid_system, boundary_system; neighborhood_search)
        ode = semidiscretize(semi, (0, 0))
        v_ode, u_ode = ode.u0.x

        return fluid_system, ode.p.semi, v_ode, u_ode
    end

    # Fluid particles closer to the wall than the kernel support
    y = tank.fluid.coordinates[2, :]
    x = tank.fluid.coordinates[1, :]
    distance_to_wall = min.(x, width .- x, y)
    near_wall = distance_to_wall .< 0.5 * particle_spacing
    far_from_wall = distance_to_wall .> 3 * smoothing_length

    solution_criterion(target) = TrixiParticles.SolutionRefinementCriterion((system,
                                                                             particle,
                                                                             v, u,
                                                                             semi) -> target)

    @testset "Default Criterion" begin
        refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.05,
                                                       min_spacing=wall_spacing,
                                                       max_spacing=particle_spacing,
                                                       reference_density=density)

        @test refinement.criteria == (TrixiParticles.SpatialRefinementCriterion(),)
    end

    @testset "Spatial Criterion" begin
        fluid_system, semi, v_ode,
        u_ode = setup(TrixiParticles.SpatialRefinementCriterion())
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(is_fixed[near_wall])
        @test all(target_spacing[near_wall] .== wall_spacing)

        @test !any(is_fixed[far_from_wall])
        @test all(target_spacing[far_from_wall] .== particle_spacing)

        # The spacing of the previous call is overwritten near boundaries
        # and kept everywhere else
        target_spacing .= 0.02
        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(target_spacing[near_wall] .== wall_spacing)
        @test all(target_spacing[far_from_wall] .== 0.02)
    end

    @testset "Solution Criterion Finer Than Boundary" begin
        target = 0.03

        # The solution criterion is listed first, but the spatial criterion is applied first
        fluid_system, semi, v_ode,
        u_ode = setup((solution_criterion(target),
                       TrixiParticles.SpatialRefinementCriterion()))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # As in the code of Haftu et al., the solution criterion does not apply to
        # particles near boundaries, so they keep the boundary spacing
        @test all(target_spacing[near_wall] .== wall_spacing)

        # All other particles take the target spacing and are fixed
        @test all(is_fixed)
        @test all(target_spacing[far_from_wall] .== target)

        # A finer spacing of the previous call is kept away from boundaries,
        # but overwritten by the boundary spacing near boundaries
        target_spacing .= 0.02
        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)
        @test all(target_spacing[far_from_wall] .== 0.02)
        @test all(target_spacing[near_wall] .== wall_spacing)
    end

    @testset "Solution Criterion Coarser Than Boundary" begin
        # Coarser than the wall spacing, but finer than `max_spacing`
        target = 0.08

        fluid_system, semi, v_ode,
        u_ode = setup((TrixiParticles.SpatialRefinementCriterion(),
                       solution_criterion(target)))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(target_spacing[near_wall] .== wall_spacing)
        @test all(target_spacing[far_from_wall] .== target)
        @test all(is_fixed)
    end

    @testset "Solution Criterion Without Spatial Criterion" begin
        target = 0.03
        fluid_system, semi, v_ode, u_ode = setup(solution_criterion(target))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # Without a spatial criterion, the boundary spacing is not used
        @test all(is_fixed)
        @test all(target_spacing .== target)
    end

    @testset "Multiple Solution Criteria" begin
        # The criterion for the particles with `x < 0.5` requests `max_spacing`,
        # so it is only active for the particles with `x >= 0.5`
        region_criterion = TrixiParticles.SolutionRefinementCriterion((system, particle,
                                                                       v, u,
                                                                       semi) -> u[1,
                                                                                  particle] <
                                                                                0.5 ?
                                                                                particle_spacing :
                                                                                0.03)

        fluid_system, semi, v_ode,
        u_ode = setup((solution_criterion(0.08), region_criterion))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # The smallest requested spacing is used
        @test all(is_fixed)
        @test all(target_spacing[x .< 0.5] .== 0.08)
        @test all(target_spacing[x .>= 0.5] .== 0.03)
    end

    @testset "Inactive Solution Criterion" begin
        fluid_system, semi, v_ode, u_ode = setup(solution_criterion(particle_spacing))
        (; target_spacing, is_fixed) = fluid_system.cache

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        # A target spacing of `max_spacing` does not fix any particle
        @test !any(is_fixed)
        @test all(target_spacing .== particle_spacing)
    end

    @testset "Inactive Particles Are Skipped" begin
        fluid_system, semi, v_ode,
        u_ode = setup((TrixiParticles.SpatialRefinementCriterion(),
                       solution_criterion(0.03)); buffer_size=2)
        (; target_spacing, is_fixed) = fluid_system.cache
        n_fluid = nparticles(tank.fluid)
        inactive = (n_fluid + 1):(n_fluid + 2)

        TrixiParticles.apply_refinement_criteria!(fluid_system, v_ode, u_ode, semi)

        @test all(is_fixed[1:n_fluid])
        @test !any(is_fixed[inactive])
        @test all(target_spacing[inactive] .== particle_spacing)
    end
end
