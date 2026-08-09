
@testset verbose=true "Surface Tension" begin
    function build_wetted_area_setup(; solver=:wcsph, angle=60.0,
                                     contact=true, ELTYPE=Float64,
                                     smoothing_kernel=WendlandC2Kernel{3}(),
                                     smoothing_length_ratio=1.4,
                                     density_calculator=ContinuityDensity(),
                                     surface_tension_model=:momentum,
                                     provide_surface_measure=true,
                                     provide_normals=true,
                                     surface_measure_mode=:connected,
                                     boundary_kind=:wall,
                                     prescribed_motion=nothing,
                                     rotation=nothing,
                                     fluid_color=1, boundary_color=0)
        particle_spacing = ELTYPE(0.1)
        smoothing_length = ELTYPE(smoothing_length_ratio) * particle_spacing
        reference_density = ELTYPE(1000)
        fluid_raw = RectangularShape(particle_spacing, (4, 4, 3),
                                     (zero(ELTYPE), zero(ELTYPE), zero(ELTYPE));
                                     density=reference_density)
        transform = isnothing(rotation) ? Matrix{ELTYPE}(I, 3, 3) : ELTYPE.(rotation)
        fluid = InitialCondition(; coordinates=transform * fluid_raw.coordinates,
                                 velocity=transform * fluid_raw.velocity,
                                 mass=fluid_raw.mass, density=fluid_raw.density,
                                 pressure=fluid_raw.pressure,
                                 particle_spacing)
        state_equation = StateEquationCole(; sound_speed=ELTYPE(10), reference_density,
                                           exponent=1)
        contact_model = contact ? WettedAreaContactAngle(ELTYPE(angle)) : nothing
        normal_method = ColorfieldSurfaceNormal(; boundary_contact_threshold=zero(ELTYPE),
                                                interface_threshold=ELTYPE(0.01),
                                                ideal_density_threshold=ELTYPE(0.95),
                                                contact_model)
        surface_tension = surface_tension_model == :momentum ?
                          SurfaceTensionMomentumMorris(;
                                                       surface_tension_coefficient=ELTYPE(0.072)) :
                          SurfaceTensionMorris(;
                                               surface_tension_coefficient=ELTYPE(0.072))
        fluid_system = if solver == :wcsph
            WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                        density_calculator, state_equation, surface_tension,
                                        surface_normal_method=normal_method,
                                        reference_particle_spacing=particle_spacing,
                                        color_value=fluid_color)
        else
            EntropicallyDampedSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                        sound_speed=ELTYPE(10), density_calculator,
                                        surface_tension,
                                        surface_normal_method=normal_method,
                                        reference_particle_spacing=particle_spacing,
                                        color_value=fluid_color)
        end

        boundary_raw = RectangularShape(particle_spacing, (4, 4, 3),
                                        (zero(ELTYPE), zero(ELTYPE),
                                         -3particle_spacing);
                                        density=reference_density)
        exposed_height = maximum(boundary_raw.coordinates[3, :])
        exposed = isapprox.(boundary_raw.coordinates[3, :], exposed_height;
                            atol=eps(ELTYPE))
        normals = zeros(ELTYPE, size(boundary_raw.coordinates))
        normals[3, exposed] .= -particle_spacing / 2
        surface_measure = zeros(ELTYPE, nparticles(boundary_raw))
        if surface_measure_mode == :connected
            surface_measure[exposed] .= particle_spacing^2
        elseif surface_measure_mode == :disconnected
            exposed_particles = findall(exposed)
            surface_measure[first(exposed_particles)] = particle_spacing^2
            surface_measure[last(exposed_particles)] = particle_spacing^2
        end
        boundary = InitialCondition(;
                                    coordinates=transform * boundary_raw.coordinates,
                                    velocity=transform * boundary_raw.velocity,
                                    mass=boundary_raw.mass, density=boundary_raw.density,
                                    pressure=boundary_raw.pressure, particle_spacing,
                                    normals=provide_normals ? transform * normals : nothing)
        boundary_model = if provide_surface_measure
            BoundaryModelDummyParticles(boundary; fluid_system,
                                        surface_measure=surface_measure)
        else
            BoundaryModelDummyParticles(boundary; fluid_system)
        end
        boundary_system = if boundary_kind == :wall
            WallBoundarySystem(boundary, boundary_model; prescribed_motion,
                               color_value=boundary_color)
        else
            RigidBodySystem(boundary; boundary_model, color_value=boundary_color)
        end
        semi = Semidiscretization(fluid_system, boundary_system)
        ode = semidiscretize(semi, (zero(ELTYPE), ELTYPE(0.01)))
        return (; fluid_system, boundary_system, semi, ode, surface_measure,
                particle_spacing)
    end

    function wetted_area_kick(setup; time=zero(eltype(setup.fluid_system)))
        v_ode, u_ode = setup.ode.u0.x
        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, setup.ode.p, time)
        fluid_dv = TrixiParticles.wrap_v(dv_ode, setup.fluid_system, setup.semi)
        return Array(fluid_dv[1:3, :]), dv_ode
    end

    @testset "wetted-area constructors and configuration" begin
        normal_method = ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1f0,
                                                interface_threshold=0.01f0,
                                                ideal_density_threshold=0.95f0,
                                                contact_model=WettedAreaContactAngle(60.0f0))
        @test normal_method.contact_model isa WettedAreaContactAngle{Float32}
        @test normal_method.contact_model.contact_angle === 60.0f0
        @test isnothing(ColorfieldSurfaceNormal().contact_model)
        @test isnothing(ColorfieldSurfaceNormal(0.1, 0.01, 0.0).contact_model)

        system_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(system_data, normal_method)
        @test system_data["surface_normal_method"]["contact_model"] ==
              "WettedAreaContactAngle"
        @test system_data["surface_normal_method"]["contact_angle"] === 60.0f0

        for angle in (-1, 181, NaN, Inf, 1im, "invalid", 0, 180)
            @test_throws ArgumentError WettedAreaContactAngle(angle)
        end
        @test_throws ArgumentError ColorfieldSurfaceNormal(contact_model=:invalid)

        setup32 = build_wetted_area_setup(; ELTYPE=Float32)
        fluid32 = setup32.fluid_system
        boundary_cache32 = setup32.boundary_system.boundary_model.cache
        @test fluid32.surface_normal_method.contact_model isa
              WettedAreaContactAngle{Float32}
        @test eltype(fluid32.cache.wetted_area_density_conjugate) == Float32
        @test eltype(boundary_cache32.surface_measure) == Float32
        @test all(>=(0), boundary_cache32.surface_measure)
        @test all(>(0),
                  boundary_cache32.wetted_area_flooded_reference[setup32.surface_measure .> 0])
        @test isfinite(fluid32.cache.wetted_area_normalized_edge_shift[])

        boundary = setup32.boundary_system.initial_condition
        second_model = BoundaryModelDummyParticles(boundary;
                                                   fluid_system=setup32.fluid_system,
                                                   surface_measure=setup32.surface_measure)
        second_boundary = WallBoundarySystem(boundary, second_model)
        multiple_semi = Semidiscretization(setup32.fluid_system,
                                           setup32.boundary_system, second_boundary)
        multiple_ode = semidiscretize(multiple_semi, (0.0f0, 0.01f0))
        multiple_dv = zero(multiple_ode.u0.x[1])
        TrixiParticles.kick!(multiple_dv, multiple_ode.u0.x...,
                             multiple_ode.p, 0.0f0)
        @test sum(abs, boundary_cache32.wetted_area_weight) > 0
        @test sum(abs, second_model.cache.wetted_area_weight) > 0
        @test setup32.fluid_system.cache.wetted_area[] > 0

        @test_throws ArgumentError build_wetted_area_setup(;
                                                           provide_surface_measure=false)
        @test_throws ArgumentError build_wetted_area_setup(; provide_normals=false)
        @test_throws ArgumentError build_wetted_area_setup(;
                                                           surface_measure_mode=:disconnected)
        @test_throws ArgumentError build_wetted_area_setup(;
                                                           surface_measure_mode=:empty)
        @test_throws ArgumentError build_wetted_area_setup(;
                                                           smoothing_length_ratio=1.5)
        @test_throws ArgumentError build_wetted_area_setup(;
                                                           smoothing_kernel=SchoenbergCubicSplineKernel{3}())
        @test_throws ArgumentError build_wetted_area_setup(;
                                                           density_calculator=SummationDensity())
        @test_throws ArgumentError build_wetted_area_setup(;
                                                           surface_tension_model=:csf)
        @test_throws ArgumentError build_wetted_area_setup(; fluid_color=2)
        @test_throws ArgumentError build_wetted_area_setup(; boundary_color=1)

        no_contact = build_wetted_area_setup(; contact=false,
                                             provide_surface_measure=false)
        @test !haskey(no_contact.boundary_system.boundary_model.cache,
                      :wetted_area_weight)
        @test !haskey(no_contact.fluid_system.cache, :wetted_area_density_conjugate)
    end

    @testset "smooth interface activity" begin
        method = ColorfieldSurfaceNormal(; boundary_contact_threshold=1,
                                         interface_threshold=0.1f0,
                                         ideal_density_threshold=0.9,
                                         interface_taper_start=0.8,
                                         support_taper_width=0.05)
        @test method isa ColorfieldSurfaceNormal{Float64}
        @test method.interface_taper_start === 0.8
        @test method.support_taper_width === 0.05
        @test !method.normal_smoothing
        @test ColorfieldSurfaceNormal(1, 1, 0) isa ColorfieldSurfaceNormal{Float64}
        @test ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1f0,
                                      interface_threshold=0.01f0,
                                      ideal_density_threshold=0.0f0,
                                      interface_taper_start=0.8f0,
                                      support_taper_width=0.025f0) isa
              ColorfieldSurfaceNormal{Float32}

        for ELTYPE in (Float32, Float64)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(-1)) === ELTYPE(0)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(0)) === ELTYPE(0)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(0.5)) === ELTYPE(0.5)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(1)) === ELTYPE(1)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(2)) === ELTYPE(1)

            method_ = ColorfieldSurfaceNormal(; boundary_contact_threshold=ELTYPE(0.1),
                                              interface_threshold=ELTYPE(0.1),
                                              ideal_density_threshold=ELTYPE(0.9),
                                              interface_taper_start=ELTYPE(0.8),
                                              support_taper_width=ELTYPE(0.05))
            @test TrixiParticles.gradient_interface_activity(ELTYPE(0.08), one(ELTYPE),
                                                             method_) === ELTYPE(0)
            @test TrixiParticles.gradient_interface_activity(ELTYPE(0.09), one(ELTYPE),
                                                             method_) ≈ ELTYPE(0.5)
            @test TrixiParticles.gradient_interface_activity(ELTYPE(0.1), one(ELTYPE),
                                                             method_) === ELTYPE(1)
            @test TrixiParticles.support_interface_activity(ELTYPE(0.9), method_) ===
                  ELTYPE(1)
            @test TrixiParticles.support_interface_activity(ELTYPE(0.925), method_) ≈
                  ELTYPE(0.5)
            @test TrixiParticles.support_interface_activity(ELTYPE(0.95), method_) ===
                  ELTYPE(0)

            step = sqrt(eps(ELTYPE))
            derivative_at_zero = TrixiParticles.cubic_smoothstep(step) / step
            derivative_at_one = (one(ELTYPE) -
                                 TrixiParticles.cubic_smoothstep(one(ELTYPE) - step)) / step
            @test abs(derivative_at_zero) < 4step
            @test abs(derivative_at_one) < 4step
        end

        disabled = ColorfieldSurfaceNormal(; ideal_density_threshold=0.0)
        @test TrixiParticles.support_interface_activity(10.0, disabled) == 1.0
        @test TrixiParticles.normalized_surface_curvature(1.0, 0.0) == 0.0
        @test TrixiParticles.normalized_surface_curvature(1.0, eps()) == 0.0
        @test TrixiParticles.normalized_surface_curvature(2.0, 0.5) == 4.0

        for threshold in (-1, NaN, Inf)
            @test_throws ArgumentError ColorfieldSurfaceNormal(interface_threshold=threshold)
            @test_throws ArgumentError ColorfieldSurfaceNormal(ideal_density_threshold=threshold)
        end
        for taper_start in (-0.1, 1.0, NaN, Inf)
            @test_throws ArgumentError ColorfieldSurfaceNormal(;
                                                               interface_taper_start=taper_start)
        end
        for taper_width in (0.0, -0.1, NaN, Inf)
            @test_throws ArgumentError ColorfieldSurfaceNormal(;
                                                               support_taper_width=taper_width)
        end
        for normal_smoothing in (0, 1, nothing)
            @test_throws ArgumentError ColorfieldSurfaceNormal(; normal_smoothing)
        end
        @test ColorfieldSurfaceNormal(; normal_smoothing=true).normal_smoothing

        system_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(system_data, method)
        @test system_data["surface_normal_method"]["interface_threshold"] ≈ 0.1
        @test system_data["surface_normal_method"]["interface_taper_start"] === 0.8
        @test system_data["surface_normal_method"]["support_taper_width"] === 0.05
        @test system_data["surface_normal_method"]["normal_smoothing"] === false
    end

    @testset "wetted-area energy and production RHS" begin
        active = build_wetted_area_setup(; angle=60.0)
        neutral = build_wetted_area_setup(; angle=90.0)
        no_contact = build_wetted_area_setup(; contact=false)
        active_acceleration, = wetted_area_kick(active)
        neutral_acceleration, = wetted_area_kick(neutral)
        no_contact_acceleration, = wetted_area_kick(no_contact)
        contact_acceleration = active_acceleration - neutral_acceleration

        @test neutral_acceleration == no_contact_acceleration
        @test neutral.fluid_system.cache.wetted_area_energy[] == 0
        @test all(iszero, neutral.fluid_system.cache.wetted_area_density_conjugate)
        @test all(iszero,
                  neutral.boundary_system.boundary_model.cache.wetted_area_weight)
        @test all(iszero,
                  neutral.boundary_system.boundary_model.cache.wetted_area_reaction)
        @test active.fluid_system.cache.wetted_area_energy[] < 0
        @test norm(contact_acceleration) > 0

        fluid_force = contact_acceleration * active.fluid_system.mass
        wall_reaction_cache = active.boundary_system.boundary_model.cache.wetted_area_reaction
        wall_reaction = vec(sum(wall_reaction_cache; dims=2))
        force_scale = sum(particle -> norm(active.fluid_system.mass[particle] *
                                           contact_acceleration[:, particle]),
                          eachparticle(active.fluid_system)) +
                      sum(particle -> norm(view(wall_reaction_cache, :, particle)),
                          eachparticle(active.boundary_system))
        @test norm(fluid_force + wall_reaction) / force_scale < 1.0e-12

        active_edac = build_wetted_area_setup(; solver=:edac, angle=60.0)
        neutral_edac = build_wetted_area_setup(; solver=:edac, angle=90.0)
        active_edac_acceleration, = wetted_area_kick(active_edac)
        neutral_edac_acceleration, = wetted_area_kick(neutral_edac)
        edac_contact_acceleration = active_edac_acceleration -
                                    neutral_edac_acceleration
        edac_force = edac_contact_acceleration * active_edac.fluid_system.mass
        edac_reaction = vec(sum(active_edac.boundary_system.boundary_model.cache.wetted_area_reaction;
                                dims=2))
        @test norm(edac_contact_acceleration) > 0
        @test norm(edac_force + edac_reaction) <
              1.0e-12 * (norm(edac_force) + norm(edac_reaction))

        active_rigid = build_wetted_area_setup(; boundary_kind=:rigid, angle=60.0)
        neutral_rigid = build_wetted_area_setup(; boundary_kind=:rigid, angle=90.0)
        wetted_area_kick(active_rigid)
        wetted_area_kick(neutral_rigid)
        rigid_reaction = active_rigid.boundary_system.boundary_model.cache.wetted_area_reaction
        rigid_contact_force = active_rigid.boundary_system.force_per_particle -
                              neutral_rigid.boundary_system.force_per_particle
        @test rigid_contact_force ≈ rigid_reaction rtol = 2eps()
        @test active_rigid.boundary_system.resultant_force[] -
              neutral_rigid.boundary_system.resultant_force[] ≈
              vec(sum(rigid_reaction; dims=2)) rtol = 2eps()
        expected_torque = zero(active_rigid.boundary_system.resultant_torque[])
        for particle in eachparticle(active_rigid.boundary_system)
            relative_position = TrixiParticles.extract_svector(active_rigid.boundary_system.relative_coordinates,
                                                               active_rigid.boundary_system,
                                                               particle)
            reaction = TrixiParticles.extract_svector(rigid_reaction,
                                                      active_rigid.boundary_system,
                                                      particle)
            expected_torque += cross(relative_position, reaction)
        end
        @test active_rigid.boundary_system.resultant_torque[] -
              neutral_rigid.boundary_system.resultant_torque[] ≈ expected_torque atol = 1.0e-12

        rotation = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]
        rotated_active = build_wetted_area_setup(; angle=60.0, rotation)
        rotated_neutral = build_wetted_area_setup(; angle=90.0, rotation)
        rotated_active_acceleration, = wetted_area_kick(rotated_active)
        rotated_neutral_acceleration, = wetted_area_kick(rotated_neutral)
        rotated_contact_acceleration = rotated_active_acceleration -
                                       rotated_neutral_acceleration
        @test rotated_contact_acceleration≈rotation*contact_acceleration rtol=2.0e-12 atol=2.0e-12

        moving_motion() = PrescribedMotion((position,
                                            time) -> begin
                                               cosine = cos(time)
                                               sine = sin(time)
                                               SVector(cosine * position[1] +
                                                       sine * position[3], position[2],
                                                       -sine * position[1] +
                                                       cosine * position[3])
                                           end,
                                           time -> true)
        moving_active = build_wetted_area_setup(; angle=60.0,
                                                prescribed_motion=moving_motion())
        moving_neutral = build_wetted_area_setup(; angle=90.0,
                                                 prescribed_motion=moving_motion())
        moving_active_acceleration, = wetted_area_kick(moving_active; time=0.02)
        moving_neutral_acceleration, = wetted_area_kick(moving_neutral; time=0.02)
        moving_contact_acceleration = moving_active_acceleration -
                                      moving_neutral_acceleration
        moving_force = moving_contact_acceleration * moving_active.fluid_system.mass
        moving_reaction = vec(sum(moving_active.boundary_system.boundary_model.cache.wetted_area_reaction;
                                  dims=2))
        @test norm(moving_contact_acceleration) > 0
        @test norm(moving_force + moving_reaction) <
              1.0e-12 * (norm(moving_force) + norm(moving_reaction))
    end

    @testset "wetted-area variational derivative" begin
        active = build_wetted_area_setup(; angle=60.0)
        neutral = build_wetted_area_setup(; angle=90.0)
        active_acceleration, = wetted_area_kick(active)
        neutral_acceleration, = wetted_area_kick(neutral)
        contact_acceleration = active_acceleration - neutral_acceleration

        v_ode, u_ode = active.ode.u0.x
        v = TrixiParticles.wrap_v(v_ode, active.fluid_system, active.semi)
        u = TrixiParticles.wrap_u(u_ode, active.fluid_system, active.semi)
        u_boundary = TrixiParticles.wrap_u(u_ode, active.boundary_system, active.semi)
        coordinates = Array(TrixiParticles.current_coordinates(u, active.fluid_system))
        boundary_coordinates = Array(TrixiParticles.current_coordinates(u_boundary,
                                                                        active.boundary_system))
        density = collect(TrixiParticles.current_density(v, active.fluid_system))
        displacement = similar(coordinates)
        displacement_scale = max(maximum(abs, coordinates), active.particle_spacing)
        for particle in eachparticle(active.fluid_system)
            displacement[1, particle] = -coordinates[1, particle] / displacement_scale
            displacement[2, particle] = -coordinates[2, particle] / displacement_scale
            displacement[3, particle] = 2coordinates[3, particle] / displacement_scale
        end
        density_rate = zeros(eltype(active.fluid_system), nparticles(active.fluid_system))
        TrixiParticles.foreach_point_neighbor(active.fluid_system, active.fluid_system,
                                              coordinates, coordinates, active.semi;
                                              points=eachparticle(active.fluid_system),
                                              parallelization_backend=SerialBackend()) do particle,
                                                                                          neighbor,
                                                                                          pos_diff,
                                                                                          distance
            gradient = TrixiParticles.smoothing_kernel_grad(active.fluid_system,
                                                            pos_diff, distance, particle)
            mass_b = TrixiParticles.hydrodynamic_mass(active.fluid_system, neighbor)
            density_rate[particle] += density[particle] / density[neighbor] * mass_b *
                                      dot(displacement[:, particle] -
                                          displacement[:, neighbor], gradient)
        end
        fluid_boundary_pairs = Tuple{Int, Int}[]
        TrixiParticles.foreach_point_neighbor(active.fluid_system,
                                              active.boundary_system,
                                              coordinates, boundary_coordinates,
                                              active.semi;
                                              points=eachparticle(active.fluid_system),
                                              parallelization_backend=SerialBackend()) do particle,
                                                                                          neighbor,
                                                                                          pos_diff,
                                                                                          distance
            push!(fluid_boundary_pairs, (particle, neighbor))
        end

        function perturbed_wetted_area_energy(epsilon)
            boundary_cache = active.boundary_system.boundary_model.cache
            colorfield = copy(boundary_cache.initial_colorfield)
            for (particle, neighbor) in fluid_boundary_pairs
                distance2 = zero(eltype(active.fluid_system))
                for dim in 1:3
                    difference = coordinates[dim, particle] +
                                 epsilon * displacement[dim, particle] -
                                 boundary_coordinates[dim, neighbor]
                    distance2 += difference^2
                end
                perturbed_density = density[particle] + epsilon * density_rate[particle]
                colorfield[neighbor] += active.fluid_system.mass[particle] /
                                        perturbed_density *
                                        TrixiParticles.smoothing_kernel(active.fluid_system,
                                                                        sqrt(distance2),
                                                                        particle)
            end
            raw_area = zero(eltype(active.fluid_system))
            for particle in eachparticle(active.boundary_system)
                measure = boundary_cache.surface_measure[particle]
                iszero(measure) && continue
                reference = boundary_cache.wetted_area_flooded_reference[particle]
                fraction = clamp(colorfield[particle] / reference, 0, 1)
                raw_area += measure * TrixiParticles.cubic_smoothstep(fraction)
            end
            raw_radius = sqrt(raw_area / pi)
            edge_shift = active.fluid_system.cache.wetted_area_normalized_edge_shift[] *
                         TrixiParticles.initial_smoothing_length(active.fluid_system)
            corrected_radius = max(raw_radius - edge_shift, zero(raw_radius))
            coefficient = TrixiParticles.wetted_area_coefficient(active.fluid_system.surface_tension,
                                                                 active.fluid_system.surface_normal_method.contact_model)
            return -coefficient * pi * corrected_radius^2
        end

        epsilon = 1.0e-5active.particle_spacing
        finite_difference = (perturbed_wetted_area_energy(epsilon) -
                             perturbed_wetted_area_energy(-epsilon)) / (2epsilon)
        analytic_derivative = zero(finite_difference)
        for particle in eachparticle(active.fluid_system)
            analytic_derivative -= active.fluid_system.mass[particle] *
                                   dot(contact_acceleration[:, particle],
                                       displacement[:, particle])
        end
        derivative_scale = max(abs(finite_difference), abs(analytic_derivative))
        @test abs(finite_difference - analytic_derivative) / derivative_scale < 1.0e-5
        @test perturbed_wetted_area_energy(0.0) ≈
              active.fluid_system.cache.wetted_area_energy[] rtol = 5eps()
    end

    @testset verbose=true "`cohesion_force_akinci`" begin
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.0
        m_b = 1.0
        pos_diff = [1.0, 1.0]

        # These values can be extracted from the graphs in the paper by Akinci et al. or by manual calculation.
        # Additional digits have been accepted from the actual calculation.
        test_distance = 0.1
        val = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance) * test_distance
        @test isapprox(val[1], 0.1443038770421044, atol=6e-15)
        @test isapprox(val[2], 0.1443038770421044, atol=6e-15)

        # Maximum repulsion force
        test_distance = 0.01
        max = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance) * test_distance
        @test isapprox(max[1], 0.15913517632298307, atol=6e-15)
        @test isapprox(max[2], 0.15913517632298307, atol=6e-15)

        # Near 0
        test_distance = 0.2725
        zero = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance) * test_distance
        @test isapprox(zero[1], 0.0004360543645195717, atol=6e-15)
        @test isapprox(zero[2], 0.0004360543645195717, atol=6e-15)

        # Maximum attraction force
        test_distance = 0.5
        maxa = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance) * test_distance
        @test isapprox(maxa[1], -0.15915494309189535, atol=6e-15)
        @test isapprox(maxa[2], -0.15915494309189535, atol=6e-15)

        # Should be 0
        test_distance = 1.0
        zero = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance) * test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)
    end

    @testset verbose=true "adhesion_force_akinci" begin
        surface_tension = TrixiParticles.SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.0
        m_b = 1.0
        pos_diff = [1.0, 1.0]

        # These values can be extracted from the graphs in the paper by Akinci et al. or by manual calculation.
        # Additional digits have been accepted from the actual calculation.
        test_distance = 0.1
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        test_distance = 0.5
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        # Near 0
        test_distance = 0.51
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0) *
               test_distance
        @test isapprox(zero[1], -0.002619160170741761, atol=6e-15)
        @test isapprox(zero[2], -0.002619160170741761, atol=6e-15)

        # Maximum adhesion force
        test_distance = 0.75
        max = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance, 1.0) *
              test_distance
        @test isapprox(max[1], -0.004949747468305833, atol=6e-15)
        @test isapprox(max[2], -0.004949747468305833, atol=6e-15)

        # Should be 0
        test_distance = 1.0
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)
    end

    @testset "Morris CSF local force" begin
        function build_morris_system(solver, particle_count; normal_smoothing=false)
            coordinates = zeros(2, particle_count)
            coordinates[1, :] .= range(0.0; step=0.25, length=particle_count)
            initial_condition = InitialCondition(; coordinates,
                                                 velocity=zeros(2, particle_count),
                                                 mass=ones(particle_count),
                                                 density=ones(particle_count),
                                                 particle_spacing=0.25)
            smoothing_kernel = WendlandC2Kernel{2}()
            surface_tension = SurfaceTensionMorris(; surface_tension_coefficient=0.7)
            normal_method = ColorfieldSurfaceNormal(; interface_threshold=0.1,
                                                    normal_smoothing)
            if solver == :wcsph
                return WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                   smoothing_length=0.5,
                                                   density_calculator=ContinuityDensity(),
                                                   state_equation=StateEquationCole(;
                                                                                    sound_speed=10.0,
                                                                                    reference_density=1.0,
                                                                                    exponent=1),
                                                   surface_tension,
                                                   surface_normal_method=normal_method,
                                                   reference_particle_spacing=0.25)
            end
            return EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                               smoothing_length=0.5, sound_speed=10.0,
                                               density_calculator=ContinuityDensity(),
                                               surface_tension,
                                               surface_normal_method=normal_method,
                                               reference_particle_spacing=0.25)
        end

        function morris_rhs_effect(system)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
            system.cache.surface_normal[1, :] .= 1.0
            system.cache.surface_normal[2, :] .= 0.0
            system.cache.curvature .= 3.0
            system.cache.delta_s .= 2.0
            system.cache.interface_activity .= 1.0

            return GC.@preserve v_ode u_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                u = TrixiParticles.wrap_u(u_ode, system, semi)
                rho_a = TrixiParticles.current_density(v, system, 1)
                expected = TrixiParticles.surface_tension_acceleration(system.surface_tension,
                                                                       system, 1, rho_a,
                                                                       SVector(0.0, 0.0))
                with_surface_tension = zeros(eltype(v), size(v))
                TrixiParticles.interact!(with_surface_tension, v, u, v, u,
                                         system, system, semi)
                system.cache.delta_s .= 0
                without_surface_tension = zeros(eltype(v), size(v))
                TrixiParticles.interact!(without_surface_tension, v, u, v, u,
                                         system, system, semi)
                return (with_surface_tension - without_surface_tension)[1:2, :],
                       expected
            end
        end

        effects = []
        for solver in (:wcsph, :edac), particle_count in (2, 4)
            effect,
            expected = morris_rhs_effect(build_morris_system(solver, particle_count))
            @test all(particle -> effect[:, particle] ≈ expected, axes(effect, 2))
            push!(effects, effect[:, 1])
        end
        @test all(effect -> effect ≈ first(effects), effects)

        system = build_morris_system(:wcsph, 2)
        system.cache.surface_normal .= [2.0 1.0; 0.0 1.0]
        system.cache.support_moment .= 0
        TrixiParticles.remove_invalid_normals!(system, system.surface_tension,
                                               system.surface_normal_method)
        @test system.cache.delta_s ≈ [4.0, 2sqrt(2)]
        @test system.cache.interface_activity == [1.0, 1.0]
        @test system.cache.surface_normal[:, 1] ≈ [1.0, 0.0]
        @test system.cache.surface_normal[:, 2] ≈ [1 / sqrt(2), 1 / sqrt(2)]
        system.cache.surface_normal[:, 1] .= [NaN, 0.0]
        TrixiParticles.remove_invalid_normals!(system, system.surface_tension,
                                               system.surface_normal_method)
        @test iszero(system.cache.surface_normal[:, 1])
        @test iszero(system.cache.delta_s[1])
        @test iszero(system.cache.interface_activity[1])

        system.cache.surface_normal[1, :] .= 1.0
        system.cache.surface_normal[2, :] .= 0.0
        system.cache.curvature .= 3.0
        system.cache.delta_s .= 2.0
        acceleration = TrixiParticles.surface_tension_acceleration(system.surface_tension,
                                                                   system, 1, 1.0,
                                                                   SVector(0.0, 0.0))
        @test acceleration ≈ SVector(-4.2, 0.0)

        smoothed_system = build_morris_system(:wcsph, 2; normal_smoothing=true)
        smoothed_system.cache.surface_normal .= [1.0 1.0; 0.0 0.0]
        smoothed_system.cache.smoothed_surface_normal .= [0.0 0.0; 1.0 1.0]
        smoothed_system.cache.curvature .= 3.0
        smoothed_system.cache.delta_s .= 2.0
        smoothed_acceleration = TrixiParticles.surface_tension_acceleration(smoothed_system.surface_tension,
                                                                            smoothed_system,
                                                                            1, 1.0,
                                                                            SVector(0.0,
                                                                                    0.0))
        @test smoothed_acceleration ≈ SVector(0.0, -4.2)
        @test TrixiParticles.surface_normal(smoothed_system, 1) == SVector(1.0, 0.0)
        system.cache.curvature[1] /= 2
        system.cache.delta_s[1] /= 2
        scaled_acceleration = TrixiParticles.surface_tension_acceleration(system.surface_tension,
                                                                          system, 1, 1.0,
                                                                          SVector(0.0,
                                                                                  0.0))
        @test scaled_acceleration ≈ acceleration / 4

        semi = Semidiscretization(system)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
        system.cache.surface_normal[1, :] .= 1.0
        system.cache.surface_normal[2, :] .= 0.0
        system.cache.curvature .= 3.0
        system.cache.delta_s .= 2.0
        system.cache.interface_activity .= 1.0
        vtk = Dict{String, Any}()
        expected_vtk_acceleration = GC.@preserve v_ode u_ode begin
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            rho_a = TrixiParticles.current_density(v, system, 1)
            velocity = TrixiParticles.current_velocity(v, system, 1)
            expected = TrixiParticles.surface_tension_acceleration(system.surface_tension,
                                                                   system, 1, rho_a,
                                                                   velocity)
            TrixiParticles.write2vtk!(vtk, v, u, 0.0, system)
            expected
        end
        @test vtk["surface_tension"][:, 1] ≈ expected_vtk_acceleration
        @test vtk["surface_delta"] == system.cache.delta_s
        @test vtk["interface_activity"] == system.cache.interface_activity
        @test vtk["surface_support_moment"] == system.cache.support_moment
        @test vtk["surface_tension_normal"][1] == SVector(1.0, 0.0)
        system.cache.surface_normal .= [1.0 0.0; 0.0 1.0]

        function curvature_with_neighbor_activity(activity)
            system.cache.interface_activity .= [1.0, activity]
            fill!(system.cache.curvature, 0)
            fill!(system.cache.correction_factor, 0)
            GC.@preserve v_ode u_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                u = TrixiParticles.wrap_u(u_ode, system, semi)
                TrixiParticles.calc_curvature!(system, system, u, v, v, u, semi,
                                               system.surface_normal_method,
                                               system.surface_normal_method)
            end
            denominator = system.cache.correction_factor[1]
            return TrixiParticles.normalized_surface_curvature(system.cache.curvature[1],
                                                               denominator)
        end

        curvature_zero = curvature_with_neighbor_activity(0.0)
        curvature_small = curvature_with_neighbor_activity(1.0e-6)
        curvature_full = curvature_with_neighbor_activity(1.0)
        @test iszero(curvature_zero)
        @test abs(curvature_small) < 1.0e-4 * abs(curvature_full)
        @test isfinite(curvature_full)

        curvature_numerator = copy(system.cache.curvature)
        correction_factor = copy(system.cache.correction_factor)
        GC.@preserve v_ode u_ode begin
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            TrixiParticles.calc_curvature!(system, system, u, v, v, u, semi,
                                           system.surface_normal_method,
                                           system.surface_normal_method)
        end
        @test system.cache.curvature ≈ 2curvature_numerator
        @test system.cache.correction_factor ≈ 2correction_factor
    end

    @testset "balanced continuum surface stress" begin
        initial_condition = InitialCondition(; coordinates=[0.0 0.75; 0.0 0.0],
                                             velocity=zeros(2, 2), mass=[2.0, 3.0],
                                             density=ones(2), particle_spacing=0.5)
        surface_tension = SurfaceTensionMomentumMorris(;
                                                       surface_tension_coefficient=0.7)
        normal_method = ColorfieldSurfaceNormal(; interface_threshold=0.1)
        system = WeaklyCompressibleSPHSystem(initial_condition;
                                             smoothing_kernel=WendlandC2Kernel{2}(),
                                             smoothing_length=0.5,
                                             density_calculator=SummationDensity(),
                                             state_equation=StateEquationCole(;
                                                                              sound_speed=10.0,
                                                                              reference_density=1.0,
                                                                              exponent=1),
                                             surface_tension,
                                             surface_normal_method=normal_method,
                                             reference_particle_spacing=0.5)

        @test haskey(system.cache, :delta_s)
        @test haskey(system.cache, :interface_activity)
        @test haskey(system.cache, :divergence_correction)
        @test haskey(system.cache, :surface_normal)
        @test !haskey(system.cache, :stress_tensor)

        # Capture the one-phase surface delta before normalizing the color gradient.
        system.cache.surface_normal .= [2.0 1.0; 0.0 1.0]
        system.cache.divergence_correction .= 0
        TrixiParticles.remove_invalid_normals!(system, surface_tension, normal_method)
        @test system.cache.delta_s ≈ [4.0, 2sqrt(2)]
        @test system.cache.interface_activity == [1.0, 1.0]
        @test system.cache.surface_normal[:, 1] ≈ [1.0, 0.0]
        @test system.cache.surface_normal[:, 2] ≈ [1 / sqrt(2), 1 / sqrt(2)]

        grad_kernel = SVector(0.3, -0.4)
        stress_gradient_1 = 4.0 .* (grad_kernel - SVector(1.0, 0.0) * 0.3)
        normal_2 = SVector(1 / sqrt(2), 1 / sqrt(2))
        stress_gradient_2 = 2sqrt(2) .* (grad_kernel -
                             normal_2 * dot(normal_2, grad_kernel))
        @test TrixiParticles.surface_stress_times_gradient(system, 1, grad_kernel) ≈
              stress_gradient_1
        @test TrixiParticles.surface_stress_times_gradient(system, 2, grad_kernel) ≈
              stress_gradient_2

        rho_a = 2.0
        rho_b = 3.0
        system.cache.divergence_correction .= [0.5, 1.0]
        divergence_correction = 2 / (0.5 + 1.0)
        pos_diff = SVector(-0.75, 0.0)
        distance = norm(pos_diff)
        dv_a = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(dv_a, surface_tension, surface_tension,
                                              system, system, 1, 2, pos_diff, distance,
                                              rho_a, rho_b, grad_kernel, 4.0)
        expected = 3divergence_correction * surface_tension.surface_tension_coefficient /
                   (rho_a * rho_b) * (stress_gradient_1 + stress_gradient_2)
        @test dv_a[] ≈ expected

        # The symmetric stress divergence conserves pairwise momentum and deliberately
        # ignores the Akinci-specific correction factor passed above.
        dv_b = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(dv_b, surface_tension, surface_tension,
                                              system, system, 2, 1, -pos_diff, distance,
                                              rho_b, rho_a, -grad_kernel, 4.0)
        @test 2dv_a[] ≈ -3dv_b[]

        semi = Semidiscretization(system)
        ode = semidiscretize(semi, (0.0, 0.01))
        v_ode, u_ode = ode.u0.x
        vtk = Dict{String, Any}()
        GC.@preserve v_ode u_ode begin
            v = TrixiParticles.wrap_v(v_ode, system, semi)
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            TrixiParticles.write2vtk!(vtk, v, u, 0.0, system)
        end
        @test vtk["surface_delta"] == system.cache.delta_s
        @test vtk["interface_activity"] == system.cache.interface_activity
        @test vtk["surface_tension_normal"] == [TrixiParticles.surface_normal(system, 1),
            TrixiParticles.surface_normal(system, 2)]
        @test vtk["surface_divergence_correction"] == [0.5, 1.0]
        @test size(vtk["surface_stress_tensor"]) == (2, 2, 2)
        @test vtk["surface_stress_tensor"][:, :, 1] ≈ [0.0 0.0; 0.0 4.0]
        @test all(isfinite, vtk["surface_stress_tensor"])

        system.cache.divergence_correction .= 0
        unsupported_force = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(unsupported_force, surface_tension,
                                              surface_tension, system, system, 1, 2,
                                              pos_diff, distance, rho_a, rho_b,
                                              grad_kernel, 1.0)
        @test iszero(unsupported_force[])

        filtered_method = ColorfieldSurfaceNormal(; interface_threshold=0.1,
                                                  ideal_density_threshold=0.9,
                                                  support_taper_width=0.05)
        system.cache.surface_normal .= 0
        system.cache.surface_normal[1, 1] = 0.2
        system.cache.divergence_correction .= [0.925, 1.0]
        TrixiParticles.remove_invalid_normals!(system, surface_tension, filtered_method)
        @test system.cache.interface_activity[1] ≈ 0.5
        @test system.cache.delta_s[1] ≈ 0.2
        @test system.cache.surface_normal[:, 1] == [1.0, 0.0]
    end

    @testset "CSS static Laplace balance" begin
        reference_density = 1000.0
        target_particles = 375
        drop_volume = 1.0e-6
        particle_spacing = cbrt(drop_volume / target_particles)
        radius = cbrt(3drop_volume / (4pi))
        initial_condition = SphereShape(particle_spacing, radius + particle_spacing / 2,
                                        (0.0, 0.0, 0.0), reference_density;
                                        sphere_type=VoxelSphere())
        smoothing_kernel = WendlandC2Kernel{3}()
        smoothing_length = 1.4particle_spacing

        function initial_acceleration(system)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
            return GC.@preserve v_ode u_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                u = TrixiParticles.wrap_u(u_ode, system, semi)
                dv = zeros(eltype(v), size(v))
                TrixiParticles.interact!(dv, v, u, v, u, system, system, semi)
                Array(dv[1:3, :])
            end
        end

        coefficient = 1.0
        css = SurfaceTensionMomentumMorris(; surface_tension_coefficient=coefficient)
        css_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                 smoothing_length,
                                                 density_calculator=ContinuityDensity(),
                                                 state_equation=StateEquationCole(;
                                                                                  sound_speed=100.0,
                                                                                  reference_density,
                                                                                  exponent=1),
                                                 surface_tension=css,
                                                 surface_normal_method=ColorfieldSurfaceNormal(;
                                                                                               boundary_contact_threshold=Inf,
                                                                                               interface_threshold=0.01,
                                                                                               ideal_density_threshold=0.95),
                                                 reference_particle_spacing=particle_spacing)
        css_acceleration = initial_acceleration(css_system)

        pressure_basis = 1.0
        sound_speed = 100.0
        pressure_reference_density = reference_density - pressure_basis / sound_speed^2
        pressure_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                      smoothing_length,
                                                      density_calculator=ContinuityDensity(),
                                                      state_equation=StateEquationCole(;
                                                                                       sound_speed,
                                                                                       reference_density=pressure_reference_density,
                                                                                       exponent=1))
        pressure_acceleration = initial_acceleration(pressure_system) / pressure_basis

        interface = findall(>(0), css_system.cache.delta_s)
        capillary = vec(css_acceleration[:, interface])
        unit_pressure = vec(pressure_acceleration[:, interface])
        pressure_jump = -dot(capillary, unit_pressure) / dot(unit_pressure, unit_pressure)
        volume = sum(css_system.mass) / reference_density
        equivalent_radius = cbrt(3volume / (4pi))
        inferred_surface_tension = pressure_jump * equivalent_radius / 2
        total_force = vec(sum(css_acceleration .* reshape(css_system.mass, 1, :);
                              dims=2))

        @test inferred_surface_tension ≈ coefficient rtol = 0.05
        @test norm(total_force) < 1.0e-12
        @test all(isfinite, css_system.cache.divergence_correction)
        @test minimum(css_system.cache.divergence_correction) > 0
    end
end
