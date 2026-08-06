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

    @testset "constructors and capabilities" begin
        constructors = (CohesionForceAkinci, SurfaceTensionAkinci,
                        SurfaceTensionMorris, SurfaceTensionMomentumMorris)

        for constructor in constructors
            model = constructor(surface_tension_coefficient=0.5f0)
            @test model.surface_tension_coefficient === 0.5f0
            @test iszero(constructor(surface_tension_coefficient=0).surface_tension_coefficient)

            for coefficient in (-1.0, NaN, Inf, -Inf, 1.0im, "invalid")
                @test_throws ArgumentError constructor(surface_tension_coefficient=coefficient)
            end
        end

        physical = SurfaceTensionAkinciCohesionPhysical(;
                                                        surface_tension_coefficient=0.072f0,
                                                        reference_density=1000.0f0)
        @test physical.surface_tension_coefficient === 0.072f0
        @test physical.reference_density === 1000.0f0
        system_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(system_data, physical)
        @test system_data["surface_tension"]["model"] ==
              "SurfaceTensionAkinciCohesionPhysical"
        @test system_data["surface_tension"]["surface_tension_coefficient"] === 0.072f0
        @test system_data["surface_tension"]["reference_density"] === 1000.0f0
        @test SurfaceTensionAkinciCohesionPhysical(;
                                                   surface_tension_coefficient=0,
                                                   reference_density=1).surface_tension_coefficient ==
              0
        for coefficient in (-1.0, NaN, Inf, -Inf, 1.0im, "invalid")
            @test_throws ArgumentError SurfaceTensionAkinciCohesionPhysical(;
                                                                            surface_tension_coefficient=coefficient,
                                                                            reference_density=1000.0)
        end
        for density in (0.0, -1.0, NaN, Inf, -Inf, 1.0im, "invalid")
            @test_throws ArgumentError SurfaceTensionAkinciCohesionPhysical(;
                                                                            surface_tension_coefficient=0.072,
                                                                            reference_density=density)
        end

        @test !TrixiParticles.requires_surface_normal(nothing)
        @test !TrixiParticles.requires_surface_normal(CohesionForceAkinci())
        @test !TrixiParticles.requires_surface_normal(physical)
        @test TrixiParticles.requires_surface_normal(SurfaceTensionAkinci())
        @test TrixiParticles.requires_surface_normal(SurfaceTensionMorris())
        @test TrixiParticles.requires_surface_normal(SurfaceTensionMomentumMorris())
        @test isinf(TrixiParticles.default_surface_normal_method(SurfaceTensionAkinci(),
                                                                 nothing).boundary_contact_threshold)
        @test TrixiParticles.default_surface_normal_method(SurfaceTensionMorris(),
                                                           nothing).boundary_contact_threshold ==
              0.1

        initial_condition_1d = InitialCondition(; coordinates=reshape([0.0, 1.0], 1, 2),
                                                density=ones(2), particle_spacing=1.0)
        smoothing_kernel_1d = WendlandC2Kernel{1}()
        state_equation_1d = StateEquationCole(sound_speed=10.0, reference_density=1.0,
                                              exponent=1)
        for surface_tension in (CohesionForceAkinci(), SurfaceTensionAkinci())
            @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition_1d;
                                                                   smoothing_kernel=smoothing_kernel_1d,
                                                                   smoothing_length=1.0,
                                                                   density_calculator=SummationDensity(),
                                                                   state_equation=state_equation_1d,
                                                                   surface_tension,
                                                                   reference_particle_spacing=1.0)
            @test_throws ArgumentError EntropicallyDampedSPHSystem(initial_condition_1d;
                                                                   smoothing_kernel=smoothing_kernel_1d,
                                                                   smoothing_length=1.0,
                                                                   sound_speed=10.0,
                                                                   surface_tension,
                                                                   reference_particle_spacing=1.0)
        end

        initial_condition_2d = InitialCondition(; coordinates=[0.0 1.0; 0.0 0.0],
                                                density=ones(2), particle_spacing=1.0)
        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition_2d;
                                                               smoothing_kernel=WendlandC2Kernel{2}(),
                                                               smoothing_length=1.0,
                                                               density_calculator=SummationDensity(),
                                                               state_equation=state_equation_1d,
                                                               surface_tension=physical)

        normal_method = ColorfieldSurfaceNormal(boundary_contact_threshold=1,
                                                interface_threshold=0.1f0,
                                                ideal_density_threshold=0.25)
        @test normal_method isa ColorfieldSurfaceNormal{Float64}
        @test normal_method.interface_taper_start === 0.8
        @test normal_method.support_taper_width === 0.025
        @test ColorfieldSurfaceNormal(boundary_contact_threshold=0.1f0,
                                      interface_threshold=0.01f0,
                                      ideal_density_threshold=0.0f0) isa
              ColorfieldSurfaceNormal{Float32}
        wetted_area = ColorfieldSurfaceNormal(boundary_contact_threshold=0.1f0,
                                              interface_threshold=0.01f0,
                                              ideal_density_threshold=0.0f0,
                                              contact_model=WettedAreaContactAngle(60.0f0))
        @test wetted_area.contact_model.contact_angle === 60.0f0
        @test isnothing(ColorfieldSurfaceNormal().contact_model)
        @test isnothing(ColorfieldSurfaceNormal(0.1, 0.01, 0.0).contact_model)
        @test ColorfieldSurfaceNormal(1, 1, 0) isa ColorfieldSurfaceNormal{Float64}
        normal_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(normal_data, normal_method)
        @test normal_data["surface_normal_method"]["interface_threshold"] ≈ 0.1
        @test normal_data["surface_normal_method"]["interface_taper_start"] === 0.8
        @test normal_data["surface_normal_method"]["support_taper_width"] === 0.025
        @test isnothing(normal_data["surface_normal_method"]["contact_model"])
        @test isnothing(normal_data["surface_normal_method"]["contact_angle"])
        akinci_normal_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(akinci_normal_data,
                                        TrixiParticles.default_surface_normal_method(SurfaceTensionAkinci(),
                                                                                     nothing))
        @test akinci_normal_data["surface_normal_method"]["boundary_contact_threshold"] ==
              "Inf"
        @test_nowarn JSON.json(akinci_normal_data)
        for (method, model_name, angle) in
            ((wetted_area, "WettedAreaContactAngle", 60.0f0),)
            data = Dict{String, Any}()
            TrixiParticles.add_system_data!(data, method)
            @test data["surface_normal_method"]["contact_model"] == model_name
            @test data["surface_normal_method"]["contact_angle"] === angle
        end
        for angle in (-1, 181, NaN, Inf, 1im, "invalid")
            @test_throws ArgumentError WettedAreaContactAngle(angle)
        end
        @test_throws ArgumentError WettedAreaContactAngle(0)
        @test_throws ArgumentError WettedAreaContactAngle(180)
        @test_throws ArgumentError ColorfieldSurfaceNormal(contact_model=:invalid)
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
    end

    @testset "wetted-area configuration and quadrature" begin
        setup32 = build_wetted_area_setup(; ELTYPE=Float32)
        fluid32 = setup32.fluid_system
        boundary_cache32 = setup32.boundary_system.boundary_model.cache
        @test fluid32.surface_normal_method.contact_model isa
              WettedAreaContactAngle{Float32}
        @test eltype(fluid32.cache.wetted_area_density_conjugate) == Float32
        @test eltype(boundary_cache32.wetted_area_surface_measure) == Float32
        @test all(>=(0), boundary_cache32.wetted_area_surface_measure)
        @test all(>(0),
                  boundary_cache32.wetted_area_flooded_reference[setup32.surface_measure .> 0])
        @test isfinite(fluid32.cache.wetted_area_normalized_edge_shift[])

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

        boundary = setup32.boundary_system.initial_condition
        fluid_system = setup32.fluid_system
        @test_throws ArgumentError BoundaryModelDummyParticles(boundary; fluid_system,
                                                               surface_measure=1.0f0)
        @test_throws ArgumentError BoundaryModelDummyParticles(boundary; fluid_system,
                                                               surface_measure=zeros(Float32,
                                                                                     nparticles(boundary) -
                                                                                     1))
        invalid_measure = copy(setup32.surface_measure)
        invalid_measure[1] = -1
        @test_throws ArgumentError BoundaryModelDummyParticles(boundary; fluid_system,
                                                               surface_measure=invalid_measure)
        invalid_measure[1] = NaN
        @test_throws ArgumentError BoundaryModelDummyParticles(boundary; fluid_system,
                                                               surface_measure=invalid_measure)
        no_quadrature = build_wetted_area_setup(; contact=false,
                                                provide_surface_measure=false)
        @test !haskey(no_quadrature.boundary_system.boundary_model.cache,
                      :wetted_area_surface_measure)

        particle_spacing = 0.1
        fluid_2d = RectangularShape(particle_spacing, (3, 3), (0.0, 0.0);
                                    density=1000.0)
        fluid_system_2d = WeaklyCompressibleSPHSystem(fluid_2d;
                                                      smoothing_kernel=WendlandC2Kernel{2}(),
                                                      smoothing_length=1.4particle_spacing,
                                                      density_calculator=ContinuityDensity(),
                                                      state_equation=StateEquationCole(;
                                                                                       sound_speed=10.0,
                                                                                       reference_density=1000.0,
                                                                                       exponent=1),
                                                      surface_tension=SurfaceTensionMomentumMorris(;
                                                                                                   surface_tension_coefficient=0.072),
                                                      surface_normal_method=ColorfieldSurfaceNormal(;
                                                                                                    contact_model=WettedAreaContactAngle(60.0)),
                                                      reference_particle_spacing=particle_spacing)
        @test_throws ArgumentError Semidiscretization(fluid_system_2d)

        setup = build_wetted_area_setup()
        boundary = setup.boundary_system.initial_condition
        second_model = BoundaryModelDummyParticles(boundary;
                                                   fluid_system=setup.fluid_system,
                                                   surface_measure=setup.surface_measure)
        second_boundary = WallBoundarySystem(boundary, second_model)
        multiple_semi = Semidiscretization(setup.fluid_system, setup.boundary_system,
                                           second_boundary)
        multiple_ode = semidiscretize(multiple_semi, (0.0, 0.01))
        multiple_dv = zero(multiple_ode.u0.x[1])
        TrixiParticles.kick!(multiple_dv, multiple_ode.u0.x...,
                             multiple_ode.p, 0.0)
        @test sum(abs, setup.boundary_system.boundary_model.cache.wetted_area_weight) > 0
        @test sum(abs, second_model.cache.wetted_area_weight) > 0
        @test setup.fluid_system.cache.wetted_area[] > 0
    end

    @testset "smooth interface activity" begin
        for ELTYPE in (Float32, Float64)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(-1)) === ELTYPE(0)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(0)) === ELTYPE(0)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(0.5)) === ELTYPE(0.5)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(1)) === ELTYPE(1)
            @test TrixiParticles.cubic_smoothstep(ELTYPE(2)) === ELTYPE(1)

            method = ColorfieldSurfaceNormal(; boundary_contact_threshold=ELTYPE(0.1),
                                             interface_threshold=ELTYPE(0.1),
                                             ideal_density_threshold=ELTYPE(0.9),
                                             interface_taper_start=ELTYPE(0.8),
                                             support_taper_width=ELTYPE(0.05))
            @test TrixiParticles.gradient_interface_activity(ELTYPE(0.08), one(ELTYPE),
                                                             method) === ELTYPE(0)
            @test TrixiParticles.gradient_interface_activity(ELTYPE(0.09), one(ELTYPE),
                                                             method) ≈ ELTYPE(0.5)
            @test TrixiParticles.gradient_interface_activity(ELTYPE(0.1), one(ELTYPE),
                                                             method) === ELTYPE(1)
            @test TrixiParticles.support_interface_activity(ELTYPE(0.9), method) ===
                  ELTYPE(1)
            @test TrixiParticles.support_interface_activity(ELTYPE(0.925), method) ≈
                  ELTYPE(0.5)
            @test TrixiParticles.support_interface_activity(ELTYPE(0.95), method) ===
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
        wall_reaction = vec(sum(wall_reaction_cache;
                                dims=2))
        force_scale = sum(particle -> norm(active.fluid_system.mass[particle] *
                                           contact_acceleration[:, particle]),
                          eachparticle(active.fluid_system)) +
                      sum(particle -> norm(view(wall_reaction_cache, :, particle)),
                          eachparticle(active.boundary_system))
        @test norm(fluid_force + wall_reaction) / force_scale < 1.0e-12

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
        density_rate = zeros(eltype(active.fluid_system),
                             nparticles(active.fluid_system))
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
                perturbed_density = density[particle] +
                                    epsilon * density_rate[particle]
                colorfield[neighbor] += active.fluid_system.mass[particle] /
                                        perturbed_density *
                                        TrixiParticles.smoothing_kernel(active.fluid_system,
                                                                        sqrt(distance2),
                                                                        particle)
            end
            raw_area = zero(eltype(active.fluid_system))
            for particle in eachparticle(active.boundary_system)
                measure = boundary_cache.wetted_area_surface_measure[particle]
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
              1.0e-12 *
              (norm(edac_force) + norm(edac_reaction))

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
        @test rotated_active_acceleration-rotated_neutral_acceleration≈
        rotation*contact_acceleration rtol=2.0e-12 atol=2.0e-12

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
              1.0e-12 *
              (norm(moving_force) + norm(moving_reaction))
    end

    @testset "Morris CSF local force" begin
        function build_morris_system(solver, particle_count)
            coordinates = zeros(2, particle_count)
            coordinates[1, :] .= range(0.0; step=0.25, length=particle_count)
            initial_condition = InitialCondition(; coordinates,
                                                 velocity=zeros(2, particle_count),
                                                 mass=ones(particle_count),
                                                 density=ones(particle_count),
                                                 particle_spacing=0.25)
            smoothing_kernel = WendlandC2Kernel{2}()
            surface_tension = SurfaceTensionMorris(; surface_tension_coefficient=0.7)
            normal_method = ColorfieldSurfaceNormal(; interface_threshold=0.1)
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
            system.cache.surface_normal .= [1.0; 0.0]
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
            @test all(particle -> effect[:, particle] ≈ expected,
                      axes(effect, 2))
            push!(effects, effect[:, 1])
        end
        @test all(effect -> effect ≈ first(effects), effects)

        system = build_morris_system(:wcsph, 2)
        system.cache.surface_normal .= [1.0; 0.0]
        system.cache.curvature .= 3.0
        system.cache.delta_s .= 2.0
        acceleration = TrixiParticles.surface_tension_acceleration(system.surface_tension,
                                                                   system, 1, 1.0,
                                                                   SVector(0.0, 0.0))
        @test acceleration ≈ SVector(-4.2, 0.0)
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
            return denominator > sqrt(eps()) ? system.cache.curvature[1] / denominator : 0.0
        end

        curvature_zero = curvature_with_neighbor_activity(0.0)
        curvature_small = curvature_with_neighbor_activity(1.0e-6)
        curvature_full = curvature_with_neighbor_activity(1.0)
        @test iszero(curvature_zero)
        @test abs(curvature_small) < 1.0e-4 * abs(curvature_full)
        @test isfinite(curvature_full)
    end

    @testset "cohesion-only systems do not require normals" begin
        coordinates = [0.0 1.0;
                       0.0 0.0]
        initial_condition = InitialCondition(; coordinates, density=ones(2),
                                             particle_spacing=1.0)
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = 1.0
        surface_tension = CohesionForceAkinci(surface_tension_coefficient=0.1)

        wcsph = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                            smoothing_length,
                                            density_calculator=SummationDensity(),
                                            state_equation=StateEquationCole(sound_speed=10.0,
                                                                             reference_density=1.0,
                                                                             exponent=1),
                                            surface_tension)
        edac = EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                           smoothing_length, sound_speed=10.0,
                                           density_calculator=SummationDensity(),
                                           surface_tension)

        for system in (wcsph, edac)
            @test isnothing(system.surface_normal_method)
            @test !haskey(system.cache, :surface_normal)
            @test !haskey(system.cache, :neighbor_count)
            @test !haskey(system.cache, :reference_particle_spacing)
        end

        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               density_calculator=SummationDensity(),
                                                               state_equation=StateEquationCole(sound_speed=10.0,
                                                                                                reference_density=1.0,
                                                                                                exponent=1),
                                                               surface_tension=SurfaceTensionAkinci())
        @test_throws ArgumentError EntropicallyDampedSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               sound_speed=10.0,
                                                               density_calculator=SummationDensity(),
                                                               surface_tension=SurfaceTensionAkinci())

        full_akinci = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                  smoothing_length,
                                                  density_calculator=SummationDensity(),
                                                  state_equation=StateEquationCole(sound_speed=10.0,
                                                                                   reference_density=1.0,
                                                                                   exponent=1),
                                                  surface_tension=SurfaceTensionAkinci(),
                                                  reference_particle_spacing=1.0)
        @test full_akinci.surface_normal_method isa ColorfieldSurfaceNormal
        @test haskey(full_akinci.cache, :surface_normal)
    end

    @testset "surface tension time-step restriction" begin
        function calculate_initial_dt(surface_tension)
            initial_condition = InitialCondition(; coordinates=[0.0 1.0;
                                                                0.0 0.0;
                                                                0.0 0.0],
                                                 density=ones(2), particle_spacing=1.0)
            reference_particle_spacing = isnothing(surface_tension) ? 0 : 1.0
            system = WeaklyCompressibleSPHSystem(initial_condition;
                                                 smoothing_kernel=WendlandC2Kernel{3}(),
                                                 smoothing_length=1.0,
                                                 density_calculator=ContinuityDensity(),
                                                 state_equation=StateEquationCole(sound_speed=10.0,
                                                                                  reference_density=1.0,
                                                                                  exponent=1),
                                                 surface_tension,
                                                 reference_particle_spacing)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.1))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
            return TrixiParticles.calculate_dt(v_ode, u_ode, 0.25, semi.systems[1], semi)
        end

        dt_without_surface_tension = calculate_initial_dt(nothing)
        dt_with_zero_csf = calculate_initial_dt(SurfaceTensionMorris(;
                                                                     surface_tension_coefficient=0.0))
        dt_with_zero_css = calculate_initial_dt(SurfaceTensionMomentumMorris(;
                                                                             surface_tension_coefficient=0.0))
        dt_with_zero_physical = calculate_initial_dt(SurfaceTensionAkinciCohesionPhysical(;
                                                                                          surface_tension_coefficient=0.0,
                                                                                          reference_density=1.0))

        @test dt_with_zero_csf == dt_without_surface_tension
        @test dt_with_zero_css == dt_without_surface_tension
        @test dt_with_zero_physical == dt_without_surface_tension

        physical = SurfaceTensionAkinciCohesionPhysical(;
                                                        surface_tension_coefficient=1000.0,
                                                        reference_density=1.0)
        @test calculate_initial_dt(physical) ≈ sqrt(1 / (2pi * 1000))
        for model in (SurfaceTensionMorris(; surface_tension_coefficient=1000.0),
             SurfaceTensionMomentumMorris(;
                                          surface_tension_coefficient=1000.0))
            @test calculate_initial_dt(model) ≈ sqrt(1 / (2pi * 1000))
        end
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
                                                   pos_diff, test_distance, Val(3)) *
              test_distance
        @test isapprox(val[1], 0.1443038770421044, atol=6e-15)
        @test isapprox(val[2], 0.1443038770421044, atol=6e-15)

        # Maximum repulsion force
        test_distance = 0.01
        max = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance, Val(3)) *
              test_distance
        @test isapprox(max[1], 0.15913517632298307, atol=6e-15)
        @test isapprox(max[2], 0.15913517632298307, atol=6e-15)

        # Near 0
        test_distance = 0.2725
        zero = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, Val(3)) *
               test_distance
        @test isapprox(zero[1], 0.0004360543645195717, atol=6e-15)
        @test isapprox(zero[2], 0.0004360543645195717, atol=6e-15)

        # Maximum attraction force
        test_distance = 0.5
        maxa = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, Val(3)) *
               test_distance
        @test isapprox(maxa[1], -0.15915494309189535, atol=6e-15)
        @test isapprox(maxa[2], -0.15915494309189535, atol=6e-15)

        # Should be 0
        test_distance = 1.0
        zero = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, Val(3)) *
               test_distance
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
                                                    pos_diff, test_distance, 1.0, Val(3)) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        test_distance = 0.5
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0, Val(3)) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        # Near 0
        test_distance = 0.51
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0, Val(3)) *
               test_distance
        @test isapprox(zero[1], -0.002619160170741761, atol=6e-15)
        @test isapprox(zero[2], -0.002619160170741761, atol=6e-15)

        # Maximum adhesion force
        test_distance = 0.75
        max = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance, 1.0, Val(3)) *
              test_distance
        @test isapprox(max[1], -0.004949747468305833, atol=6e-15)
        @test isapprox(max[2], -0.004949747468305833, atol=6e-15)

        # Should be 0
        test_distance = 1.0
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0, Val(3)) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        support_radius_f32 = 15.594092f0
        distance_f32 = prevfloat(support_radius_f32)
        near_support = TrixiParticles.adhesion_force_akinci(surface_tension,
                                                            support_radius_f32, 1.0f0,
                                                            Float32[1, 0], distance_f32,
                                                            1.0f0, Val(3))
        @test eltype(near_support) == Float32
        @test all(isfinite, near_support)
        @test 0 < norm(near_support) < eps(Float32)
    end

    @testset "two-dimensional Akinci kernels" begin
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.0
        cohesion_normalization = 25280 / (627 * pi)

        for distance in (0.25, 0.75)
            pos_diff = SVector(distance, 0.0)
            shape = if distance > 0.5 * support_radius
                (support_radius - distance)^3 * distance^3
            else
                2 * (support_radius - distance)^3 * distance^3 - support_radius^6 / 64
            end
            expected = -cohesion_normalization * shape * pos_diff / distance
            force = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius,
                                                         1.0, pos_diff, distance, Val(2))
            @test isapprox(force, expected; rtol=5eps(), atol=5eps())
        end

        distance = 0.75
        pos_diff = SVector(distance, 0.0)
        radicand = -4 * distance^2 / support_radius + 6 * distance -
                   2 * support_radius
        expected = -(13 / 1200) * radicand^(1 / 4) * pos_diff / distance
        force = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, 1.0,
                                                     pos_diff, distance, 1.0, Val(2))
        @test isapprox(force, expected; rtol=5eps(), atol=5eps())

        surface_tension_f32 = SurfaceTensionAkinci(surface_tension_coefficient=1.0f0)
        distance_f32 = 0.75f0
        pos_diff_f32 = SVector(distance_f32, 0.0f0)
        cohesion_f32 = TrixiParticles.cohesion_force_akinci(surface_tension_f32, 1.0f0,
                                                            1.0f0, pos_diff_f32,
                                                            distance_f32, Val(2))
        adhesion_f32 = TrixiParticles.adhesion_force_akinci(surface_tension_f32, 1.0f0,
                                                            1.0f0, pos_diff_f32,
                                                            distance_f32, 1.0f0, Val(2))
        @test eltype(cohesion_f32) == Float32
        @test eltype(adhesion_f32) == Float32
        @test all(isfinite, cohesion_f32)
        @test all(isfinite, adhesion_f32)
    end

    @testset "Akinci kernel resolution scaling" begin
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.8)
        adhesion_coefficient = 0.6

        function forces(scale, dimensions::Val{NDIMS}) where {NDIMS}
            support_radius = scale
            distance = 0.75 * support_radius
            pos_diff = SVector{NDIMS}(ntuple(i -> i == 1 ? distance : zero(distance),
                                             NDIMS))
            mass = scale^NDIMS
            cohesion = TrixiParticles.cohesion_force_akinci(surface_tension,
                                                            support_radius, mass,
                                                            pos_diff, distance, dimensions)
            adhesion = TrixiParticles.adhesion_force_akinci(surface_tension,
                                                            support_radius, mass,
                                                            pos_diff, distance,
                                                            adhesion_coefficient,
                                                            dimensions)
            return cohesion, adhesion
        end

        for dimensions in (Val(2), Val(3))
            reference_cohesion, reference_adhesion = forces(1.0, dimensions)
            for scale in (0.25, 0.5, 2.0, 4.0)
                cohesion, adhesion = forces(scale, dimensions)
                @test isapprox(cohesion, reference_cohesion; rtol=5eps(), atol=5eps())
                @test isapprox(adhesion, reference_adhesion; rtol=5eps(), atol=5eps())
            end
        end
    end

    @testset "Akinci kernel integral matching" begin
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.3

        function pos_diff_at_radius(radius, ::Val{NDIMS}) where {NDIMS}
            return SVector{NDIMS}(ntuple(i -> i == 1 ? radius : zero(radius), NDIMS))
        end

        function integrate_cohesion(dimensions::Val{NDIMS}) where {NDIMS}
            radial_integral,
            _ = quadgk(0.0, support_radius / 2, support_radius;
                       rtol=1e-13) do radius
                pos_diff = pos_diff_at_radius(radius, dimensions)
                force = TrixiParticles.cohesion_force_akinci(surface_tension,
                                                             support_radius, 1.0,
                                                             pos_diff, radius, dimensions)
                return radius^(NDIMS - 1) * -force[1]
            end
            surface_measure = NDIMS == 2 ? 2pi : 4pi
            return surface_measure * radial_integral
        end

        function integrate_adhesion(dimensions::Val{NDIMS}) where {NDIMS}
            radial_integral,
            _ = quadgk(support_radius / 2, support_radius;
                       rtol=1e-13) do radius
                pos_diff = pos_diff_at_radius(radius, dimensions)
                force = TrixiParticles.adhesion_force_akinci(surface_tension,
                                                             support_radius, 1.0,
                                                             pos_diff, radius, 1.0,
                                                             dimensions)
                return radius^(NDIMS - 1) * -force[1]
            end
            surface_measure = NDIMS == 2 ? 2pi : 4pi
            return surface_measure * radial_integral
        end

        cohesion_2d = integrate_cohesion(Val(2))
        cohesion_3d = integrate_cohesion(Val(3))
        @test isapprox(cohesion_2d, 79 / 336; rtol=1e-12)
        @test isapprox(cohesion_3d, 79 / 336; rtol=1e-12)
        @test isapprox(integrate_adhesion(Val(2)), integrate_adhesion(Val(3));
                       rtol=1e-12)
    end

    @testset "physical Akinci cohesion" begin
        smoothing_kernel = WendlandC2Kernel{3}()
        smoothing_length = 0.5
        support_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)
        reference_density = 2.0
        surface_tension = SurfaceTensionAkinciCohesionPhysical(;
                                                               surface_tension_coefficient=0.3,
                                                               reference_density)
        internal_coefficient = 0.3 /
                               ((21 / 7040) * reference_density^2 * support_radius^2)
        @test TrixiParticles.akinci_physical_cohesion_coefficient(surface_tension,
                                                                  support_radius) ≈
              internal_coefficient
        @test TrixiParticles.akinci_physical_cohesion_coefficient(surface_tension,
                                                                  support_radius / 2) ≈
              4internal_coefficient

        initial_condition = InitialCondition(;
                                             coordinates=[0.0 0.75; 0.0 0.0; 0.0 0.0],
                                             velocity=zeros(3, 2),
                                             mass=ones(2),
                                             density=fill(reference_density, 2),
                                             particle_spacing=0.5)
        system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                             smoothing_length,
                                             density_calculator=ContinuityDensity(),
                                             state_equation=StateEquationCole(;
                                                                              sound_speed=10.0,
                                                                              reference_density,
                                                                              exponent=1),
                                             surface_tension)
        @test isnothing(system.surface_normal_method)
        @test !haskey(system.cache, :surface_normal)

        pos_diff = SVector(-0.75, 0.0, 0.0)
        distance = norm(pos_diff)
        correction = 1.25
        dv_a = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(dv_a, surface_tension, surface_tension,
                                              system, system, 1, 2, pos_diff, distance,
                                              reference_density, reference_density,
                                              zero(pos_diff), correction)
        empirical = CohesionForceAkinci(;
                                        surface_tension_coefficient=internal_coefficient)
        expected = correction *
                   TrixiParticles.cohesion_force_akinci(empirical, support_radius, 1.0,
                                                        pos_diff, distance, Val(3))
        @test isapprox(dv_a[], expected; rtol=2eps(), atol=2eps())

        dv_b = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(dv_b, surface_tension, surface_tension,
                                              system, system, 2, 1, -pos_diff, distance,
                                              reference_density, reference_density,
                                              zero(pos_diff), correction)
        @test isapprox(dv_a[], -dv_b[]; rtol=2eps(), atol=2eps())

        boundary_condition = InitialCondition(;
                                              coordinates=reshape([0.75, 0.0, 0.0], 3, 1),
                                              mass=[2.0],
                                              density=[reference_density],
                                              particle_spacing=0.5)
        boundary_model = BoundaryModelDummyParticles(boundary_condition;
                                                     fluid_system=system)
        boundary_system = WallBoundarySystem(boundary_condition, boundary_model;
                                             adhesion_coefficient=0.5)
        dv_wall = Ref(zero(pos_diff))
        TrixiParticles.adhesion_force!(dv_wall, surface_tension, system, boundary_system,
                                       1, 1, pos_diff, distance)
        wall_model = CohesionForceAkinci(;
                                         surface_tension_coefficient=0.5internal_coefficient)
        expected_wall = TrixiParticles.cohesion_force_akinci(wall_model, support_radius,
                                                             2.0, pos_diff, distance,
                                                             Val(3))
        @test isapprox(dv_wall[], expected_wall; rtol=2eps(), atol=2eps())

        rigid_system = RigidBodySystem(boundary_condition; boundary_model,
                                       adhesion_coefficient=0.5)
        dv_rigid = Ref(zero(pos_diff))
        TrixiParticles.adhesion_force!(dv_rigid, surface_tension, system, rigid_system,
                                       1, 1, pos_diff, distance)
        @test isapprox(dv_rigid[], expected_wall; rtol=2eps(), atol=2eps())
    end

    @testset "complete Akinci pair force" begin
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.0
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.7)
        initial_condition = InitialCondition(; coordinates=[0.0 1.5; 0.0 0.0],
                                             velocity=zeros(2, 2), mass=ones(2),
                                             density=ones(2), particle_spacing=1.0)
        system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                             smoothing_length,
                                             density_calculator=ContinuityDensity(),
                                             state_equation=StateEquationCole(sound_speed=10.0,
                                                                              reference_density=1.0,
                                                                              exponent=1),
                                             surface_tension,
                                             reference_particle_spacing=1.0)
        system.cache.surface_normal[:, 1] .= (0.2, -0.1)
        system.cache.surface_normal[:, 2] .= (-0.3, 0.4)

        pos_diff = SVector(-1.5, 0.0)
        distance = norm(pos_diff)
        correction = 1.25
        dv_a = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(dv_a, surface_tension, surface_tension,
                                              system, system, 1, 2, pos_diff, distance,
                                              1.0, 1.0, zero(pos_diff), correction)

        support_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)
        normal_a = support_radius * SVector(0.2, -0.1)
        normal_b = support_radius * SVector(-0.3, 0.4)
        expected = correction *
                   (TrixiParticles.cohesion_force_akinci(surface_tension,
                                                         support_radius, 1.0,
                                                         pos_diff, distance, Val(2)) -
                    surface_tension.surface_tension_coefficient *
                    (normal_a - normal_b))
        @test isapprox(dv_a[], expected; rtol=2eps(), atol=2eps())

        dv_b = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(dv_b, surface_tension, surface_tension,
                                              system, system, 2, 1, -pos_diff, distance,
                                              1.0, 1.0, zero(pos_diff), correction)
        @test isapprox(dv_a[], -dv_b[]; rtol=2eps(), atol=2eps())
    end

    @testset "Akinci free-surface correction" begin
        correction = AkinciFreeSurfaceCorrection(1000.0)
        @test TrixiParticles.free_surface_correction(correction, nothing, 1000.0,
                                                     1000.0) == (1.0, 1, 1.0)
        expected = 1000.0 / ((500.0 + 1000.0) / 2)
        viscosity, pressure,
        surface_tension = TrixiParticles.free_surface_correction(correction, nothing,
                                                                 500.0, 1000.0)
        @test viscosity == expected
        @test pressure == 1
        @test surface_tension == expected
        @test TrixiParticles.free_surface_correction(correction, nothing, 1000.0,
                                                     500.0) == (expected, 1, expected)
    end

    @testset "Akinci ContinuityDensity reconstruction" begin
        particle_spacing = 1.0
        rho0 = 1000.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        state_equation = StateEquationCole(sound_speed=10.0, reference_density=rho0,
                                           exponent=1)
        correction = AkinciFreeSurfaceCorrection(rho0)
        fluid = RectangularShape(particle_spacing, (7, 7), (0.0, 0.0); density=rho0)

        function correction_density_values(density_calculator)
            system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                 smoothing_length=particle_spacing,
                                                 density_calculator, state_equation,
                                                 correction)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
            density = GC.@preserve v_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                collect(TrixiParticles.current_density(v, system))
            end
            correction_density = [TrixiParticles.correction_density(correction, system,
                                                                    particle,
                                                                    density[particle])
                                  for particle in TrixiParticles.eachparticle(system)]
            return system, density, correction_density
        end

        continuity_system, continuity_density,
        continuity_correction_density = correction_density_values(ContinuityDensity())
        _, summation_density,
        summation_correction_density = correction_density_values(SummationDensity())

        @test all(==(rho0), continuity_density)
        @test isapprox(continuity_system.cache.kernel_summation_density,
                       summation_density; rtol=2eps())
        @test isapprox(continuity_correction_density,
                       summation_correction_density; rtol=2eps())

        coordinates = fluid.coordinates
        particle_at(position) = findfirst(particle -> coordinates[:, particle] == position,
                                          axes(coordinates, 2))
        center = particle_at([3.5, 3.5])
        face = particle_at([3.5, 0.5])
        corner = particle_at([0.5, 0.5])
        k = rho0 ./ continuity_correction_density

        @test isapprox(k[center], 1; atol=0.002)
        @test k[face] > 1.15
        @test k[corner] > k[face]

        # Dummy boundary masses complete the kernel sum at a wall, so wall particles are
        # not mistaken for a free surface by the reconstructed density.
        tank = RectangularTank(particle_spacing, (7.0, 5.0), (7.0, 8.0), rho0;
                               n_layers=2, faces=(false, false, true, false))
        wall_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                  smoothing_length=particle_spacing,
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation, correction)
        boundary_model = BoundaryModelDummyParticles(tank.boundary;
                                                     fluid_system=wall_system,
                                                     boundary_density_calculator=AdamiPressureExtrapolation())
        boundary_system = WallBoundarySystem(tank.boundary, boundary_model)
        wall_semi = Semidiscretization(wall_system, boundary_system)
        wall_ode = semidiscretize(wall_semi, (0.0, 0.01))
        TrixiParticles.update_systems_and_nhs(wall_ode.u0.x..., wall_semi, 0.0)

        wall_coordinates = tank.fluid.coordinates
        wall_particle_at(position) = findfirst(particle -> wall_coordinates[:, particle] ==
                                                           position,
                                               axes(wall_coordinates, 2))
        bottom = wall_particle_at([3.5, 0.5])
        interior = wall_particle_at([3.5, 2.5])
        top = wall_particle_at([3.5, 4.5])
        reconstructed_density = wall_system.cache.kernel_summation_density
        wall_k = rho0 ./ reconstructed_density

        @test isapprox(wall_k[bottom], wall_k[interior]; rtol=2eps())
        @test isapprox(wall_k[interior], 1; atol=0.002)
        @test wall_k[top] > 1.15
    end

    @testset "Akinci pipeline force assembly" begin
        # End-to-end verification: the surface normals are computed by the actual update
        # pipeline (not injected into the cache) and the resulting fluid-fluid RHS
        # contribution is compared against equations 1-5 of Akinci et al. (2013),
        # implemented independently below.
        particle_spacing = 1.0
        # The compact support radius 2.2 lies strictly between the lattice distances
        # 2 and sqrt(5), so the pair set is unambiguous, and every particle of the
        # 4^3 block keeps at least 2^3 + 1 = 9 neighbors, so no normal is filtered.
        smoothing_length = 1.1 * particle_spacing
        smoothing_kernel = SchoenbergCubicSplineKernel{3}()
        support_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)

        coordinates = RectangularShape(particle_spacing, (4, 4, 4), (0.0, 0.0, 0.0);
                                       density=1000.0).coordinates
        n_particles = size(coordinates, 2)
        rho0 = 1000.0
        # Perturb the densities so that the free-surface correction K_ij deviates from one
        density = rho0 .+ 40 .* sin.(range(0, 2pi, length=n_particles))
        mass = fill(rho0 * particle_spacing^3, n_particles)
        surface_tension_coefficient = 0.7
        state_equation = StateEquationCole(sound_speed=10.0, reference_density=rho0,
                                           exponent=7)

        # Note that all variables of this closure must not be assigned anywhere in the
        # enclosing test sets. Otherwise, the closure captures and overwrites them.
        function fluid_fluid_dv(surface_tension_model, correction_model)
            fluid_ic = InitialCondition(; coordinates,
                                        velocity=zeros(3, n_particles),
                                        mass, density, particle_spacing)
            fluid_sys = WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel,
                                                    smoothing_length,
                                                    density_calculator=ContinuityDensity(),
                                                    state_equation,
                                                    surface_tension=surface_tension_model,
                                                    correction=correction_model,
                                                    reference_particle_spacing=particle_spacing)
            semi_ = Semidiscretization(fluid_sys)
            ode_ = semidiscretize(semi_, (0.0, 0.01))
            v_ode_, u_ode_ = ode_.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode_, u_ode_, semi_, 0.0)
            # `wrap_v` and `wrap_u` return raw-pointer arrays. Inside a function, the ODE
            # vectors must be preserved manually, since the garbage collector might
            # otherwise free them after their last syntactic use.
            dv_ = GC.@preserve v_ode_ u_ode_ begin
                v_ = TrixiParticles.wrap_v(v_ode_, fluid_sys, semi_)
                u_ = TrixiParticles.wrap_u(u_ode_, fluid_sys, semi_)
                dv_inner = zeros(eltype(v_), size(v_))
                TrixiParticles.interact!(dv_inner, v_, u_, v_, u_, fluid_sys, fluid_sys,
                                         semi_)
                dv_inner
            end
            return fluid_sys, dv_
        end

        system_akinci,
        dv_akinci = fluid_fluid_dv(SurfaceTensionAkinci(; surface_tension_coefficient),
                                   AkinciFreeSurfaceCorrection(rho0))
        _, dv_without = fluid_fluid_dv(nothing, nothing)

        # The free-surface correction does not modify the pressure force and no viscosity
        # is used, so the difference of the two right-hand sides isolates the cohesion,
        # curvature and K_ij contributions of the Akinci model.
        dv_surface_tension = (dv_akinci-dv_without)[1:3, :]

        # No normal may have been removed by the neighbor-count filter
        @test all(>=(2^3 + 1), system_akinci.cache.neighbor_count)

        # Independent references for the colorfield gradient of section 2.2 and the
        # auxiliary summation density used by the ContinuityDensity extension.
        gradients = zeros(3, n_particles)
        kernel_summation_density = zeros(n_particles)
        for a in 1:n_particles, b in 1:n_particles
            pos_diff = SVector{3}(coordinates[:, a] - coordinates[:, b])
            distance = norm(pos_diff)
            kernel_summation_density[a] += mass[b] *
                                           TrixiParticles.kernel(smoothing_kernel, distance,
                                                                 smoothing_length)
            (distance < eps() || distance > support_radius) && continue
            grad = TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, distance,
                                              smoothing_length)
            gradients[:, a] .+= mass[b] / density[b] .* grad
        end

        # The pipeline stores the unscaled colorfield gradient
        @test isapprox(gradients, system_akinci.cache.surface_normal; atol=1e-12)
        @test isapprox(kernel_summation_density,
                       system_akinci.cache.kernel_summation_density; rtol=2eps())

        # Independent reference for equations 1-5. The normal of equation 2 is the
        # gradient scaled with the compact support radius.
        dv_expected = zeros(3, n_particles)
        for a in 1:n_particles, b in 1:n_particles
            pos_diff = SVector{3}(coordinates[:, a] - coordinates[:, b])
            distance = norm(pos_diff)
            (distance < eps() || distance > support_radius) && continue
            correction_factor = 2 * rho0 /
                                (kernel_summation_density[a] +
                                 kernel_summation_density[b])
            cohesion_kernel = if 2 * distance > support_radius
                (support_radius - distance)^3 * distance^3
            else
                2 * (support_radius - distance)^3 * distance^3 - support_radius^6 / 64
            end
            cohesion_kernel *= 32 / (pi * support_radius^9)
            normal_difference = support_radius .* (gradients[:, a] - gradients[:, b])
            dv_expected[:,
                        a] .+= correction_factor .*
                               (-surface_tension_coefficient .* mass[b] .*
                                cohesion_kernel .* pos_diff ./ distance .-
                                surface_tension_coefficient .* normal_difference)
        end

        @test maximum(abs, dv_expected) > 0
        @test isapprox(dv_surface_tension, dv_expected; rtol=1e-8,
                       atol=1e-9 * maximum(abs, dv_expected))
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
        @test !haskey(system.cache, :boundary_normal)

        # The surface delta must be captured before the color gradient is normalized.
        system.cache.surface_normal .= [2.0 1.0; 0.0 1.0]
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
        system.cache.interface_activity .= [0.25, 0.75]
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

        system.cache.divergence_correction .= 0
        unsupported_force = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(unsupported_force, surface_tension,
                                              surface_tension, system, system, 1, 2,
                                              pos_diff, distance, rho_a, rho_b,
                                              grad_kernel, 1.0)
        @test iszero(unsupported_force[])
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
                                                                                               ideal_density_threshold=0.9),
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
