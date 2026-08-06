using LinearAlgebra
using OrdinaryDiffEqLowStorageRK
using Printf
using Serialization
using Statistics
using TrixiParticles

include(joinpath(@__DIR__, "boundary_volume.jl"))
include(joinpath(@__DIR__, "simulate.jl"))
include(joinpath(@__DIR__, "corrected_wetted_area_contact.jl"))

function wetted_area_contact_diagnostics(model, system, boundary_system)
    return corrected_wetted_area_contact_diagnostics(model)
end

function wetted_area_contact_diagnostics(::WettedAreaContactAngle, system,
                                         boundary_system)
    cache = system.cache
    reaction = boundary_system.boundary_model.cache.wetted_area_reaction
    wall_resultant = vec(sum(reaction; dims=2))
    wall_force_scale = sum(particle -> norm(view(reaction, :, particle)),
                           eachparticle(boundary_system))
    return (; energy=cache.wetted_area_energy[], raw_area=cache.wetted_area_raw_area[],
            corrected_area=cache.wetted_area[], wall_resultant, wall_force_scale,
            normalized_edge_shift=cache.wetted_area_normalized_edge_shift[],
            evaluations=cache.wetted_area_evaluations[],
            cache_bytes=Base.summarysize((cache.wetted_area_density_conjugate,
                                          cache.wetted_area_energy,
                                          cache.wetted_area_raw_area,
                                          cache.wetted_area,
                                          cache.wetted_area_normalized_edge_shift,
                                          reaction,
                                          boundary_system.boundary_model.cache.wetted_area_surface_measure,
                                          boundary_system.boundary_model.cache.wetted_area_weight,
                                          boundary_system.boundary_model.cache.wetted_area_flooded_reference)))
end

function spherical_cap_initial_condition(contact_angle; drop_volume=1.0e-6,
                                         target_particle_count=750,
                                         reference_density=1000.0,
                                         surface_tension_coefficient=0.072,
                                         lattice_phase=(0.0, 0.0))
    cosine = cosd(contact_angle)
    volume_factor = (1 - cosine)^2 * (2 + cosine)
    sphere_radius = cbrt(3drop_volume / (pi * volume_factor))
    sphere_center_z = -sphere_radius * cosine
    particle_spacing = cbrt(drop_volume / target_particle_count)

    cap_radius = sphere_radius * sind(contact_angle)
    horizontal_radius = contact_angle <= 90 ? cap_radius : sphere_radius
    cap_height = sphere_radius * (1 - cosine)
    n_horizontal = 2ceil(Int, horizontal_radius / particle_spacing)
    n_vertical = ceil(Int, cap_height / particle_spacing)
    horizontal_x = ((1:n_horizontal) .- (n_horizontal + 1) / 2 .+
                    lattice_phase[1]) .* particle_spacing
    horizontal_y = ((1:n_horizontal) .- (n_horizontal + 1) / 2 .+
                    lattice_phase[2]) .* particle_spacing
    vertical = ((1:n_vertical) .- 0.5) .* particle_spacing
    coordinates = [SVector(x, y, z)
                   for z in vertical, y in horizontal_y, x in horizontal_x
                   if x^2 + y^2 + (z - sphere_center_z)^2 <= sphere_radius^2]
    coordinates = reduce(hcat, coordinates)
    mass = fill(reference_density * particle_spacing^3, size(coordinates, 2))

    pressure_jump = 2surface_tension_coefficient / sphere_radius
    state_equation = StateEquationCole(; sound_speed=100.0, reference_density,
                                       exponent=7, clip_negative_pressure=true)
    initial_density = TrixiParticles.inverse_state_equation(state_equation, pressure_jump)
    initial_condition = InitialCondition(;
                                         coordinates,
                                         velocity=zeros(3, size(coordinates, 2)), mass,
                                         density=fill(initial_density,
                                                      size(coordinates, 2)),
                                         particle_spacing)

    return (; initial_condition, state_equation, sphere_radius, sphere_center_z,
            cap_radius, horizontal_radius, pressure_jump, lattice_phase)
end

function apparent_spherical_cap_angle(coordinates, volume, particle_spacing)
    height = maximum(coordinates[3, :]) + particle_spacing / 2
    radius_squared = max(2volume / (pi * height) - height^2 / 3, zero(height))
    base_radius = sqrt(radius_squared)
    angle = 2atand(height / base_radius)
    return (; angle, height, base_radius)
end

function local_circle_contact_angle(coordinates, interface, support_radius,
                                    particle_spacing)
    radial = sqrt.(coordinates[1, :] .^ 2 .+ coordinates[2, :] .^ 2)
    near_wall_particles = findall(interface .&
                                  (coordinates[3, :] .>= -particle_spacing) .&
                                  (coordinates[3, :] .<= 2support_radius))
    length(near_wall_particles) >= 3 ||
        return (; angle=NaN, radius=NaN, center_z=NaN, residual=Inf, particles=0)

    # Reduce each horizontal particle layer to its outer meridional envelope.
    layers = Dict{Int, Vector{Int}}()
    for particle in near_wall_particles
        # Initial particles are centered at half-integer heights. Floor-based bins keep
        # mildly deformed rows together instead of splitting them at rounding ties.
        layer = floor(Int, coordinates[3, particle] / particle_spacing)
        push!(get!(layers, layer, Int[]), particle)
    end
    length(layers) >= 3 ||
        return (; angle=NaN, radius=NaN, center_z=NaN, residual=Inf,
                particles=length(layers))

    z = [mean(coordinates[3, particles]) for particles in values(layers)]
    r = [quantile(radial[particles], 0.9) for particles in values(layers)]
    design = hcat(2z, ones(length(z)))
    rhs = r .^ 2 .+ z .^ 2
    center_z, intercept = design \ rhs
    radius_squared = intercept + center_z^2
    radius_squared > 0 ||
        return (; angle=NaN, radius=NaN, center_z, residual=Inf,
                particles=length(layers))
    radius = sqrt(radius_squared)
    angle = acosd(clamp(-center_z / radius, -1, 1))
    residual = sqrt(mean(abs2, design * [center_z, intercept] - rhs)) / radius^2
    return (; angle, radius, center_z, residual, particles=length(layers))
end

function cap_shape_acceleration(acceleration, coordinates)
    return 2mean(coordinates[3, :] .* acceleration[3, :] .-
                 (coordinates[1, :] .* acceleration[1, :] .+
                  coordinates[2, :] .* acceleration[2, :]) ./ 2)
end

function initial_acceleration_diagnostics(semi, fluid_system, boundary_system, ode)
    v_ode, u_ode = ode.u0.x
    dv_ode = zero(v_ode)
    # The first pass initializes all surface and boundary interaction caches.
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
    v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
    v_boundary = TrixiParticles.wrap_v(v_ode, boundary_system, semi)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    dv = TrixiParticles.wrap_v(dv_ode, fluid_system, semi)
    coordinates = TrixiParticles.current_coordinates(u, fluid_system)
    boundary_coordinates = TrixiParticles.current_coordinates(u_boundary, boundary_system)
    boundary_acceleration = zeros(eltype(fluid_system), 3,
                                  size(coordinates, 2))

    TrixiParticles.foreach_point_neighbor(fluid_system, boundary_system, coordinates,
                                          boundary_coordinates,
                                          semi) do particle, neighbor, pos_diff,
                                                   distance
        rho_a = TrixiParticles.current_density(v, fluid_system, particle)
        rho_b = TrixiParticles.current_density(v_boundary, boundary_system, neighbor)
        grad_kernel = TrixiParticles.smoothing_kernel_grad(fluid_system, pos_diff,
                                                           distance, particle)
        acceleration = Ref(zero(pos_diff))
        TrixiParticles.surface_tension_force!(acceleration, fluid_system.surface_tension,
                                              nothing, fluid_system, boundary_system,
                                              particle, neighbor, pos_diff, distance,
                                              rho_a, rho_b, grad_kernel, 1)
        boundary_acceleration[:, particle] .+= acceleration[]
    end

    total_acceleration = Array(dv[1:3, :])
    return (; total=cap_shape_acceleration(total_acceleration, coordinates),
            boundary=cap_shape_acceleration(boundary_acceleration, coordinates))
end

function css_sessile_drop(contact_angle, final_time, output_path=nothing;
                          drop_volume=1.0e-6, target_particle_count=750,
                          reference_density=1000.0,
                          surface_tension_coefficient=0.072,
                          damping_coefficient=10.0,
                          mechanism=:wetted_area,
                          initial_contact_angle=contact_angle,
                          boundary_contact_threshold=0.0,
                          lattice_phase=(0.0, 0.0),
                          smoothing_length_ratio=1.4,
                          validation_contact_model=nothing,
                          parallelization_backend=SerialBackend())
    setup = spherical_cap_initial_condition(initial_contact_angle; drop_volume,
                                            target_particle_count,
                                            reference_density,
                                            surface_tension_coefficient,
                                            lattice_phase)
    (; initial_condition, state_equation) = setup
    particle_spacing = initial_condition.particle_spacing
    smoothing_kernel = WendlandC2Kernel{3}()
    smoothing_length = smoothing_length_ratio * particle_spacing
    surface_tension = SurfaceTensionMomentumMorris(; surface_tension_coefficient)
    contact_model = if mechanism == :corrected_wetted_area
        isnothing(validation_contact_model) &&
            throw(ArgumentError("`:corrected_wetted_area` requires `validation_contact_model`"))
        validation_contact_model
    elseif mechanism == :wetted_area
        WettedAreaContactAngle(contact_angle)
    elseif mechanism == :none
        nothing
    else
        throw(ArgumentError("unknown contact-angle `mechanism`: $mechanism"))
    end
    surface_normal_method = ColorfieldSurfaceNormal(;
                                                    boundary_contact_threshold,
                                                    interface_threshold=0.01,
                                                    ideal_density_threshold=0.95,
                                                    contact_model)
    active_contact_model = surface_normal_method.contact_model
    viscosity = ArtificialViscosityMonaghan(; alpha=0.2, beta=0.0)
    source_terms = SourceTermDamping(; damping_coefficient)

    fluid_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=ContinuityDensity(),
                                               density_diffusion=DensityDiffusionAntuono(;
                                                                                         delta=0.1),
                                               state_equation,
                                               viscosity, surface_tension,
                                               surface_normal_method,
                                               reference_particle_spacing=particle_spacing,
                                               source_terms)

    target_setup = spherical_cap_initial_condition(contact_angle; drop_volume,
                                                   target_particle_count,
                                                   reference_density,
                                                   surface_tension_coefficient)
    horizontal_radius = max(setup.horizontal_radius, target_setup.horizontal_radius)
    plate_size = max(4horizontal_radius, 12particle_spacing)
    n_plate = round.(Int, (plate_size, plate_size) ./ particle_spacing)
    plate_raw = RectangularShape(particle_spacing, (n_plate..., 3),
                                 (-plate_size / 2, -plate_size / 2,
                                  -3particle_spacing);
                                 density=reference_density)
    surface_measure = nothing
    plate = plate_raw
    if active_contact_model isa WettedAreaContactAngle
        exposed_height = maximum(plate_raw.coordinates[3, :])
        exposed = isapprox.(plate_raw.coordinates[3, :], exposed_height;
                            atol=10eps(abs(exposed_height) + particle_spacing))
        normals = zeros(eltype(plate_raw), size(plate_raw.coordinates))
        normals[3, exposed] .= -particle_spacing / 2
        surface_measure = zeros(eltype(plate_raw), nparticles(plate_raw))
        surface_measure[exposed] .= particle_spacing^2
        plate = InitialCondition(; coordinates=plate_raw.coordinates,
                                 velocity=plate_raw.velocity, mass=plate_raw.mass,
                                 density=plate_raw.density, pressure=plate_raw.pressure,
                                 particle_spacing, normals)
    end
    boundary_mass = akinci_boundary_hydrodynamic_mass(plate, smoothing_kernel,
                                                      smoothing_length,
                                                      reference_density)
    boundary_model = if isnothing(surface_measure)
        BoundaryModelDummyParticles(plate; fluid_system,
                                    hydrodynamic_mass=boundary_mass,
                                    boundary_density_calculator=AdamiPressureExtrapolation(),
                                    viscosity, clip_negative_pressure=true)
    else
        BoundaryModelDummyParticles(plate; fluid_system,
                                    hydrodynamic_mass=boundary_mass,
                                    boundary_density_calculator=AdamiPressureExtrapolation(),
                                    viscosity, clip_negative_pressure=true,
                                    surface_measure)
    end
    boundary_system = WallBoundarySystem(plate, boundary_model)

    semi = Semidiscretization(fluid_system, boundary_system; parallelization_backend)
    ode = semidiscretize(semi, (0.0, final_time))
    acceleration = initial_acceleration_diagnostics(semi, fluid_system, boundary_system,
                                                    ode)
    initial_contact_diagnostics = wetted_area_contact_diagnostics(active_contact_model,
                                                                  fluid_system,
                                                                  boundary_system)
    initial_active_interface = fluid_system.cache.delta_s .> 0
    initial_circle = local_circle_contact_angle(initial_condition.coordinates,
                                                initial_active_interface,
                                                TrixiParticles.compact_support(smoothing_kernel,
                                                                               smoothing_length),
                                                particle_spacing)
    saveat = unique([collect(0.0:0.05:final_time); final_time])
    minimum_dt = Ref(Inf)
    accepted_dt = Float64[]
    record_dt = DiscreteCallback((_, time, _) -> time > 0,
                                 integrator -> begin
                                     dt = abs(integrator.t - integrator.tprev)
                                     minimum_dt[] = min(minimum_dt[], dt)
                                     push!(accepted_dt, dt)
                                     u_modified!(integrator, false)
                                 end;
                                 save_positions=(false, false))
    dtmax = 5.0e-4
    cfl_number = 1.0
    initial_v_ode, initial_u_ode = ode.u0.x
    dt_reference = min(dtmax,
                       TrixiParticles.calculate_dt(initial_v_ode, initial_u_ode, cfl_number,
                                                   semi))
    solution = nothing
    runtime = @elapsed solution = solve(ode, RDPK3SpFSAL35(); abstol=1.0e-7,
                                        reltol=1.0e-4, dtmax,
                                        save_everystep=false, saveat,
                                        callback=record_dt)

    frames = map(snapshot_frame, solution.u, Iterators.repeated(semi), solution.t)
    snapshot = (; times=collect(solution.t), frames)
    if !isnothing(output_path)
        open(output_path, "w") do io
            serialize(io, snapshot)
        end
    end

    initial_system = frames[1].systems[1]
    final_system = frames[end].systems[1]
    v_ode, u_ode = last(solution.u).x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, final_time)
    final_contact_diagnostics = wetted_area_contact_diagnostics(active_contact_model,
                                                                fluid_system,
                                                                boundary_system)
    has_boundary_normal = haskey(fluid_system.cache, :boundary_normal)
    boundary_normal = has_boundary_normal ? fluid_system.cache.boundary_normal :
                      zeros(eltype(fluid_system), 3, nparticles(fluid_system))
    wall_contact = vec(sum(abs2, boundary_normal; dims=1)) .> eps()
    active_interface = fluid_system.cache.delta_s .> 0
    active_contact = count(wall_contact .& active_interface)
    contact_correction = has_boundary_normal ?
                         fluid_system.cache.divergence_correction[wall_contact .& active_interface] :
                         Float64[]
    contact_particles = findall(wall_contact .& active_interface)
    dynamic_angles = [acosd(clamp(dot(boundary_normal[:, particle],
                                      fluid_system.cache.surface_normal[:, particle]),
                                  -1, 1)) for particle in contact_particles]
    normal_weights = fluid_system.cache.delta_s[contact_particles]
    measured_contact_angle = isempty(contact_particles) ? NaN :
                             sum(normal_weights .* dynamic_angles) / sum(normal_weights)
    contact_delta = zeros(eltype(fluid_system), nparticles(fluid_system))
    line_particles = findall(>(0), contact_delta)
    contact_line_delta_range = isempty(line_particles) ? (NaN, NaN) :
                               extrema(contact_delta[line_particles])
    line_contact_angle = if isempty(line_particles)
        NaN
    else
        line_angles = [acosd(clamp(dot(boundary_normal[:, particle],
                                       fluid_system.cache.surface_normal[:, particle]),
                                   -1, 1)) for particle in line_particles]
        sum(contact_delta[line_particles] .* line_angles) /
        sum(contact_delta[line_particles])
    end
    initial_volume = sum(initial_condition.mass ./ initial_system.density)
    final_volume = sum(initial_condition.mass ./ final_system.density)
    initial = apparent_spherical_cap_angle(initial_system.coordinates, initial_volume,
                                           particle_spacing)
    final = apparent_spherical_cap_angle(final_system.coordinates, final_volume,
                                         particle_spacing)
    circle = local_circle_contact_angle(final_system.coordinates, active_interface,
                                        TrixiParticles.compact_support(smoothing_kernel,
                                                                       smoothing_length),
                                        particle_spacing)
    angle_history = map(frames) do frame
        system = frame.systems[1]
        volume = sum(initial_condition.mass ./ system.density)
        apparent_spherical_cap_angle(system.coordinates, volume,
                                     particle_spacing).angle
    end
    speed = sqrt.(vec(sum(abs2, final_system.velocity; dims=1)))
    below_wall = count(<(0), final_system.coordinates[3, :])
    rms_speed = sqrt(mean(abs2, speed))
    density_range = extrema(final_system.density)
    settled = rms_speed < 5.0e-3
    rejected_fraction = solution.stats.nreject /
                        max(solution.stats.naccept + solution.stats.nreject, 1)
    eta_p01 = NaN
    eta_median = NaN
    eta_tail_head = NaN
    if length(accepted_dt) > 7
        eta = accepted_dt[6:(end - 1)] ./ dt_reference
        eta_p01 = quantile(eta, 0.01)
        eta_median = median(eta)
        segment_length = max(1, floor(Int, 0.2length(eta)))
        eta_tail_head = median(last(eta, segment_length)) /
                        median(first(eta, segment_length))
    end
    @printf("CSS cap target=%6.1f deg mechanism=%s particles=%d initial=%7.2f deg cap=%7.2f deg circle=%7.2f deg normal=%7.2f deg below=%d contact=%d/%d rho=[%.3f, %.3f] vrms=%.4e m/s runtime=%.2f s\n",
            contact_angle, String(mechanism), size(final_system.coordinates, 2),
            initial.angle, final.angle, circle.angle, measured_contact_angle,
            below_wall, active_contact, count(wall_contact),
            density_range..., rms_speed, runtime)
    @printf("  angle history: %s\n",
            join((@sprintf("%.2f", angle) for angle in angle_history), ", "))
    @printf("  initial circle: %.2f deg shape acceleration total/boundary: %.4e / %.4e m^2/s^2\n",
            initial_circle.angle, acceleration.total, acceleration.boundary)
    if !isempty(contact_correction)
        @printf("  contact q: [%.4f, %.4f], median %.4f\n",
                extrema(contact_correction)..., median(contact_correction))
    end
    if !isempty(line_particles)
        @printf("  CLF angle: %.2f deg over %d particles, delta=[%.3e, %.3e]\n",
                line_contact_angle, length(line_particles),
                contact_line_delta_range...)
    end
    return (; solution, snapshot, initial, initial_circle, final, circle,
            measured_contact_angle,
            line_contact_angle, below_wall, active_contact,
            wall_contact_particles=count(wall_contact), contact_line_delta_range,
            density_range, rms_speed, settled, runtime, minimum_dt=minimum_dt[],
            dt_reference, eta_p01, eta_median, eta_tail_head,
            accepted_steps=solution.stats.naccept,
            rejected_steps=solution.stats.nreject, rejected_fraction,
            shape_acceleration=acceleration.total,
            boundary_shape_acceleration=acceleration.boundary,
            initial_contact_diagnostics, final_contact_diagnostics,
            mechanism, contact_angle,
            initial_contact_angle, particle_count=size(final_system.coordinates, 2))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (4, 5, 6) ||
        error("usage: css_sessile_drop.jl CONTACT_ANGLE FINAL_TIME OUTPUT.jls MECHANISM " *
              "[TARGET_PARTICLE_COUNT] [DAMPING_COEFFICIENT]")
    contact_angle = parse(Float64, ARGS[1])
    final_time = parse(Float64, ARGS[2])
    mechanism = Symbol(ARGS[4])
    target_particle_count = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 750
    damping_coefficient = length(ARGS) == 6 ? parse(Float64, ARGS[6]) : 10.0
    css_sessile_drop(contact_angle, final_time, ARGS[3]; target_particle_count,
                     damping_coefficient, mechanism)
end
