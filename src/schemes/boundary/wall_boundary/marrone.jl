function initialize_marrone!(model, initial_condition)
    return model
end

function initialize_marrone!(model::BoundaryModelDummyParticles{MarronePressureExtrapolation},
                             initial_condition)
    (; coordinates, normals) = initial_condition
    isnothing(normals) &&
        throw(ArgumentError("`MarronePressureExtrapolation` requires boundary normals"))

    interpolation_coordinates = model.cache.interpolation_coordinates
    size(coordinates) == size(interpolation_coordinates) ||
        throw(ArgumentError("the boundary model and initial condition must have the same size"))
    all(isfinite, normals) || throw(ArgumentError("boundary normals must be finite"))

    for particle in axes(normals, 2)
        any(!iszero, view(normals, :, particle)) ||
            throw(ArgumentError("boundary normals must be nonzero for every particle"))
    end

    initial_interpolation_coordinates = model.cache.initial_interpolation_coordinates
    initial_interpolation_coordinates .= coordinates .- 2 .* normals
    interpolation_coordinates .= initial_interpolation_coordinates

    return model
end

function update_marrone_interpolation_coordinates!(system, model, prescribed_motion, t,
                                                   semi)
    return model
end

function update_marrone_interpolation_coordinates!(system,
                                                   model::BoundaryModelDummyParticles{MarronePressureExtrapolation},
                                                   prescribed_motion::PrescribedMotion,
                                                   t, semi)
    system.ismoving[] || return model

    (; movement_function, moving_particles) = prescribed_motion
    (; interpolation_coordinates, initial_interpolation_coordinates) = model.cache

    @threaded semi for particle in moving_particles
        initial_position = extract_svector(initial_interpolation_coordinates, system,
                                           particle)
        position = movement_function(initial_position, t)
        for dimension in eachindex(position)
            interpolation_coordinates[dimension, particle] = position[dimension]
        end
    end

    return model
end

function compute_pressure!(model, ::MarronePressureExtrapolation,
                           system, v, u, v_ode, u_ode, semi)
    (; cache, pressure) = model
    set_zero!(pressure)
    set_zero!(cache.moment_matrix)
    set_zero!(cache.pressure_rhs)
    set_zero!(cache.velocity_rhs)
    set_zero!(cache.volume)
    if haskey(cache, :wall_velocity)
        set_zero!(cache.wall_velocity)
    end

    system_coordinates = current_coordinates(u, system)

    @trixi_timeit timer() "compute boundary pressure" begin
        foreach_system(semi) do neighbor_system
            has_system_interaction(system, neighbor_system, semi) || return
            neighbor_system isa AbstractFluidSystem || return

            v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
            neighbor_coordinates = current_coordinates(u_neighbor_system, neighbor_system)

            accumulate_marrone!(model, system, neighbor_system, system_coordinates,
                                neighbor_coordinates, v_neighbor_system, semi)
        end
    end

    @trixi_timeit timer() "inverse state equation" @threaded semi for particle in
                                                                      eachparticle(system)
        finalize_marrone!(model, system, v, particle)
    end

    return model
end

function accumulate_marrone!(model, system, neighbor_system, system_coordinates,
                             neighbor_coordinates, v_neighbor_system, semi)
    interpolation_coordinates = model.cache.interpolation_coordinates

    foreach_point_neighbor(system, neighbor_system, interpolation_coordinates,
                           neighbor_coordinates, semi;
                           points=eachparticle(system)) do particle, neighbor,
                                                           pos_diff, distance
        @inbounds accumulate_marrone_pair!(model, system, neighbor_system,
                                           system_coordinates, v_neighbor_system,
                                           particle, neighbor, pos_diff, distance)
    end

    return model
end

@propagate_inbounds function accumulate_marrone_pair!(model, system, neighbor_system,
                                                      system_coordinates,
                                                      v_neighbor_system, particle,
                                                      neighbor, pos_diff, distance)
    (; cache, smoothing_length) = model
    NDIMS = ndims(system)

    # Scaling the linear basis by the smoothing length does not change the MLS
    # interpolant, but makes its conditioning independent of the unit of length.
    basis = SVector{NDIMS + 1}(ntuple(i -> i == 1 ? one(distance) :
                                           -pos_diff[i - 1] / smoothing_length,
                                      NDIMS + 1))
    density = current_density(v_neighbor_system, neighbor_system, neighbor)
    iszero(density) && return model

    volume = hydrodynamic_mass(neighbor_system, neighbor) / density
    weight = smoothing_kernel(model, distance, particle) * volume

    boundary_position = extract_svector(system_coordinates, system, particle)
    interpolation_position = extract_svector(cache.interpolation_coordinates, system,
                                             particle)
    acceleration = acceleration_source(neighbor_system) -
                   current_acceleration(system, particle)
    pressure = current_pressure(v_neighbor_system, neighbor_system, neighbor) +
               density * dot(acceleration, boundary_position - interpolation_position)
    velocity = current_velocity(v_neighbor_system, neighbor_system, neighbor)

    for i in 1:(NDIMS + 1)
        cache.pressure_rhs[i, particle] += weight * basis[i] * pressure
        for j in 1:(NDIMS + 1)
            cache.moment_matrix[i, j, particle] += weight * basis[i] * basis[j]
        end
        for dimension in 1:NDIMS
            velocity_contribution = weight * basis[i] * velocity[dimension]
            cache.velocity_rhs[i, dimension, particle] += velocity_contribution
        end
    end

    return model
end

# Return the coefficients that evaluate the MLS polynomial at the origin. For
# deficient support, use the constant part of the moment matrix (Shepard interpolation).
@inline function marrone_mls_coefficients(moment::SMatrix{N, N, ELTYPE}) where {N, ELTYPE}
    unit = SVector{N, ELTYPE}(ntuple(i -> i == 1 ? one(ELTYPE) : zero(ELTYPE), N))
    volume = moment[1, 1]
    volume > eps(ELTYPE) || return zero(unit)

    normalized_moment = moment / volume
    if abs(det(normalized_moment)) > sqrt(eps(ELTYPE))
        coefficients = inv(normalized_moment) * unit / volume
        all(isfinite, coefficients) && return coefficients
    end

    return unit / volume
end

function finalize_marrone!(model, system, v, particle)
    (; cache, pressure, viscosity, state_equation) = model
    NDIMS = ndims(system)
    N = NDIMS + 1
    ELTYPE = eltype(cache.density)

    moment = SMatrix{N, N, ELTYPE}(ntuple(i -> cache.moment_matrix[mod1(i, N), cld(i, N),
                                                                   particle], N * N))
    coefficients = marrone_mls_coefficients(moment)
    pressure_rhs = SVector{N, ELTYPE}(ntuple(i -> cache.pressure_rhs[i, particle], N))
    particle_pressure = dot(coefficients, pressure_rhs)

    if clip_negative_pressure(model)
        particle_pressure = max(particle_pressure, zero(particle_pressure))
    end
    pressure[particle] = particle_pressure
    inverse_state_equation!(cache.density, state_equation, pressure, particle)

    if !isnothing(viscosity) && !iszero(coefficients)
        velocity_rhs = SMatrix{N, NDIMS, ELTYPE}(ntuple(i -> cache.velocity_rhs[mod1(i, N),
                                                                                cld(i, N),
                                                                                particle],
                                                        N * NDIMS))
        interpolated_velocity = transpose(velocity_rhs) * coefficients
        for dimension in 1:NDIMS
            cache.wall_velocity[dimension, particle] = interpolated_velocity[dimension]
        end
        # `compute_wall_velocity!` expects the interpolated velocity in `wall_velocity`
        # and a normalization factor in `volume`. The MLS result is already normalized.
        cache.volume[particle] = one(ELTYPE)
        compute_wall_velocity!(viscosity, system, v, particle)
    end

    return model
end
