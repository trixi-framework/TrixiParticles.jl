struct ParticlePackingSystem{NDIMS, ELTYPE <: Real, B, K, S, C, N} <: FluidSystem{NDIMS}
    initial_condition              :: InitialCondition{ELTYPE}
    boundary                       :: B
    smoothing_kernel               :: K
    smoothing_length               :: ELTYPE
    background_pressure            :: ELTYPE
    tlsph                          :: Bool
    signed_distance_field          :: S
    is_boundary                    :: Bool
    shift_condition                :: ELTYPE
    constrain_particles_on_surface :: C
    neighborhood_search            :: N

    function ParticlePackingSystem(initial_condition;
                                   smoothing_kernel=SchoenbergCubicSplineKernel{ndims(initial_condition)}(),
                                   smoothing_length=1.2initial_condition.particle_spacing,
                                   signed_distance_field=nothing,
                                   is_boundary=false,
                                   neighborhood_search=true,
                                   background_pressure, boundary, tlsph=false)
        (; particle_spacing, density) = initial_condition
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        # Search radius for the corresponding neighborhood search.
        # A boundary needs a larger radius, since constraining the boundary particles to the
        # surface needs at least a `search_radius` of the thickness of the boundary.
        search_radius = if is_boundary
            signed_distance_field.max_signed_distance
        else
            compact_support(smoothing_kernel, smoothing_length)
        end

        # Sample boundary
        if is_boundary
            if isnothing(signed_distance_field)
                throw(ArgumentError("`ParticlePackingSystem` needs a `SignedDistanceField` " *
                                    "when `is_boundary=true` to sample the shell of `boundary`"))
            end

            (; positions, distances) = signed_distance_field

            shift_condition = tlsph ? 0.5particle_spacing : particle_spacing

            boundary_coordinates = stack(positions[distances .> shift_condition])

            ic = InitialCondition(; coordinates=boundary_coordinates,
                                  density=first(density), particle_spacing)
        else
            ic = initial_condition
        end

        # Create neighborhood search
        if neighborhood_search && (is_boundary || isnothing(signed_distance_field))
            nhs = FaceNeighborhoodSearch{NDIMS}(search_radius)
            initialize!(nhs, boundary)

        elseif signed_distance_field isa SignedDistanceField
            nhs = GridNeighborhoodSearch{NDIMS}(search_radius,
                                                length(signed_distance_field.distances))

            # Initialize neighborhood search with signed distances
            initialize!(nhs, stack(signed_distance_field.positions))
        else
            nhs = TrivialNeighborhoodSearch{NDIMS}(search_radius, eachface(boundary))
        end

        shift_condition = if is_boundary
            -signed_distance_field.max_signed_distance
        else
            tlsph ? zero(ELTYPE) : 0.5particle_spacing
        end

        constrain_particles_on_surface = if is_boundary || isnothing(signed_distance_field)
            constrain_particles_on_surface_1!
        else
            constrain_particles_on_surface_2!
        end

        return new{NDIMS, ELTYPE, typeof(boundary), typeof(smoothing_kernel),
                   typeof(signed_distance_field),
                   typeof(constrain_particles_on_surface),
                   typeof(nhs)}(ic, boundary, smoothing_kernel, smoothing_length,
                                background_pressure, tlsph, signed_distance_field,
                                is_boundary, shift_condition,
                                constrain_particles_on_surface, nhs)
    end
end

function Base.show(io::IO, system::ParticlePackingSystem)
    @nospecialize system # reduce precompilation time

    print(io, "ParticlePackingSystem{", ndims(system), "}(")
    print(io, ", ", system.smoothing_kernel)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::ParticlePackingSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "ParticlePackingSystem{$(ndims(system))}")
        summary_line(io, "neighborhood search",
                     system.neighborhood_search |> typeof |> nameof)
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "#faces", nfaces(system.boundary))
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "mesh", system.boundary |> typeof |> nameof)
        summary_line(io, "tlsph", system.tlsph ? "yes" : "no")
        summary_line(io, "boundary", system.is_boundary ? "yes" : "no")
        summary_footer(io)
    end
end

function write2vtk!(vtk, v, u, t, system::ParticlePackingSystem; write_meta_data=true)
    if write_meta_data
        vtk["n_neigboring_faces"] = [length(collect(eachneighbor(current_coords(u, system,
                                                                                particle),
                                                                 system.neighborhood_search)))
                                     for particle in eachparticle(system)]
        # vtk["signed_distances"] = system.signed_distances
    end
end

write_v0!(v0, system::ParticlePackingSystem) = v0 .= zero(eltype(system))

@inline source_terms(system::ParticlePackingSystem) = nothing
@inline add_acceleration!(dv, particle, system::ParticlePackingSystem) = dv

# Number of particles in the system
@inline nparticles(system::ParticlePackingSystem) = length(system.initial_condition.mass)

function update_particle_packing(system::ParticlePackingSystem, v_ode, u_ode,
                                 semi, integrator)
    u = wrap_u(u_ode, system, semi)

    update_position!(u, system)

    update_transport_velocity!(system, v_ode, semi)
end

function update_position!(u, system::ParticlePackingSystem)
    (; constrain_particles_on_surface) = system

    func_name = "$(constrain_particles_on_surface)"
    @trixi_timeit timer() func_name constrain_particles_on_surface(u, system)

    return u
end

# TODO: Naming not really creative
function constrain_particles_on_surface_1!(u, system::ParticlePackingSystem)
    (; boundary, shift_condition, neighborhood_search, initial_condition) = system
    (; particle_spacing) = initial_condition

    @threaded for particle in eachparticle(system)
        particle_position = current_coords(u, system, particle)

        distance2 = Inf
        distance_sign = true
        normal_vector = fill(0.0, SVector{ndims(system), eltype(system)})

        # Calculate minimum unsigned distance to boundary
        for face in eachneighbor(particle_position, neighborhood_search)
            new_distance_sign, new_distance2, n = signed_point_face_distance(particle_position,
                                                                             boundary, face)

            if new_distance2 < distance2
                distance2 = new_distance2
                distance_sign = new_distance_sign
                normal_vector = n
            end
        end

        distance = distance_sign ? -sqrt(distance2) : sqrt(distance2)

        if distance >= -shift_condition
            # Constrain outside particles onto surface
            shift = (distance + shift_condition) * normal_vector

            for dim in 1:ndims(system)
                u[dim, particle] -= shift[dim]
            end
        end

        if system.is_boundary
            shift_condition2 = system.tlsph ? particle_spacing : 0.5particle_spacing
            if distance <= shift_condition2
                # Constrain outside particles onto surface
                shift = (distance - shift_condition2) * normal_vector

                for dim in 1:ndims(system)
                    u[dim, particle] -= shift[dim]
                end
            end
        end
    end
end

function constrain_particles_on_surface_2!(u, system::ParticlePackingSystem)
    (; neighborhood_search, signed_distance_field, shift_condition) = system
    (; positions, distances, normals) = signed_distance_field

    search_radius2 = compact_support(system, system)^2

    @threaded for particle in eachparticle(system)
        particle_position = current_coords(u, system, particle)

        volume = zero(eltype(system))
        distance_signed = zero(eltype(system))
        normal_vector = fill(volume, SVector{ndims(system), eltype(system)})

        for neighbor in eachneighbor(particle_position, neighborhood_search)
            pos_diff = positions[neighbor] - particle_position
            distance2 = dot(pos_diff, pos_diff)
            distance2 > search_radius2 && continue

            distance = sqrt(distance2)
            kernel_weight = smoothing_kernel(system, distance)

            distance_signed += distances[neighbor] * kernel_weight

            normal_vector += normals[neighbor] * kernel_weight

            volume += kernel_weight
        end

        if volume > eps()
            distance_signed /= volume
            normal_vector /= volume
            if distance_signed >= -shift_condition
                # Constrain outside particles onto surface
                shift = (distance_signed + shift_condition) * normal_vector

                for dim in 1:ndims(system)
                    u[dim, particle] -= shift[dim]
                end
            end
        end
    end

    return u
end

@inline function update_transport_velocity!(system::ParticlePackingSystem, v_ode, semi)
    v = wrap_v(v_ode, system, semi)
    for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            # Use the initial condition to store the evolved velocity
            system.initial_condition.velocity[i, particle] = v[i, particle]

            # The particle velocity is set to zero at the beginning of each time step to
            # achieve a fully stationary state.
            v[i, particle] = zero(eltype(system))
        end
    end

    return system
end

# Evolved velocity is stored in `initial_condition`
@inline function add_velocity!(du, v, particle, system::ParticlePackingSystem)
    for i in 1:ndims(system)
        du[i, particle] = system.initial_condition.velocity[i, particle]
    end

    return du
end
