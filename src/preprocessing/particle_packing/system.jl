"""
    ParticlePackingSystem(initial_condition;
                          boundary, tlsph=false, signed_distance_field=nothing, is_boundary=false,
                          smoothing_kernel=SchoenbergCubicSplineKernel{ndims(initial_condition)}(),
                          smoothing_length=1.2initial_condition.particle_spacing,
                          neighborhood_search=true, background_pressure)
System to generate body fitted particles for complex shapes.
For more information about the methods, see desription below.

# Arguments
- `initial_condition`: [`InitialCondition`](@ref) to be packed.

# Keywords
- `boundary`: Shape returned by [`load_shape`](@ref).
- `background_pressure`: Constant background pressure to physically pack the particles.
                         A large `background_pressure` can cause high accelerations which requires a properly adjusted time-step criterion.
- `tlsph`: With the [`TotalLagrangianSPHSystem`](@ref), particles need to be placed
           on the boundary of the shape and not one particle radius away, as for fluids.
           When `tlsph=true`, particles will be placed on the boundary of the shape.
- `is_boundary`: When `is_boundary=true`, boundary particles will be sampled and packed in an offset surface of the `boundary`.
                 The thickness of the boundary is specified by passing the [`SignedDistanceField`](@ref) of `boundary` with:
                    - `use_for_boundary_packing=true`
                    - `max_signed_distance=boundary_thickness`
- `signed_distance_field`: To constrain particles onto the surface, the information about
                          the signed distance from a particle to a face is required.
                          - (Recommended) If specified with the [`SignedDistanceField`](@ref) of the boundary,
                            the precalculated signed distances will be interpolated to each particle.
                          - If set to `nothing` the signed distance from a particle to a face is calculated directly on the fly
                            which is only recommended when the number of faces are significantly less than the number of particles.
                          Note when `is_boundary=true`, the [`SignedDistanceField`](@ref) of the boundary is mandatory.
- `neighborhood_search`: If set to `true` an internaly generated neighborhood search is used to increase performance.
- `smoothing_kernel`: Smoothing kernel to be used for this system.
                      See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`: Smoothing length to be used for this system.
                      See [Smoothing Kernels](@ref smoothing_kernel).
"""
struct ParticlePackingSystem{NDIMS, ELTYPE <: Real, IC, B, K, S,
                             C, N} <: FluidSystem{NDIMS, IC}
    initial_condition              :: IC
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
    signed_distances               :: Vector{ELTYPE} # Only for visualization
    buffer                         :: Nothing
    update_callback_used           :: Ref{Bool}

    function ParticlePackingSystem(initial_condition;
                                   smoothing_kernel=SchoenbergCubicSplineKernel{ndims(initial_condition)}(),
                                   smoothing_length=1.2initial_condition.particle_spacing,
                                   signed_distance_field=nothing,
                                   is_boundary=false,
                                   neighborhood_search=true,
                                   background_pressure, boundary, tlsph=false)
        (; particle_spacing) = initial_condition
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        search_radius = compact_support(smoothing_kernel, smoothing_length)

        # Create neighborhood search
        if neighborhood_search && isnothing(signed_distance_field)
            # Calculate signed distance on the fly using a face nhs

            nhs = FaceNeighborhoodSearch{NDIMS}(search_radius)
            initialize!(nhs, boundary)

        elseif signed_distance_field isa SignedDistanceField
            # Use precalculated signed distance field and interpolate the distances to the particles
            # using a grid nhs
            (; distances, positions) = signed_distance_field

            nhs = GridNeighborhoodSearch{NDIMS}(search_radius, length(distances))

            # Initialize neighborhood search with signed distances
            PointNeighbors.initialize_grid!(nhs, stack(positions))
        else
            nhs = TrivialNeighborhoodSearch{NDIMS}(search_radius, eachface(boundary))
        end

        shift_condition = if is_boundary
            -signed_distance_field.max_signed_distance
        else
            tlsph ? zero(ELTYPE) : 0.5particle_spacing
        end

        constrain_particles_on_surface = if isnothing(signed_distance_field)
            direct_particle_face_distance!
        else
            interpolate_particle_face_distance!
        end

        return new{NDIMS, ELTYPE, typeof(initial_condition), typeof(boundary),
                   typeof(smoothing_kernel), typeof(signed_distance_field),
                   typeof(constrain_particles_on_surface),
                   typeof(nhs)}(initial_condition, boundary, smoothing_kernel,
                                smoothing_length, background_pressure, tlsph,
                                signed_distance_field, is_boundary, shift_condition,
                                constrain_particles_on_surface, nhs,
                                fill(zero(ELTYPE), nparticles(initial_condition)), nothing,
                                false)
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

function reset_callback_flag(system::ParticlePackingSystem)
    system.update_callback_used[] = false

    return system
end

update_callback_used!(system::ParticlePackingSystem) = system.update_callback_used[] = true

function write2vtk!(vtk, v, u, t, system::ParticlePackingSystem; write_meta_data=true)
    if write_meta_data
        vtk["signed_distances"] = system.signed_distances
    end
end

write_v0!(v0, system::ParticlePackingSystem) = v0 .= zero(eltype(system))

@inline source_terms(system::ParticlePackingSystem) = nothing
@inline add_acceleration!(dv, particle, system::ParticlePackingSystem) = dv

# Number of particles in the system
@inline nparticles(system::ParticlePackingSystem) = length(system.initial_condition.mass)

@inline function hydrodynamic_mass(system::ParticlePackingSystem, particle)
    return system.initial_condition.mass[particle]
end

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

function update_final!(system::ParticlePackingSystem, v, u, v_ode, u_ode, semi, t;
                       update_from_callback=false)
    if !update_from_callback && !(system.update_callback_used[])
        throw(ArgumentError("`UpdateCallback` is required when using `ParticlePackingSystem`"))
    end

    return system
end

function direct_particle_face_distance!(u, system::ParticlePackingSystem)
    (; boundary, neighborhood_search) = system

    @threaded system for particle in eachparticle(system)
        particle_position = current_coords(u, system, particle)

        distance2 = Inf
        distance_sign = true
        normal_vector = fill(zero(eltype(system)), SVector{ndims(system), eltype(system)})

        # Calculate minimum unsigned distance to boundary
        for face in PointNeighbors.eachneighbor(particle_position, neighborhood_search)
            new_distance_sign, new_distance2, n = signed_point_face_distance(particle_position,
                                                                             boundary, face)

            if new_distance2 < distance2
                distance2 = new_distance2
                distance_sign = new_distance_sign
                normal_vector = n
            end
        end

        distance = distance_sign ? -sqrt(distance2) : sqrt(distance2)

        # Store signed distance for visualization
        system.signed_distances[particle] = distance

        constrain_particles!(u, system, particle, distance, normal_vector)
    end

    return u
end

function interpolate_particle_face_distance!(u, system::ParticlePackingSystem)
    (; neighborhood_search, signed_distance_field) = system
    (; positions, distances, normals) = signed_distance_field

    search_radius2 = compact_support(system, system)^2

    @threaded system for particle in eachparticle(system)
        particle_position = current_coords(u, system, particle)

        volume = zero(eltype(system))
        distance_signed = zero(eltype(system))
        normal_vector = fill(volume, SVector{ndims(system), eltype(system)})

        # Interpolate signed distances and normals
        for neighbor in PointNeighbors.eachneighbor(particle_position, neighborhood_search)
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

            # Store signed distance for visualization
            system.signed_distances[particle] = distance_signed

            constrain_particles!(u, system, particle, distance_signed, normal_vector)
        end
    end

    return u
end

function constrain_particles!(u, system, particle, distance_signed, normal_vector)
    (; shift_condition) = system

    if distance_signed >= -shift_condition
        # Constrain outside particles onto surface
        shift = (distance_signed + shift_condition) * normal_vector

        for dim in 1:ndims(system)
            u[dim, particle] -= shift[dim]
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
