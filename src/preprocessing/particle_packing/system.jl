mutable struct ParticlePackingSystem{NDIMS, ELTYPE <: Real, B, K} <: FluidSystem{NDIMS}
    initial_condition   :: InitialCondition{ELTYPE}
    boundary            :: B
    smoothing_kernel    :: K
    smoothing_length    :: ELTYPE
    background_pressure :: ELTYPE
    tlsph               :: Bool
    closest_face        :: Vector{Int} # TODO: Remove this (only for debugging)
    nhs_faces           :: Union{TrivialNeighborhoodSearch, FaceNeighborhoodSearch}

    function ParticlePackingSystem(initial_condition, smoothing_kernel, smoothing_length;
                                   background_pressure, boundary, tlsph=false)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        closest_face = zeros(Int, nparticles(initial_condition))
        return new{NDIMS, ELTYPE, typeof(boundary),
                   typeof(smoothing_kernel)}(initial_condition, boundary, smoothing_kernel,
                                             smoothing_length, background_pressure, tlsph,
                                             closest_face)
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
        summary_line(io, "neighborhood search", system.nhs_faces |> typeof |> nameof)
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "#faces", nfaces(system.boundary))
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "mesh", system.boundary |> typeof |> nameof)
        summary_line(io, "tlsph", system.tlsph ? "yes" : "no")
        summary_footer(io)
    end
end

function write2vtk!(vtk, v, u, t, system::ParticlePackingSystem; write_meta_data=true)
    if write_meta_data
        vtk["n_neigboring_faces"] = [length(eachneighborface(current_coords(u, system,
                                                                            particle),
                                                             system.nhs_faces))
                                     for particle in eachparticle(system)]
        vtk["closest_face"] = system.closest_face
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

    @trixi_timeit timer() "bounding method" update_position!(u, system)

    update_transport_velocity!(system, v_ode, semi)
end

function update_position!(u, system::ParticlePackingSystem)
    (; boundary, initial_condition, nhs_faces, closest_face) = system
    (; particle_spacing) = initial_condition

    shift_condition = system.tlsph ? zero(eltype(system)) : 0.5particle_spacing

    @threaded for particle in eachparticle(system)
        particle_position = current_coords(u, system, particle)

        distance2 = Inf
        distance_sign = -1.0
        normal_vector = fill(0.0, SVector{ndims(system)})

        # Determine minimal unsigned distance to boundary
        for face in eachneighborface(particle_position, nhs_faces)
            new_distance_sign, new_distance2, n = signed_point_face_distance(particle_position,
                                                                             boundary, face)

            if new_distance2 < distance2
                distance2 = new_distance2
                distance_sign = new_distance_sign
                normal_vector = n
                closest_face[particle] = face
            end
        end

        distance = distance_sign * sqrt(distance2)

        if distance >= -shift_condition
            # Constrain outside particles onto surface
            shift = (distance + shift_condition) * normal_vector

            for dim in 1:ndims(system)
                u[dim, particle] -= shift[dim]
            end
        end
    end
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
