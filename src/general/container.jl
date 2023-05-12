struct ParticleContainer{S, ELTYPE <: Real, NDIMS}
    initial_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    :: Array{ELTYPE, 2} # [dimension, particle]
    mass                :: Array{ELTYPE, 1} # [particle]
    scheme              :: S
    acceleration        :: SVector{NDIMS, ELTYPE}

    # Convenience constructor for passing a setup as first argument
    function ParticleContainer(setup, scheme;
                               acceleration=ntuple(_ -> 0.0,
                                                   size(setup.coordinates, 1)))
        return ParticleContainer(setup.coordinates, setup.velocities, setup.masses, scheme,
                                 acceleration=acceleration)
    end

    function ParticleContainer(coordinates, velocities, masses, scheme;
                               acceleration=ntuple(_ -> 0.0,
                                                   size(coordinates, 1)))
        NDIMS = size(coordinates, 1)
        ELTYPE = eltype(coordinates)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)

        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        n_particles = size(coordinates, 2)
        if length(masses) != n_particles
            throw(ArgumentError("`masses` must be a vector of length $(n_particles)"))
        end

        return new{typeof(scheme), ELTYPE, NDIMS}(coordinates, velocities, masses, scheme,
                                                  acceleration)
    end
end

function Base.show(io::IO, container::ParticleContainer)
    @nospecialize container # Reduce precompilation time

    print(io, "ParticleContainer{", ndims(container), "}(")
    print(io, container.scheme)
    print(io, ", ", container.acceleration)
    print(io, ") with ", nparticles(container), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", container::ParticleContainer)
    @nospecialize container # Reduce precompilation time

    if get(io, :compact, false)
        show(io, container)
    else
        summary_header(io, "ParticleContainer{$(ndims(container))}")
        summary_line(io, "#particles", nparticles(container))
        summary_line(io, "scheme", container.scheme)
        summary_line(io, "acceleration", container.acceleration)
        summary_footer(io)
    end
end

@inline function initialize!(container::ParticleContainer, neighborhood_search)
    initialize!(container, container.scheme, ndims(container), nparticles(container),
                neighborhood_search)
end

@inline Base.ndims(::ParticleContainer{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(container::ParticleContainer) = eltype(container.initial_coordinates)

# Number of integrated variables in the first component of the ODE system (coordinates)
@inline function u_nvariables(container::ParticleContainer)
    u_nvariables(container, container.scheme)
end

# By default, these are the coordinates. This can be dispatched by scheme.
@inline u_nvariables(container, scheme) = ndims(container)

# Number of integrated variables in the second component
# of the ODE system (velocity and sometimes density)
@inline v_nvariables(container) = ndims(container)

# Number of particles in the container
@inline nparticles(container) = length(container.mass)

# Number of particles in the container whose positions are to be integrated (corresponds to the size of u and du)
@inline n_moving_particles(container) = nparticles(container, container.scheme)

# By default, these are the number of particles. This can be dispatched by scheme.
@inline nparticles(container, scheme) = nparticles(container)

@inline eachparticle(container) = Base.OneTo(nparticles(container))
@inline each_moving_particle(container) = Base.OneTo(n_moving_particles(container))

# This should not be dispatched by container type. We always expect to get a column of `A`.
@inline function extract_svector(A, container, i)
    extract_svector(A, Val(ndims(container)), i)
end

# Return the `i`-th column of the array `A` as an `SVector`.
@inline function extract_svector(A, ::Val{NDIMS}, i) where {NDIMS}
    return SVector(ntuple(@inline(dim->A[dim, i]), NDIMS))
end

# Return `A[:, :, i]` as an `SMatrix`.
@inline function extract_smatrix(A, container, particle)
    # Extract the matrix elements for this particle as a tuple to pass to SMatrix
    return SMatrix{ndims(container), ndims(container)}(
                                                       # Convert linear index to Cartesian index
                                                       ntuple(@inline(i->A[mod(i - 1, ndims(container)) + 1,
                                                                           div(i - 1, ndims(container)) + 1,
                                                                           particle]),
                                                              Val(ndims(container)^2)))
end

# Specifically get the current coordinates of a particle for all container types.
@inline function current_coords(u, container, particle)
    return extract_svector(current_coordinates(u, container), container, particle)
end

@inline function current_coordinates(u, container::ParticleContainer)
    return current_coordinates(u, container.scheme)
end

# This can be dispatched by scheme type, since for some schemes, the current coordinates
# are stored in u, for others in the scheme itself. By default, try to extract them from u.
@inline current_coordinates(u, scheme) = u

# Specifically get the initial coordinates of a particle for all container types.
@inline function initial_coords(container, particle)
    return extract_svector(initial_coordinates(container), container, particle)
end

@inline initial_coordinates(container) = container.initial_coordinates

@inline function current_velocity(v, container, particle)
    return current_velocity(v, container, container.scheme, particle)
end

# This can be dispatched by scheme type
@inline function current_velocity(v, container, scheme, particle)
    return extract_svector(v, container, particle)
end

@inline function smoothing_kernel(container, distance)
    @unpack smoothing_kernel, smoothing_length = container.scheme
    return kernel(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_kernel_deriv(container, distance)
    @unpack smoothing_kernel, smoothing_length = container.scheme
    return kernel_deriv(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_kernel_grad(container, pos_diff, distance)
    @unpack smoothing_kernel, smoothing_length = container.scheme
    return kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)
end
