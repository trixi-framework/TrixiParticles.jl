struct PeriodicBox{NDIMS, ELTYPE}
    min_corner :: SVector{NDIMS, ELTYPE}
    max_corner :: SVector{NDIMS, ELTYPE}
    size       :: SVector{NDIMS, ELTYPE}

    function PeriodicBox(min_corner, max_corner)
        new{length(min_corner), eltype(min_corner)}(min_corner, max_corner,
                                                    max_corner - min_corner)
    end
end

# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `each_moving_particle(system)`.
@inline function for_particle_neighbor(f, system, neighbor_system,
                                       system_coords, neighbor_coords, neighborhood_search;
                                       particles=each_moving_particle(system),
                                       parallel=true)
    for_particle_neighbor(f, system_coords, neighbor_coords, neighborhood_search,
                          particles=particles, parallel=parallel)
end

@inline function for_particle_neighbor(f, system_coords, neighbor_coords,
                                       neighborhood_search;
                                       particles=axes(system_coords, 2), parallel=true)
    for_particle_neighbor(f, system_coords, neighbor_coords, neighborhood_search, particles,
                          Val(parallel))
end

@inline function for_particle_neighbor(f, system_coords::CuArray, neighbor_coords,
                                       neighborhood_search;
                                       particles=axes(system_coords, 2), parallel=true)
    # CUDA.@sync CUDA.@cuda threads=size(system_coords, 2) for_particle_neighbor_kernel(f, system_coords, neighbor_coords, neighborhood_search)
    backend = get_backend(system_coords)
    kernel = for_particle_neighbor_kernel(backend)
    kernel(f, system_coords, neighbor_coords, neighborhood_search, ndrange=length(particles))
    synchronize(backend)
end

@kernel function for_particle_neighbor_kernel(f, system_coords, neighbor_coords, neighborhood_search)
    # particle = CUDA.threadIdx().x
    particle = @index(Global)
    for_particle_neighbor_inner(f, system_coords, neighbor_coords, neighborhood_search, particle)
end

@inline function for_particle_neighbor(f, system_coords, neighbor_coords,
                                       neighborhood_search, particles, parallel::Val{true})
    @threaded for particle in particles
        for_particle_neighbor_inner(f, system_coords, neighbor_coords, neighborhood_search,
                                    particle)
    end

    return nothing
end

@inline function for_particle_neighbor(f, system_coords, neighbor_coords,
                                       neighborhood_search, particles, parallel::Val{false})
    for particle in particles
        for_particle_neighbor_inner(f, system_coords, neighbor_coords, neighborhood_search,
                                    particle)
    end

    return nothing
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function for_particle_neighbor_inner(f, system_coords, neighbor_system_coords,
                                             neighborhood_search, particle)
    (; search_radius, periodic_box) = neighborhood_search

    particle_coords = extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                      particle)
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        neighbor_coords = extract_svector(neighbor_system_coords,
                                          Val(ndims(neighborhood_search)), neighbor)

        pos_diff = particle_coords - neighbor_coords
        distance2 = dot(pos_diff, pos_diff)

        pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2, search_radius,
                                                        periodic_box)

        if distance2 <= search_radius^2
            distance = sqrt(distance2)

            # Inline to avoid loss of performance
            # compared to not using `for_particle_neighbor`.
            @inline f(particle, neighbor, pos_diff, distance)
        end
    end
end

@inline function compute_periodic_distance(pos_diff, distance2, search_radius,
                                           periodic_box::Nothing)
    return pos_diff, distance2
end

@inline function compute_periodic_distance(pos_diff, distance2, search_radius,
                                           periodic_box)
    if distance2 > search_radius^2
        # Use periodic `pos_diff`
        pos_diff -= periodic_box.size .* round.(pos_diff ./ periodic_box.size)
        distance2 = dot(pos_diff, pos_diff)
    end

    return pos_diff, distance2
end

@inline function periodic_coords(coords, periodic_box)
    (; min_corner, size) = periodic_box

    # Move coordinates into the periodic box
    box_offset = floor.((coords .- min_corner) ./ size)

    return coords - box_offset .* size
end

@inline function periodic_coords(coords, periodic_box::Nothing)
    return coords
end

include("trivial_nhs.jl")
include("grid_nhs.jl")
