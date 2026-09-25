# Particle graph search and volume-equivalent fallback meshes for sparse components.
function particle_components(points, radius)
    radius > 0 && isfinite(radius) ||
        throw(ArgumentError("particle-component radius must be finite and positive"))
    particle_count = size(points, 2)
    parent = Int32.(1:particle_count)
    sizes = ones(Int32, particle_count)
    # Dense coordinates for the neighborhood search; avoid copies when possible.
    coordinates = points isa Matrix{Float64} ? points : Matrix{Float64}(points)
    search = GridNeighborhoodSearch{size(points, 1)}(; search_radius=radius)
    PointNeighbors.initialize!(search, coordinates, coordinates)
    # Serial: concurrent union-find updates would race. Connectivity partitions are
    # order-independent, so the result matches a cell-list traversal exactly.
    foreach_point_neighbor(coordinates, coordinates, search;
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        neighbor == particle && return nothing
        union_vertices!(parent, sizes, Int32(particle), Int32(neighbor))
        return nothing
    end

    by_root = Dict{Int32, Vector{Int32}}()
    for particle in axes(points, 2)
        root = find_root!(parent, Int32(particle))
        push!(get!(by_root, root, Int32[]), Int32(particle))
    end
    components = collect(values(by_root))
    sort!(components; by=first)
    return components
end

function triangulated_sphere_mesh(center::SVector{3, Float64}, radius;
                                  latitude_segments=SPARSE_FALLBACK_LATITUDE_SEGMENTS,
                                  longitude_segments=SPARSE_FALLBACK_LONGITUDE_SEGMENTS)
    latitude_segments >= 3 && longitude_segments >= 3 ||
        throw(ArgumentError("sphere tessellation requires at least three segments per axis"))
    radius > 0 && isfinite(radius) ||
        throw(ArgumentError("sphere radius must be finite and positive"))
    vertices = SVector{3, Float32}[SVector{3, Float32}(center +
                                                       SVector{3, Float64}(0, 0, radius))]
    for latitude in 1:(latitude_segments - 1)
        theta = pi * latitude / latitude_segments
        for longitude in 0:(longitude_segments - 1)
            phi = 2pi * longitude / longitude_segments
            offset = radius * SVector{3, Float64}(sin(theta) * cos(phi),
                                         sin(theta) * sin(phi), cos(theta))
            push!(vertices, SVector{3, Float32}(center + offset))
        end
    end
    push!(vertices, SVector{3, Float32}(center - SVector{3, Float64}(0, 0, radius)))

    faces = Face[]
    sizehint!(faces, 2longitude_segments * (latitude_segments - 1))
    ring(latitude,
         longitude) = 2 + (latitude - 1) * longitude_segments +
                      mod(longitude, longitude_segments)
    function add_outward!(first, second, third)
        a, b,
        c = SVector{3, Float64}(vertices[first]), SVector{3, Float64}(vertices[second]),
            SVector{3, Float64}(vertices[third])
        radial = (a + b + c) / 3 - center
        face = dot(cross(b - a, c - a), radial) > 0 ?
               Face(first, second, third) : Face(first, third, second)
        push!(faces, face)
    end
    for longitude in 0:(longitude_segments - 1)
        add_outward!(1, ring(1, longitude), ring(1, longitude + 1))
    end
    for latitude in 1:(latitude_segments - 2), longitude in 0:(longitude_segments - 1)
        lower_left = ring(latitude, longitude)
        lower_right = ring(latitude, longitude + 1)
        upper_left = ring(latitude + 1, longitude)
        upper_right = ring(latitude + 1, longitude + 1)
        add_outward!(lower_left, upper_left, upper_right)
        add_outward!(lower_left, upper_right, lower_right)
    end
    south = length(vertices)
    for longitude in 0:(longitude_segments - 1)
        add_outward!(south, ring(latitude_segments - 1, longitude + 1),
                     ring(latitude_segments - 1, longitude))
    end
    return SurfaceMesh(vertices, faces)
end

function equivalent_volume_sphere_mesh(center::SVector{3, Float64}, target_volume)
    target_volume > 0 && isfinite(target_volume) ||
        throw(ArgumentError("fallback sphere volume must be finite and positive"))
    unit = triangulated_sphere_mesh(SVector{3, Float64}(0, 0, 0), 1.0)
    radius = cbrt(target_volume / abs(mesh_signed_volume(unit)))
    mesh = triangulated_sphere_mesh(center, radius)
    # Compensate the small Float32 translation/rounding error in exported geometry.
    for _ in 1:2
        radius *= cbrt(target_volume / abs(mesh_signed_volume(mesh)))
        mesh = triangulated_sphere_mesh(center, radius)
    end
    return mesh, radius
end

function component_has_primary_region(field, constraint, points, particles, center,
                                      sphere_radius, origin, spacing, isovalue)
    for particle in particles
        point = point3(points, particle)
        if trilinear_field_value(field, point, origin, spacing) > isovalue &&
           trilinear_field_value(constraint, point, origin, spacing) > 0
            return true
        end
    end

    lower = center - SVector{3, Float64}(sphere_radius, sphere_radius, sphere_radius)
    upper = center + SVector{3, Float64}(sphere_radius, sphere_radius, sphere_radius)
    for particle in particles
        point = point3(points, particle)
        lower = min.(lower, point)
        upper = max.(upper, point)
    end
    lower -= SVector{3, Float64}(spacing, spacing, spacing)
    upper += SVector{3, Float64}(spacing, spacing, spacing)
    first_index = max.(0, floor.(Int, (lower - origin) / spacing))
    last_index = min.(collect(size(field)) .- 1,
                      ceil.(Int, (upper - origin) / spacing))
    for z in first_index[3]:last_index[3], y in first_index[2]:last_index[2],
        x in first_index[1]:last_index[1]
        index = (x + 1, y + 1, z + 1)
        @inbounds if field[index...] > isovalue && constraint[index...] > 0
            return true
        end
    end
    return false
end

function unresolved_sparse_components(field, constraint, points, volumes, components,
                                      excluded_components, origin, spacing, isovalue)
    fallbacks = NamedTuple[]
    for (component_index, particles) in enumerate(components)
        component_index in excluded_components && continue
        source_volume = sum(volumes[particles])
        center = SVector{3, Float64}(0, 0, 0)
        for particle in particles
            center += volumes[particle] * point3(points, particle)
        end
        center /= source_volume
        continuum_radius = cbrt(3source_volume / (4pi))
        component_has_primary_region(field, constraint, points, particles, center,
                                     continuum_radius, origin, spacing, isovalue) &&
            continue
        length(particles) <= SPARSE_COMPONENT_MAXIMUM_PARTICLES ||
            error("unresolved source component $component_index has $(length(particles)) particles; " *
                  "the equivalent-sphere fallback is limited to compact components with at most " *
                  "$SPARSE_COMPONENT_MAXIMUM_PARTICLES particles")

        mesh, radius = equivalent_volume_sphere_mesh(center, source_volume)
        maximum_particle_distance = maximum(norm(point3(points, particle) - center)
                                            for particle in particles; init=0.0)
        maximum_particle_distance <= radius + MESH_CLEANUP_TOLERANCE_M ||
            error("unresolved source component $component_index is not compact enough for its " *
                  "volume-equivalent sphere")
        minimum_constraint = minimum(trilinear_field_value(constraint,
                                                           SVector{3, Float64}(vertex),
                                                           origin, spacing)
                                     for vertex in mesh.vertices)
        minimum_constraint > 0 ||
            error("volume-equivalent sphere for source component $component_index crosses a solid " *
                  "constraint; no unsupported clipping prior is applied")
        push!(fallbacks,
              (component_index=component_index,
               particles=particles,
               source_volume=source_volume,
               center=center,
               radius=radius,
               maximum_particle_distance=maximum_particle_distance,
               minimum_constraint=minimum_constraint,
               mesh=mesh))
    end
    return fallbacks
end

function validate_disjoint_fallbacks(fallbacks)
    for first_index in eachindex(fallbacks),
        second_index in (first_index + 1):length(fallbacks)
        first = fallbacks[first_index]
        second = fallbacks[second_index]
        norm(first.center - second.center) > first.radius + second.radius ||
            error("volume-equivalent spheres for source components $(first.component_index) and " *
                  "$(second.component_index) overlap; merge geometry requires a different prior")
    end
    return fallbacks
end
