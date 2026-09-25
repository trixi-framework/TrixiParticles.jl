# Implicit domain and boundary level-set constraints.
"""
    ReconstructionDomain(min_corner, max_corner; open_faces=(false, false, false, false, false, false))

Axis-aligned domain that clips the reconstructed surface. `min_corner` / `max_corner` are
the interior corners of the domain box; the surface is constrained to lie inside it.
`open_faces` selects faces that are not walls, in `(-x, +x, -y, +y, -z, +z)` order (the
`RectangularTank` convention): an open face is treated as infinitely far away. The
production tank convention is `(false, false, false, true, false, false)` — only the
`+y` face open, so the free surface may rise above the tank. Use all `false` for a
closed box, e.g. the padded grid itself.
"""
struct ReconstructionDomain{NDIMS, NFACES}
    min_corner::SVector{NDIMS, Float64}
    max_corner::SVector{NDIMS, Float64}
    open_faces::NTuple{NFACES, Bool}
end

function ReconstructionDomain(min_corner, max_corner;
                              open_faces=ntuple(_ -> false, 2length(min_corner)))
    n = length(min_corner)
    n in (2, 3) && length(max_corner) == n && length(open_faces) == 2n ||
        throw(ArgumentError("domain corners and face flags must describe a 2D or 3D box"))
    return ReconstructionDomain(SVector{n, Float64}(min_corner),
                                SVector{n, Float64}(max_corner),
                                NTuple{2n, Bool}(open_faces))
end

# Production tank: lower corner at the world origin, open at the top.
function tank_domain(tank_size)
    n = length(tank_size)
    return ReconstructionDomain(zero(SVector{n, Float64}), SVector{n, Float64}(tank_size);
                                open_faces=ntuple(face -> face == 4, 2n))
end

# Signed distance to the domain walls (positive inside). Open faces contribute `Inf`;
# `min`/`max` are exact, so open faces never change the result bitwise.
@inline function domain_distance(domain::ReconstructionDomain, x, y, z)
    (; min_corner, max_corner, open_faces) = domain
    return min(open_faces[1] ? Inf : x - min_corner[1],
               open_faces[2] ? Inf : max_corner[1] - x,
               open_faces[3] ? Inf : y - min_corner[2],
               open_faces[4] ? Inf : max_corner[2] - y,
               open_faces[5] ? Inf : z - min_corner[3],
               open_faces[6] ? Inf : max_corner[3] - z)
end

# Every value depends only on its grid index, so computing a `region` gives bitwise the
# same values as the full grid.
function domain_constraint!(constraint, origin, spacing, domain::ReconstructionDomain,
                            distance_scale; backend=PolyesterBackend(),
                            region=axes(constraint))
    x_indices, y_indices, z_indices = region
    @threaded backend for z_index in z_indices
        z = origin[3] + (z_index - 1) * spacing
        @inbounds for y_index in y_indices
            y = origin[2] + (y_index - 1) * spacing
            @simd for x_index in x_indices
                x = origin[1] + (x_index - 1) * spacing
                distance = domain_distance(domain, x, y, z)
                constraint[x_index, y_index, z_index] = Float32(distance / distance_scale)
            end
        end
    end
    return constraint
end

# Largest violation of the domain walls by a point (0 when inside).
@inline function domain_violation(domain::ReconstructionDomain, point)
    return max(zero(eltype(point)), -domain_distance(domain, point...))
end

# 0-based lower and upper grid indices of the voxels sampled for a boundary
function boundary_constraint_box(boundary, origin, spacing, clearance, dimensions)
    margin = clearance + 2spacing
    lower = floor.(Int, (boundary.lower .- margin .- origin) ./ spacing)
    upper = ceil.(Int, (boundary.upper .+ margin .- origin) ./ spacing)
    return max.(lower, 0), min.(upper, SVector{3, Int}(dimensions) .- 1)
end

# Grid regions (1-based ranges) that `add_boundary_constraints!` modifies
function boundary_constraint_regions(boundaries, origin, spacing, clearance, dimensions)
    return map(boundaries) do boundary
        lower,
        upper = boundary_constraint_box(boundary, origin, spacing, clearance,
                                        dimensions)
        return ntuple(axis -> (lower[axis] + 1):(upper[axis] + 1), 3)
    end
end

function add_boundary_constraints!(constraint, boundaries, origin, spacing, clearance,
                                   distance_scale; backend=PolyesterBackend())
    sampled_points = 0
    for boundary in boundaries
        lower,
        upper = boundary_constraint_box(boundary, origin, spacing, clearance,
                                        size(constraint))
        # Boundaries away from the grid do not constrain it (and an empty box would
        # otherwise be traversed with negative extents)
        any(upper .< lower) && continue
        local_dimensions = upper - lower .+ 1
        local_count = prod(local_dimensions)
        sampled_points += local_count
        local_x, local_y, _ = local_dimensions
        @threaded backend for local_linear in 0:(local_count - 1)
            local_index = local_linear
            x_index = lower[1] + rem(local_index, local_x)
            local_index = fld(local_index, local_x)
            y_index = lower[2] + rem(local_index, local_y)
            z_index = lower[3] + fld(local_index, local_y)
            point = origin + spacing * SVector{3, Float64}(x_index, y_index, z_index)
            value = Float32((signed_distance(boundary.bvh, point) - clearance) /
                            distance_scale)
            array_index = (x_index + 1, y_index + 1, z_index + 1)
            @inbounds constraint[array_index...] = min(constraint[array_index...], value)
        end
    end
    return sampled_points
end
