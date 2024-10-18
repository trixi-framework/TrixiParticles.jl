struct NaiveWinding end

@inline function (winding::NaiveWinding)(polygon::Polygon{2}, query_point)
    (; edge_vertices_ids) = polygon

    return naive_winding(polygon, edge_vertices_ids, query_point)
end

@inline function (winding::NaiveWinding)(mesh::TriangleMesh{3}, query_point)
    (; face_vertices_ids) = mesh

    return naive_winding(mesh, face_vertices_ids, query_point)
end

@inline function naive_winding(polygon::Polygon{2}, edges, query_point)
    winding_number = sum(edges, init=zero(eltype(polygon))) do edge
        v1 = polygon.vertices[edge[1]]
        v2 = polygon.vertices[edge[2]]

        a = v1 - query_point
        b = v2 - query_point

        return atan(det([a b]), (dot(a, b)))
    end

    return winding_number
end

@inline function naive_winding(mesh::TriangleMesh{3}, faces, query_point)
    winding_number = sum(faces, init=zero(eltype(mesh))) do face
        v1 = mesh.vertices[face[1]]
        v2 = mesh.vertices[face[2]]
        v3 = mesh.vertices[face[3]]

        # Eq. 6 of Jacobsen et al. Based on A. Van Oosterom (1983),
        # "The Solid Angle of a Plane Triangle" (doi: 10.1109/TBME.1983.325207)
        a = v1 - query_point
        b = v2 - query_point
        c = v3 - query_point
        a_ = norm(a)
        b_ = norm(b)
        c_ = norm(c)

        divisor = a_ * b_ * c_ + dot(a, b) * c_ + dot(b, c) * a_ + dot(c, a) * b_

        return 2 * atan(det([a b c]), divisor)
    end

    return winding_number
end

"""
    WindingNumberJacobson(; geometry=nothing, winding_number_factor=sqrt(eps()),
                          hierarchical_winding=false)
Algorithm for inside-outside segmentation of a complex geometry proposed by [Jacobson2013](@cite).

# Keywords
- `geometry`: Complex geometry returned by [`load_geometry`](@ref) and is only required when using
              `hierarchical_winding=true`.
- `hierarchical_winding`: If set to `true`, an optimized hierarchical approach will be used,
                          which gives a significant speedup. For further information see [Hierarchical Winding](@ref hierarchical_winding).
- `winding_number_factor`: For leaky geometries, a factor of `0.4` will give a better inside-outside segmentation.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct WindingNumberJacobson{ELTYPE, W}
    winding_number_factor :: ELTYPE
    winding               :: W

    function WindingNumberJacobson(; geometry=nothing, winding_number_factor=sqrt(eps()),
                                   hierarchical_winding=false)
        if hierarchical_winding && geometry isa Nothing
            throw(ArgumentError("`geometry` must be of type `Polygon` (2D) or `TriangleMesh` (3D) when using hierarchical winding"))
        end

        winding = hierarchical_winding ? HierarchicalWinding(geometry) : NaiveWinding()

        return new{typeof(winding_number_factor), typeof(winding)}(winding_number_factor,
                                                                   winding)
    end
end

function (point_in_poly::WindingNumberJacobson)(geometry, points;
                                                store_winding_number=false)
    (; winding_number_factor, winding) = point_in_poly

    # We cannot use a `BitVector` here, as writing to a `BitVector` is not thread-safe
    inpoly = fill(false, length(points))

    winding_numbers = Float64[]
    store_winding_number && (winding_numbers = resize!(winding_numbers, length(inpoly)))

    divisor = ndims(geometry) == 2 ? 2pi : 4pi

    @threaded points for query_point in eachindex(points)
        p = points[query_point]

        winding_number = winding(geometry, p) / divisor

        store_winding_number && (winding_numbers[query_point] = winding_number)

        # Relaxed restriction of `(winding_number != 0.0)`
        if !(-winding_number_factor < winding_number < winding_number_factor)
            inpoly[query_point] = true
        end
    end

    return inpoly, winding_numbers
end
