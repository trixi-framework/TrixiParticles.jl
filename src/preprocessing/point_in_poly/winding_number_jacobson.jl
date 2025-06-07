struct NaiveWinding end

@inline function (winding::NaiveWinding)(polygon::Polygon, query_point)
    (; edge_vertices_ids) = polygon

    return naive_winding(polygon, edge_vertices_ids, query_point)
end

@inline function (winding::NaiveWinding)(mesh::TriangleMesh, query_point)
    (; face_vertices_ids) = mesh

    return naive_winding(mesh, face_vertices_ids, query_point)
end

@inline function naive_winding(polygon::Polygon, edges, query_point)
    winding_number = sum(edges, init=zero(eltype(polygon))) do edge
        v1 = polygon.vertices[edge[1]]
        v2 = polygon.vertices[edge[2]]

        a = v1 - query_point
        b = v2 - query_point

        return atan(det([a b]), (dot(a, b)))
    end

    return winding_number
end

@inline function naive_winding(mesh::TriangleMesh, faces, query_point)
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
struct WindingNumberJacobson{ELTYPE, W, C}
    winding_number_factor :: ELTYPE
    winding               :: W
    cache                 :: C
end

function WindingNumberJacobson(geometry::Geometry; winding=HierarchicalWinding(geometry),
                               winding_number_factor=eltype(geometry) == Float32 ?
                                                     10 * sqrt(eps()) : sqrt(eps()),
                               store_winding_number=false)
    ELTYPE = eltype(geometry)

    # Only for debugging purposes
    if store_winding_number
        NDIMS = ndims(geometry)

        cache = (winding_numbers=ELTYPE[], grid=SVector{NDIMS, ELTYPE}[])
    else
        cache = nothing
    end

    return WindingNumberJacobson(ELTYPE(winding_number_factor), winding, cache)
end

function Base.show(io::IO, winding::WindingNumberJacobson)
    @nospecialize winding # reduce precompilation time

    print(io, "WindingNumberJacobson{$(type2string(winding.winding))}()")
end

function Base.show(io::IO, ::MIME"text/plain", winding::WindingNumberJacobson)
    @nospecialize winding # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "WindingNumberJacobson")
        summary_line(io, "winding number factor",
                     "$(round(winding.winding_number_factor; digits=3))")
        summary_line(io, "winding", "$(type2string(winding.winding))")
        summary_footer(io)
    end
end

store_winding_number(::WindingNumberJacobson{<:Any, <:Any, Nothing}) = false
store_winding_number(::WindingNumberJacobson) = true

function (point_in_poly::WindingNumberJacobson{ELTYPE})(geometry, points) where {ELTYPE}
    (; winding_number_factor, winding) = point_in_poly

    inpoly = allocate(geometry.parallelization_backend, Bool, length(points))

    if store_winding_number(point_in_poly)
        resize!(point_in_poly.cache.winding_numbers, length(points))
        resize!(point_in_poly.cache.grid, length(points))
        copyto!(point_in_poly.cache.grid, points)
    end

    divisor = ndims(geometry) == 2 ? ELTYPE(2pi) : ELTYPE(4pi)

    @threaded geometry for query_point in eachindex(points)
        p = points[query_point]
        inpoly[query_point] = false

        winding_number = winding(geometry, p) / divisor

        # Relaxed restriction of `(winding_number != 0.0)`
        if !(-winding_number_factor < winding_number < winding_number_factor)
            inpoly[query_point] = true
        end

        if store_winding_number(point_in_poly)
            point_in_poly.cache.winding_numbers[query_point] = winding_number
        end
    end

    return inpoly
end
