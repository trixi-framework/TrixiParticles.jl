# This is the data format returned by `load(file)` when used with `.stl` files
struct TriangleMesh{NDIMS, ELTYPE}
    vertices          :: Vector{SVector{NDIMS, ELTYPE}}
    face_vertices     :: Vector{NTuple{3, SVector{NDIMS, ELTYPE}}}
    face_vertices_ids :: Vector{NTuple{3, Int}}
    face_edges_ids    :: Vector{NTuple{3, Int}}
    edge_vertices_ids :: Vector{NTuple{2, Int}}
    vertex_normals    :: Vector{SVector{NDIMS, ELTYPE}}
    edge_normals      :: Vector{SVector{NDIMS, ELTYPE}}
    face_normals      :: Vector{SVector{NDIMS, ELTYPE}}
    min_corner        :: SVector{NDIMS, ELTYPE}
    max_corner        :: SVector{NDIMS, ELTYPE}

    function TriangleMesh(face_vertices, face_normals, vertices)
        NDIMS = 3

        return TriangleMesh{NDIMS}(face_vertices, face_normals, vertices)
    end

    # Function barrier to make `NDIMS` static and therefore `SVector`s type-stable
    function TriangleMesh{NDIMS}(face_vertices, face_normals, vertices_) where {NDIMS}
        # Sort vertices by the first entry of the vector and return only unique vertices
        vertices = unique_sorted(vertices_)

        ELTYPE = eltype(first(face_normals))
        n_faces = length(face_normals)

        face_vertices_ids = fill((0, 0, 0), n_faces)

        @threaded default_backend(face_vertices) for i in 1:n_faces
            v1 = face_vertices[i][1]
            v2 = face_vertices[i][2]
            v3 = face_vertices[i][3]

            # Since it's only sorted by the first entry, `v1` might be one of the following vertices
            vertex_id1 = searchsortedfirst(vertices, v1 .- 1e-14)
            for vertex_id in eachindex(vertices)[vertex_id1:end]
                if isapprox(vertices[vertex_id], v1)
                    vertex_id1 = vertex_id
                    break
                end
            end

            # Since it's only sorted by the first entry, `v2` might be one of the following vertices
            vertex_id2 = searchsortedfirst(vertices, v2 .- 1e-14)
            for vertex_id in eachindex(vertices)[vertex_id2:end]
                if isapprox(vertices[vertex_id], v2)
                    vertex_id2 = vertex_id
                    break
                end
            end

            # Since it's only sorted by the first entry, `v3` might be one of the following vertices
            vertex_id3 = searchsortedfirst(vertices, v3 .- 1e-14)
            for vertex_id in eachindex(vertices)[vertex_id3:end]
                if isapprox(vertices[vertex_id], v3)
                    vertex_id3 = vertex_id
                    break
                end
            end

            face_vertices_ids[i] = (vertex_id1, vertex_id2, vertex_id3)
        end

        _edges = Dict{NTuple{2, Int}, Int}()
        face_edges_ids = fill((0, 0, 0), n_faces)
        edge_normals = fill(fill(zero(ELTYPE), SVector{NDIMS}), 3n_faces)
        vertex_normals = fill(fill(zero(ELTYPE), SVector{NDIMS}), length(vertices))
        edge_vertices_ids = fill((0, 0), 3n_faces)

        # Not thread supported (yet)
        edge_id = 0
        for i in 1:n_faces
            v1 = face_vertices_ids[i][1]
            v2 = face_vertices_ids[i][2]
            v3 = face_vertices_ids[i][3]

            # Make sure that edges are unique
            if haskey(_edges, (v1, v2))
                edge_id_1 = _edges[(v1, v2)]
            elseif haskey(_edges, (v2, v1))
                edge_id_1 = _edges[(v2, v1)]
            else
                edge_id += 1
                _edges[(v1, v2)] = edge_id
                edge_id_1 = edge_id
            end
            edge_vertices_ids[edge_id_1] = (v1, v2)

            if haskey(_edges, (v2, v3))
                edge_id_2 = _edges[(v2, v3)]
            elseif haskey(_edges, (v3, v2))
                edge_id_2 = _edges[(v3, v2)]
            else
                edge_id += 1
                _edges[(v2, v3)] = edge_id
                edge_id_2 = edge_id
            end
            edge_vertices_ids[edge_id_2] = (v2, v3)

            if haskey(_edges, (v3, v1))
                edge_id_3 = _edges[(v3, v1)]
            elseif haskey(_edges, (v1, v3))
                edge_id_3 = _edges[(v1, v3)]
            else
                edge_id += 1
                _edges[(v3, v1)] = edge_id
                edge_id_3 = edge_id
            end
            edge_vertices_ids[edge_id_3] = (v3, v1)

            face_edges_ids[i] = (edge_id_1, edge_id_2, edge_id_3)

            # Edge normal is the sum of the normals of the two adjacent faces
            edge_normals[edge_id_1] += face_normals[i]
            edge_normals[edge_id_2] += face_normals[i]
            edge_normals[edge_id_3] += face_normals[i]

            angles = incident_angles(face_vertices[i])

            vertex_normals[v1] += angles[1] * face_normals[i]
            vertex_normals[v2] += angles[2] * face_normals[i]
            vertex_normals[v3] += angles[3] * face_normals[i]
        end

        resize!(edge_normals, length(_edges))
        resize!(edge_vertices_ids, length(_edges))

        min_corner = SVector([minimum(v[i] for v in vertices) for i in 1:NDIMS]...)
        max_corner = SVector([maximum(v[i] for v in vertices) for i in 1:NDIMS]...)

        for i in eachindex(edge_normals)
            # Skip zero normals, which would be normalized to `NaN` vectors.
            # The edge normals are only used for the `SignedDistanceField`, which is
            # essential for the packing.
            # Zero normals are caused by exactly or nearly duplicated faces.
            if !iszero(norm(edge_normals[i]))
                edge_normals[i] = normalize(edge_normals[i])
            end
        end

        return new{NDIMS, ELTYPE}(vertices, face_vertices, face_vertices_ids,
                                  face_edges_ids, edge_vertices_ids,
                                  normalize.(vertex_normals), edge_normals,
                                  face_normals, min_corner, max_corner)
    end
end

function Base.show(io::IO, geometry::TriangleMesh)
    @nospecialize geometry # reduce precompilation time

    print(io, "TriangleMesh{$(ndims(geometry)), $(eltype(geometry))}()")
end

function Base.show(io::IO, ::MIME"text/plain", geometry::TriangleMesh)
    @nospecialize geometry # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "TriangleMesh{$(ndims(geometry)), $(eltype(geometry))}")
        summary_line(io, "#faces", "$(nfaces(geometry))")
        summary_line(io, "#vertices", "$(length(geometry.vertices))")
        summary_footer(io)
    end
end

@inline Base.ndims(::TriangleMesh{NDIMS}) where {NDIMS} = NDIMS

@inline Base.eltype(::TriangleMesh{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

@inline face_normal(triangle, geometry::TriangleMesh) = geometry.face_normals[triangle]

@inline function Base.deleteat!(mesh::TriangleMesh, indices)
    (; face_vertices, face_vertices_ids, face_edges_ids, face_normals) = mesh

    deleteat!(face_vertices, indices)
    deleteat!(face_vertices_ids, indices)
    deleteat!(face_edges_ids, indices)
    deleteat!(face_normals, indices)

    return mesh
end

@inline nfaces(mesh::TriangleMesh) = length(mesh.face_normals)

@inline function face_vertices(triangle, geometry::TriangleMesh)
    v1 = geometry.face_vertices[triangle][1]
    v2 = geometry.face_vertices[triangle][2]
    v3 = geometry.face_vertices[triangle][3]

    return v1, v2, v3
end

function incident_angles(triangle_points)
    a = triangle_points[1]
    b = triangle_points[2]
    c = triangle_points[3]

    # Squares of the lengths of the sides
    ab2 = dot(b - a, b - a)
    bc2 = dot(b - c, b - c)
    ac2 = dot(c - a, c - a)

    # Applying the law of cosines
    # https://en.wikipedia.org/wiki/Law_of_cosines
    cos_alpha = (ab2 + ac2 - bc2) / (2 * sqrt(ab2 * ac2))
    cos_beta = (bc2 + ab2 - ac2) / (2 * sqrt(bc2 * ab2))
    cos_gamma = (ac2 + bc2 - ab2) / (2 * sqrt(ac2 * bc2))

    # If one side has length zero, this assures that the two adjacent angles
    # become pi/2, while the third angle becomes zero.
    cos_alpha = isfinite(cos_alpha) ? clamp(cos_alpha, -1, 1) : zero(eltype(ab2))
    cos_beta = isfinite(cos_beta) ? clamp(cos_beta, -1, 1) : zero(eltype(ab2))
    cos_gamma = isfinite(cos_gamma) ? clamp(cos_gamma, -1, 1) : zero(eltype(ab2))

    alpha = acos(cos_alpha)
    beta = acos(cos_beta)
    gamma = acos(cos_gamma)

    return alpha, beta, gamma
end

function unique_sorted(vertices)
    # Sort by the first entry of the vectors
    compare_first_element = (x, y) -> x[1] < y[1]
    vertices_sorted = sort!(vertices, lt=compare_first_element)
    # We cannot use a `BitVector` here, as writing to a `BitVector` is not thread-safe
    keep = fill(true, length(vertices_sorted))

    @threaded default_backend(vertices_sorted) for i in eachindex(vertices_sorted)
        # We only sorted by the first entry, so we have to check all previous vertices
        # until the first entry is too far away.
        j = i - 1
        while j >= 1 && isapprox(vertices_sorted[j][1], vertices_sorted[i][1], atol=1e-14)
            if isapprox(vertices_sorted[i], vertices_sorted[j], atol=1e-14)
                keep[i] = false
            end
            j -= 1
        end
    end

    return vertices_sorted[keep]
end

function volume(mesh::TriangleMesh)
    volume = sum(mesh.face_vertices) do vertices

        # Formula for the volume of a tetrahedron:
        # V = (1/6) * |a · (b × c)|, where a, b, and c are vectors defining the tetrahedron.
        # Reference: https://en.wikipedia.org/wiki/Tetrahedron#Volume
        return dot(vertices[1], cross(vertices[2], vertices[3])) / 6
    end

    return volume
end

"""
    extrude_geometry(geometry_bottom::TriangleMesh,
                     extrude_length::Real; omit_top_face=false,
                     omit_bottom_face=false)

Extruding a 3D geometry returned by [`load_geometry`](@ref) along its averaged face normal.

!!! note
    The extrusion direction is computed as the normalized average of `geometry_bottom.face_normals`,
    assuming the input is planar. The function does not check planarity.
    For non-planar inputs the averaged normal may point in an arbitrary direction;
    verify planarity if strict behavior is required.

Arguments
- `geometry_bottom`: Geometry returned by [`load_geometry`](@ref) representing the base (bottom) surface.
- `extrude_length`: distance to extrude along the averaged face normal.

Keywords
- `omit_top_face=false`: if `true`, the top horizontal faces is not included.
- `omit_bottom_face=false`: if `true`, the bottom horizontal faces is not included.
"""
function extrude_geometry(geometry_bottom::TriangleMesh{NDIMS, ELTYPE},
                          extrude_length::Real; omit_top_face=false,
                          omit_bottom_face=false) where {NDIMS, ELTYPE}
    face_normal = normalize(sum(geometry_bottom.face_normals) / nfaces(geometry_bottom))

    shift = face_normal * extrude_length

    # Bottom (original) geometry data
    vertices_bottom = copy(geometry_bottom.vertices)
    # Flip the winding of the bottom faces so that after extrusion the bottom faces'
    # normals point outward from the resulting solid (consistent with the top faces).
    # We reverse the triangle vertex order (v2 <-> v3) and negate the face normals to
    # reflect that orientation change.
    flipped_faces = copy(geometry_bottom.face_vertices)
    face_vertices_bottom = [(v1, v3, v2) for (v1, v2, v3) in flipped_faces]
    normals_bottom = .-copy(geometry_bottom.face_normals)

    # Top (shifted) geometry data.
    # Shift every bottom vertex by the same vector to create the top vertices.
    vertices_top = [v .+ shift for v in vertices_bottom]
    normals_top = copy(geometry_bottom.face_normals)
    # Shift each face-tuple component-wise to create top face tuples
    face_vertices_top = [(v1 .+ shift, v2 .+ shift, v3 .+ shift)
                         for (v1, v2, v3) in flipped_faces]

    # Build a temporary `TriangleMesh` for the top layer to reuse its indexing helpers
    geometry_top = TriangleMesh(face_vertices_top, normals_top, vertices_top)

    # Find boundary edges on bottom to generate side faces only for exterior edges.
    directed_edges = zeros(Int, length(geometry_bottom.edge_normals))
    for face in eachindex(geometry_bottom.face_vertices)
        (v1, v2, v3) = geometry_bottom.face_vertices_ids[face]
        (e1, e2, e3) = geometry_bottom.face_edges_ids[face]
        directed_edges[e1] += (geometry_bottom.edge_vertices_ids[e1] == (v1, v2) ? 1 : -1)
        directed_edges[e2] += (geometry_bottom.edge_vertices_ids[e2] == (v2, v3) ? 1 : -1)
        directed_edges[e3] += (geometry_bottom.edge_vertices_ids[e3] == (v3, v1) ? 1 : -1)
    end

    boundary_edges = findall(!iszero, directed_edges)

    @inline function triangle_normal(v1, v2, v3)
        n = cross(v2 - v1, v3 - v1)
        norm(n) > 0 && return normalize(n)

        # Return zero for degenerate triangles
        return zero(n)
    end

    # Compute approximate geometric center (used to ensure face winding produces outward normals)
    center = (reduce(+, vertices_bottom) / length(vertices_bottom)) .+ (shift / 2)

    faces_vertices_closing = Tuple{SVector{3, ELTYPE}, SVector{3, ELTYPE},
                                   SVector{3, ELTYPE}}[]
    normals_closing = SVector{3, ELTYPE}[]
    for edge in boundary_edges
        # Bottom edge endpoints (in the bottom mesh vertex list)
        v1b = geometry_bottom.vertices[geometry_bottom.edge_vertices_ids[edge][1]]
        v2b = geometry_bottom.vertices[geometry_bottom.edge_vertices_ids[edge][2]]

        # Corresponding top edge endpoints (in the top mesh vertex list)
        v1t = geometry_top.vertices[geometry_top.edge_vertices_ids[edge][1]]
        v2t = geometry_top.vertices[geometry_top.edge_vertices_ids[edge][2]]

        # Split the quad (v1b, v2b, v2t, v1t) into two triangles
        face_1 = (v1b, v2b, v2t)
        face_2 = (v1b, v2t, v1t)

        n1 = triangle_normal(face_1[1], face_1[2], face_1[3])
        n2 = triangle_normal(face_2[1], face_2[2], face_2[3])

        # Compute correct winding so normals point away from the body center
        c1 = (face_1[1] + face_1[2] + face_1[3]) / 3
        if dot(n1, c1 - center) < 0
            face_1 = (face_1[1], face_1[3], face_1[2])
            n1 = -n1
        end

        c2 = (face_2[1] + face_2[2] + face_2[3]) / 3
        if dot(n2, c2 - center) < 0
            face_2 = (face_2[1], face_2[3], face_2[2])
            n2 = -n2
        end

        push!(faces_vertices_closing, face_1)
        push!(normals_closing, n1)
        push!(faces_vertices_closing, face_2)
        push!(normals_closing, n2)
    end

    vertices_new = vcat(vertices_bottom, vertices_top)
    if omit_bottom_face && omit_top_face
        # Only sides
        face_vertices_new = vcat(faces_vertices_closing)
        normals_new = vcat(normals_closing)
    elseif omit_bottom_face
        # top + sides
        face_vertices_new = vcat(face_vertices_top, faces_vertices_closing)
        normals_new = vcat(normals_top, normals_closing)
    elseif omit_top_face
        # bottom + sides
        face_vertices_new = vcat(face_vertices_bottom, faces_vertices_closing)
        normals_new = vcat(normals_bottom, normals_closing)
    else
        # bottom + top + sides (default)
        face_vertices_new = vcat(face_vertices_bottom, face_vertices_top,
                                 faces_vertices_closing)
        normals_new = vcat(normals_bottom, normals_top, normals_closing)
    end

    return TriangleMesh(face_vertices_new, normals_new, vertices_new)
end

function Base.union(geometry::TriangleMesh, geometries::TriangleMesh...)
    other_geometry = first(geometries)

    vertices_other = copy(other_geometry.vertices)
    face_vertices_other = copy(other_geometry.face_vertices)
    normals_other = copy(other_geometry.face_normals)

    vertices = copy(geometry.vertices)
    face_vertices = copy(geometry.face_vertices)
    normals = copy(geometry.face_normals)

    vertices_new = vcat(vertices, vertices_other)
    face_vertices_new = vcat(face_vertices, face_vertices_other)
    normals_new = vcat(normals, normals_other)

    result = TriangleMesh(face_vertices_new, normals_new, vertices_new)

    return union(result, Base.tail(geometries)...)
end

Base.union(geometry::TriangleMesh) = geometry
