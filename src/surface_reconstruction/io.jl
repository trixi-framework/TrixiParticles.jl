# Binary PLY output with nesting-aware face winding.
"""
    surface_vertex_normals(mesh)

Area-weighted per-vertex normals with nesting-aware face winding (outer shells point
outward, cavity shells inward), shared by the PLY and VTK writers.
"""
function surface_vertex_normals(mesh; analysis=nothing)
    parent, reverse_winding = reflected_component_winding(mesh; analysis)
    reverse_flag = vertex_reverse_flags(parent, reverse_winding, length(mesh.vertices))
    normals = fill(SVector{3, Float64}(0, 0, 0), length(mesh.vertices))
    @inbounds for face in mesh.faces
        first, second,
        third = reverse_flag[face[1]] ?
                (face[1], face[3], face[2]) :
                (face[1], face[2], face[3])
        a = SVector{3, Float64}(mesh.vertices[first])
        b = SVector{3, Float64}(mesh.vertices[second])
        c = SVector{3, Float64}(mesh.vertices[third])
        face_normal = cross(b - a, c - a)
        normals[first] += face_normal
        normals[second] += face_normal
        normals[third] += face_normal
    end
    @inbounds for index in eachindex(normals)
        magnitude = norm(normals[index])
        normals[index] = magnitude > eps(Float64) ? normals[index] / magnitude :
                         SVector{3, Float64}(0, 1, 0)
    end
    return normals, parent, reverse_winding
end

# Store `value` as little-endian bytes (the writer requires a little-endian host)
@inline function store_little_endian!(bytes, offset, value::Int32)
    @inbounds for byte in 0:3
        bytes[offset + byte] = UInt8((value >> (8 * byte)) & 0xff)
    end
    return bytes
end

"""
    write_ply(mesh, path; analysis=nothing)

Write a `SurfaceMesh` as a binary little-endian PLY file with per-vertex normals and
nesting-aware face winding, in simulation coordinates. Outer-shell normals point out of
the liquid, and cavity normals into the cavity. `analysis` optionally supplies the
`mesh_liquid_analysis` of the current mesh; it is invalid after changing vertices or faces.
Planar contours are written as PLY `edge` elements in the z=0 plane.
"""
function write_ply(mesh, path; analysis=nothing)
    Base.ENDIAN_BOM == 0x04030201 ||
        error("binary PLY writer requires a little-endian host")
    normals, parent, reverse_winding = surface_vertex_normals(mesh; analysis)
    reverse_flag = vertex_reverse_flags(parent, reverse_winding, length(mesh.vertices))

    # Bulk buffers instead of one `write` call per value: identical bytes, far fewer calls
    vertex_data = Matrix{Float32}(undef, 6, length(mesh.vertices))
    @inbounds for index in eachindex(mesh.vertices)
        point = mesh.vertices[index]
        normal = normals[index]
        for component in 1:3
            vertex_data[component, index] = Float32(point[component])
            vertex_data[component + 3, index] = Float32(normal[component])
        end
    end
    face_data = Vector{UInt8}(undef, 13 * length(mesh.faces))
    @inbounds for (face_number, face) in enumerate(mesh.faces)
        first, second,
        third = reverse_flag[face[1]] ?
                (face[1], face[3], face[2]) :
                (face[1], face[2], face[3])
        offset = 13 * (face_number - 1)
        face_data[offset + 1] = 0x03
        store_little_endian!(face_data, offset + 2, Int32(first - 1))
        store_little_endian!(face_data, offset + 6, Int32(second - 1))
        store_little_endian!(face_data, offset + 10, Int32(third - 1))
    end

    open(path, "w") do io
        print(io,
              "ply\n",
              "format binary_little_endian 1.0\n",
              "comment Julia volume-CIC surface reconstruction\n",
              "element vertex $(length(mesh.vertices))\n",
              "property float x\nproperty float y\nproperty float z\n",
              "property float nx\nproperty float ny\nproperty float nz\n",
              "element face $(length(mesh.faces))\n",
              "property list uchar int vertex_indices\n",
              "end_header\n")
        write(io, vertex_data)
        write(io, face_data)
    end
    return path
end

# PLY edge elements preserve a 2D contour as lines in the z=0 plane. No artificial
# thickness or triangulated interior is introduced.
function write_ply(mesh::SurfaceMesh{T, I, 2}, path; analysis=nothing) where {T, I}
    Base.ENDIAN_BOM == 0x04030201 ||
        error("binary PLY writer requires a little-endian host")
    normals, analysis = surface_vertex_normals(mesh; analysis)
    vertices = zeros(Float32, 6, length(mesh.vertices))
    for index in eachindex(mesh.vertices)
        vertices[1:2, index] .= mesh.vertices[index]
        vertices[4:5, index] .= normals[index]
    end
    edges = Matrix{Int32}(undef, 2, length(mesh.faces))
    for (index, edge) in enumerate(analysis.oriented_faces)
        edges[:, index] .= edge .- 1
    end
    open(path, "w") do io
        print(io, "ply\nformat binary_little_endian 1.0\n",
              "comment Julia area-CIC surface reconstruction\n",
              "element vertex $(length(mesh.vertices))\n",
              "property float x\nproperty float y\nproperty float z\n",
              "property float nx\nproperty float ny\nproperty float nz\n",
              "element edge $(length(mesh.faces))\n",
              "property int vertex1\nproperty int vertex2\nend_header\n")
        write(io, vertices)
        write(io, edges)
    end
    return path
end

function surface_vtk_data(mesh::SurfaceMesh{T, I, 3}, analysis) where {T, I}
    points = Matrix{T}(undef, 3, length(mesh.vertices))
    for (index, vertex) in enumerate(mesh.vertices)
        points[:, index] = vertex
    end
    normals, parent, reverse_winding = surface_vertex_normals(mesh; analysis)
    reverse_flag = vertex_reverse_flags(parent, reverse_winding, length(mesh.vertices))
    cells = [MeshCell(PolyData.Polys(),
                      reverse_flag[face[1]] ?
                      (face[1], face[3], face[2]) : (face[1], face[2], face[3]))
             for face in mesh.faces]
    return points, cells, [normal[index] for index in 1:3, normal in normals]
end

function surface_vtk_data(mesh::SurfaceMesh{T, I, 2}, analysis) where {T, I}
    points = zeros(T, 3, length(mesh.vertices))
    normal_data = zeros(Float64, 3, length(mesh.vertices))
    normals, analysis = surface_vertex_normals(mesh; analysis)
    for index in eachindex(mesh.vertices)
        points[1:2, index] .= mesh.vertices[index]
        normal_data[1:2, index] .= normals[index]
    end
    cells = [MeshCell(PolyData.Lines(), Tuple(edge)) for edge in analysis.oriented_faces]
    return points, cells, normal_data
end

"""
    trixi2vtk(mesh::SurfaceMesh; output_directory="out", prefix="", filename="surface",
              iter=nothing, overwrite=isnothing(iter),
              append_collection=_default_append_collection(iter), t=0.0, compress=1,
              point_data=nothing)

Write a reconstructed surface as VTK PolyData (`.vtp`) with per-vertex `Normals` (reserved
VTK name, used automatically for lighting) and nesting-aware face winding. With `iter`,
files are numbered and collected in a PVD time series; with `overwrite`, a single
`_current` file is reused. Mirrors the naming and collection logic of
[`trixi2vtk`](@ref) for particle data. `point_data` optionally maps names to per-vertex
values (vectors, or matrices with one column per vertex) written as additional VTK point
data, e.g. SPH quantities interpolated onto the surface. `analysis` optionally supplies
the `mesh_liquid_analysis` of the current mesh contents; mutating vertices or faces
invalidates it.
In 2D, faces become VTK line cells in the z=0 plane, with in-plane outward normals.
`compress` selects the zlib level (`0`–`9` or booleans); level 1 compresses almost as
well as level 6 (`true`) at a fifth of the cost.
"""
function trixi2vtk(mesh::SurfaceMesh; output_directory="out", prefix="",
                   filename="surface", iter=nothing, overwrite=isnothing(iter),
                   append_collection=_default_append_collection(iter), t=0.0,
                   compress=1, point_data=nothing, analysis=nothing)
    mkpath(output_directory)

    if isempty(mesh.vertices)
        return nothing
    end

    file_ = joinpath(output_directory,
                     add_underscore_to_optional_prefix(prefix) * filename)
    file, pvd = vtk_output_file_and_collection(file_, iter, overwrite, append_collection)

    # Same element type as the vertices (usually `Float32`): an exact round-trip that
    # halves the point payload compared to `Float64`
    points, cells, normals = surface_vtk_data(mesh, analysis)

    @trixi_timeit timer() "write to vtk" vtk_grid(VTKPolyData(), file, points, cells;
                                                  compress) do vtk
        vtk["Normals"] = normals
        vtk["time"] = t
        vtk["ndims"] = ndims(mesh)

        if point_data !== nothing
            for (name, values) in point_data
                vtk[string(name)] = values
            end
        end

        if !isnothing(pvd)
            # Add to collection
            pvd[t] = vtk
        end
    end

    isnothing(pvd) || vtk_save(pvd)

    return file
end
