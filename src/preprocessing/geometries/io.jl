"""
    load_geometry(filename; element_type=Float64)

Load file and return corresponding type for [`ComplexShape`](@ref).
Supported file formats are `.stl`, `.asc` and `dxf`.
For comprehensive information about the supported file formats, refer to the documentation at
[Read geometries from file](@ref read_geometries_from_file).

# Arguments
- `filename`: Name of the file to be loaded.

# Keywords
- `element_type`: Element type (default is `Float64`)
"""
function load_geometry(filename; element_type=Float64)
    ELTYPE = element_type

    file_extension = splitext(filename)[end]

    if file_extension == ".asc"
        geometry = load_ascii(filename; ELTYPE, skipstart=1)
    elseif file_extension == ".dxf"
        geometry = load_dxf(filename; ELTYPE)
    elseif file_extension == ".stl"
        geometry = load(FileIO.query(filename); ELTYPE)
    else
        throw(ArgumentError("Only `.stl`, `.asc` and `.dxf` files are supported (yet)."))
    end

    return geometry
end

function load_ascii(filename; ELTYPE=Float64, skipstart=1)

    # Read the data from the ASCII file in as a matrix of coordinates.
    # Ignore the first `skipstart` lines of the file (e.g. headers).
    points = DelimitedFiles.readdlm(filename, ' ', ELTYPE, '\n'; skipstart)[:, 1:2]

    return Polygon(copy(points'))
end

function load_dxf(filename; ELTYPE=Float64)
    points = Tuple{ELTYPE, ELTYPE}[]

    load_dxf!(points, filename)

    return Polygon(stack(points))
end

function load_dxf!(points::Vector{Tuple{T, T}}, filename) where {T}
    open(filename) do io
        lines = readlines(io)

        # Check if the DXF file contains line entities ("LINE") or polyline entities ("POLYLINE")
        idx_first_line = findfirst(x -> strip(x) == "LINE", lines)
        idx_first_polyline = findfirst(x -> strip(x) == "POLYLINE", lines)

        if !isnothing(idx_first_line) && isnothing(idx_first_polyline)
            # The file contains only simple line entities ("LINE")
            i = idx_first_line

            while i <= length(lines)
                if strip(lines[i]) == "LINE"
                    if idx_first_line == i
                        # For a polygon, we only need to store the start point of the first line entity.
                        # For subsequent lines, the end point of the previous edge is the start point of the current edge,
                        # so storing all start points would result in duplicate vertices.
                        # Therefore, we only push the start point for the very first line.

                        # Search for the coordinates of the start point:
                        # Group codes "10", "20", "30" represent x, y, z of the start point
                        while i <= length(lines) && strip(lines[i]) != "10"
                            i += 1
                        end
                        x1 = parse(T, strip(lines[i + 1]))
                        @assert strip(lines[i + 2]) == "20"
                        y1 = parse(T, strip(lines[i + 3]))

                        push!(points, (x1, y1))
                    end

                    # Search for the end point coordinates:
                    # Group codes "11", "21", "31" represent x, y, z of the end point
                    while i <= length(lines) && strip(lines[i]) != "11"
                        i += 1
                    end
                    x2 = parse(T, strip(lines[i + 1]))
                    @assert strip(lines[i + 2]) == "21"
                    y2 = parse(T, strip(lines[i + 3]))

                    # Add end point of the line to the point list
                    push!(points, (x2, y2))
                end
                i += 1
            end

        elseif isnothing(idx_first_line) && !isnothing(idx_first_polyline)
            # The file contains only polyline entities ("POLYLINE")
            i = idx_first_polyline

            while i <= length(lines)
                line = strip(lines[i])
                if line == "VERTEX"
                    # Search for the coordinates of the current vertex:
                    # Group codes "10", "20", "30" represent x, y, z of the vertex
                    while i <= length(lines) && strip(lines[i]) != "10"
                        i += 1
                    end
                    x = parse(T, strip(lines[i + 1]))
                    @assert strip(lines[i + 2]) == "20"
                    y = parse(T, strip(lines[i + 3]))

                    # Add the vertex to the point list
                    push!(points, (x, y))

                elseif line == "SEQEND"
                    # End of the polyline reached
                    break
                end
                i += 1
            end
        else
            throw(ArgumentError("This entity type is not supported. Only 'LINE' OR 'POLYLINE' are allowed."))
        end
    end

    # Remove duplicate points from the list
    unique(points)

    return points
end

# FileIO.jl docs:
# https://juliaio.github.io/FileIO.jl/stable/implementing/#All-at-once-I/O:-implementing-load-and-save
function load(fn::FileIO.File{FileIO.format"STL_BINARY"}; element_types...)
    open(fn) do s
        FileIO.skipmagic(s) # skip over the magic bytes
        load(s; element_types...)
    end
end

function load(fs::FileIO.Stream{FileIO.format"STL_BINARY"}; ELTYPE=Float64)
    # Binary STL
    # https://en.wikipedia.org/wiki/STL_%28file_format%29#Binary_STL
    io = FileIO.stream(fs)
    read(io, 80) # Throw out 80 bytes header
    n_faces = read(io, UInt32)

    face_vertices = fill(ntuple(_ -> fill(zero(ELTYPE), SVector{3}), 3), n_faces)
    vertices = fill(fill(zero(ELTYPE), SVector{3}), 3n_faces)
    normals = fill(fill(zero(ELTYPE), SVector{3}), n_faces)

    load_data!(face_vertices, vertices, normals, io)

    return TriangleMesh(face_vertices, normals, vertices)
end

function load_data!(face_vertices::Vector{Tuple{SVector{3, T}, SVector{3, T},
                                                SVector{3, T}}},
                    vertices, normals, io) where {T}
    i = 0
    while !eof(io)
        normals[i + 1] = SVector{3, T}(read(io, Float32), read(io, Float32),
                                       read(io, Float32))

        v1 = SVector{3, T}(read(io, Float32), read(io, Float32), read(io, Float32))
        v2 = SVector{3, T}(read(io, Float32), read(io, Float32), read(io, Float32))
        v3 = SVector{3, T}(read(io, Float32), read(io, Float32), read(io, Float32))

        face_vertices[i + 1] = (v1, v2, v3)

        vertices[3 * i + 1] = v1
        vertices[3 * i + 2] = v2
        vertices[3 * i + 3] = v3

        # After the 48 bytes of the normal and the vertices follows a 2-byte unsigned integer
        # that is the "attribute byte count" – in the standard format, this should be zero
        # because most software does not understand anything else.
        skip(io, 2) # Skip attribute byte count
        i += 1
    end
end

function trixi2vtk(geometry::Polygon; output_directory="out", prefix="",
                   filename="points", custom_quantities...)
    vertex_normals = Vector{SVector{2, eltype(geometry)}}()
    vertices = Vector{SVector{2, eltype(geometry)}}()

    # Add each vertex twice (once at the end of an edge and once at the start of the next edge)
    # with corresponding normals to make ParaView work.
    for edge in eachface(geometry)
        push!(vertex_normals, geometry.vertex_normals[edge][1])
        push!(vertex_normals, geometry.vertex_normals[edge][2])
        push!(vertices, geometry.vertices[geometry.edge_vertices_ids[edge][1]])
        push!(vertices, geometry.vertices[geometry.edge_vertices_ids[edge][2]])
    end

    return trixi2vtk(stack(vertices); output_directory, filename, prefix,
                     vertex_normals=vertex_normals, custom_quantities...)
end

function trixi2vtk(geometry::TriangleMesh; output_directory="out", prefix="",
                   filename="points", custom_quantities...)
    vertex_normals = stack([geometry.vertex_normals[face]
                            for face in eachindex(geometry.vertices)])

    return trixi2vtk(stack(geometry.vertices); output_directory, filename, prefix,
                     vertex_normals=vertex_normals, custom_quantities...)
end
