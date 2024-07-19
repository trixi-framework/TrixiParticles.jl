"""
    load_shape(filename; element_type=Float64)

Load file and return corresponding type for [`ComplexShape`](@ref).
Supported file formats are `.stl` and `.asc`.

# Arguments
- `filename`: Name of the file to be loaded.

# Keywords
- `element_type`: Element type (default is `Float64`)
"""
function load_shape(filename; element_type=Float64)
    ELTYPE = element_type

    file_extension = splitext(filename)[end]

    if file_extension == ".asc"
        shape = load_ascii(filename; ELTYPE, skipstart=1)
    elseif file_extension == ".stl"
        shape = load(query(filename); ELTYPE)
    else
        throw(ArgumentError("Only `.stl` and `.asc` files are supported (yet)."))
    end

    return shape
end

function load_ascii(filename; ELTYPE=Float64, skipstart=1)

    # Read the data from the ASCII file in as a matrix of coordinates.
    # Ignore the first `skipstart` lines of the file (e.g. headers).
    points = DelimitedFiles.readdlm(filename, ' ', ELTYPE, '\n'; skipstart)[:, 1:2]

    return Polygon(copy(points'))
end

# FileIO.jl docs:
# https://juliaio.github.io/FileIO.jl/stable/implementing/#All-at-once-I/O:-implementing-load-and-save
function load(fn::File{format"STL_BINARY"}; element_types...)
    open(fn) do s
        FileIO.skipmagic(s) # skip over the magic bytes
        load(s; element_types...)
    end
end

function load(fs::Stream{format"STL_BINARY"}; ELTYPE=Float64)
    # Binary STL
    # https://en.wikipedia.org/wiki/STL_%28file_format%29#Binary_STL
    io = stream(fs)
    read(io, 80) # Throw out 80 bytes header
    n_faces = read(io, UInt32)

    face_vertices = fill(ntuple(_ -> fill(zero(ELTYPE), SVector{3}), 3), n_faces)
    vertices = fill(fill(zero(ELTYPE), SVector{3}), 3n_faces)
    normals = fill(fill(zero(ELTYPE), SVector{3}), n_faces)

    i = 0
    while !eof(io)
        normals[i + 1] = SVector{3, ELTYPE}(read(io, Float32), read(io, Float32),
                                            read(io, Float32))

        v1 = SVector{3, ELTYPE}(read(io, Float32), read(io, Float32), read(io, Float32))
        v2 = SVector{3, ELTYPE}(read(io, Float32), read(io, Float32), read(io, Float32))
        v3 = SVector{3, ELTYPE}(read(io, Float32), read(io, Float32), read(io, Float32))

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

    return TriangleMesh(face_vertices, normals, vertices)
end

function trixi2vtk(shape::Polygon; output_directory="out", prefix="",
                   filename="points", custom_quantities...)
    vertex_normals = stack([shape.vertex_normals[edge][1] for edge in eachface(shape)])

    return trixi2vtk(stack(shape.vertices); output_directory, filename, prefix,
                     vertex_normals=vertex_normals, custom_quantities...)
end

function trixi2vtk(shape::TriangleMesh; output_directory="out", prefix="",
                   filename="points", custom_quantities...)
    vertex_normals = stack([shape.vertex_normals[face]
                            for face in eachindex(shape.vertices)])

    return trixi2vtk(stack(shape.vertices); output_directory, filename, prefix,
                     vertex_normals=vertex_normals, custom_quantities...)
end