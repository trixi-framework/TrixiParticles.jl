"""
    load_shape(filename; element_type=Float64)

Load file and return corresponding `Shape` type for [`ComplexShape`](@ref).
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

    # Read in the ASCII file as an Tuple containing the coordinates of the points and the
    # header.

    # Either `header=true` which returns a tuple `(data_cells, header_cells)`
    # or ignoring the corresponding number of lines from the input with `skipstart`
    points = readdlm(filename, ' ', ELTYPE, '\n'; skipstart)[:, 1:2]

    return Polygon(copy(points'))
end

# FileIO.jl docs:
# https://juliaio.github.io/FileIO.jl/stable/implementing/#All-at-once-I/O:-implementing-load-and-save
function load(fn::File{format}; element_types...) where {format}
    open(fn) do s
        skipmagic(s) # skip over the magic bytes
        load(s; element_types...)
    end
end

function load(fs::Stream{format"STL_BINARY"}; ELTYPE=Float64)
    # Binary STL
    # https://en.wikipedia.org/wiki/STL_%28file_format%29#Binary_STL
    io = stream(fs)
    read(io, 80) # throw out header
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

        skip(io, 2) # throwout 16bit attribute
        i += 1
    end

    union!(vertices)

    return TriangleMesh(face_vertices, normals, vertices)
end

# ==========================================================================================
# ==== The following functions are only used for debugging yet

function save(filename, mesh; faces=eachface(mesh))
    save(File{format"STL_BINARY"}(filename), mesh; faces)
end

function save(fn::File{format}, mesh::TriangleMesh; faces=eachface(mesh)) where {format}
    open(fn, "w") do s
        save(s, mesh; faces)
    end
end

function save(f::Stream{format"STL_BINARY"}, mesh::TriangleMesh; faces)
    io = stream(f)
    points = mesh.face_vertices
    normals = mesh.normals_face

    # Implementation made according to https://en.wikipedia.org/wiki/STL_%28file_format%29#Binary_STL
    for i in 1:80 # write empty header
        write(io, 0x00)
    end

    write(io, UInt32(length(faces))) # write triangle count
    for i in faces
        n = SVector{3, Float32}(normals[i])
        triangle = points[i]
        foreach(j -> write(io, n[j]), 1:3)
        for point in triangle
            foreach(p -> write(io, p), SVector{3, Float32}(point))
        end
        write(io, 0x0000) # write 16bit empty bit
    end
end

function trixi2vtk(shape::Shapes{2}; output_directory="out", prefix="",
                   filename="points", custom_quantities...)
    normals_vertex = stack([shape.normals_vertex[edge][1] for edge in eachface(shape)])

    return trixi2vtk(stack(shape.vertices); output_directory, filename, prefix,
                     normals_vertex=normals_vertex)
end

function trixi2vtk(shape::Shapes{3}; output_directory="out", prefix="",
                   filename="points", custom_quantities...)
    normals_vertex = stack([shape.normals_vertex[face]
                            for face in eachindex(shape.vertices)])

    return trixi2vtk(stack(shape.vertices); output_directory, filename, prefix,
                     normals_vertex=normals_vertex)
end
