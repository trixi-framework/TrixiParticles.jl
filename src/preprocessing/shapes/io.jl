function load_shape(filename; scale_factor=nothing, ELTYPE=Float64, skipstart=1)
    file_extension = splitext(filename)[end]

    if file_extension == ".asc"
        shape = load_ascii(filename; scale_factor, ELTYPE, skipstart)
    elseif file_extension == ".stl"
        shape = load(query(filename); ELTYPE)
    else
        throw(ArgumentError("Only `.stl` and `.asc` files are supported (yet)."))
    end

    return shape
end

function load_ascii(filename; scale_factor=nothing, ELTYPE=Float64, skipstart=1)

    # Read in the ASCII file as an Tuple containing the coordinates of the points and the
    # header.

    # Either `header=true` which returns a tuple `(data_cells, header_cells)`
    # or ignoring the corresponding number of lines from the input with `skipstart`
    points = readdlm(filename, ' ', ELTYPE, '\n'; skipstart)[:, 1:2]
    if scale_factor isa ELTYPE
        points .*= scale_factor
    elseif scale_factor !== nothing
        throw(ArgumentError("`scale_factor` must be of type $ELTYPE"))
    end

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
