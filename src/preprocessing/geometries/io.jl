
function save(filename, mesh; faces=eachface(mesh))
    save(FileIO.File{FileIO.format"STL_BINARY"}(filename), mesh; faces)
end

function save(fn::FileIO.File{FileIO.format"STL_BINARY"},
              mesh::TriangleMesh; faces=eachface(mesh))
    open(fn, "w") do s
        save(s, mesh; faces)
    end
end

function save(f::FileIO.Stream{FileIO.format"STL_BINARY"},
              mesh::TriangleMesh; faces)
    io = FileIO.stream(f)
    points = mesh.face_vertices
    normals = mesh.face_normals

    # Implementation made according to https://en.wikipedia.org/wiki/STL_%28file_format%29#Binary_STL
    for i in 1:80 # Write empty header
        write(io, 0x00)
    end

    write(io, UInt32(length(faces))) # Write triangle count
    for i in faces
        n = SVector{3, Float32}(normals[i])
        triangle = points[i]

        for j in 1:3
            write(io, n[j])
        end

        for point in triangle, p in point
            write(io, Float32(p))
        end

        # After the 48 bytes of the normal and the vertices follows a 2-byte unsigned integer
        # that is the "attribute byte count" – in the standard format, this should be zero
        # because most software does not understand anything else.
        write(io, 0x0000) # 16 empty bits
    end
end
