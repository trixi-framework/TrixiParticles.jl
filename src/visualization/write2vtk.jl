function write2vtk(u, container::ParticleContainer{ELTYPE, 2}, timestep) where ELTYPE
    mkpath("out")
    filename = timestep === nothing ? "out/data" : "out/data_$timestep"

    points = view(u, 1:2, :)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in eachparticle(container)]

    vtk_grid(filename, points, cells) do vtk
        vtk["v"] = view(u, 3:4, :)
    end
end

function write2vtk(u, container::ParticleContainer{ELTYPE, 3}, timestep) where ELTYPE
    mkpath("out")
    filename = timestep === nothing ? "out/data" : "out/data_$timestep"

    points = view(u, 1:3, :)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in eachparticle(container)]

    vtk_grid(filename, points, cells) do vtk
        vtk["v"] = view(u, 4:6, :)
    end
end


function write2vtk(sol, container)
    for i in eachindex(sol)
        write2vtk(sol[i], container, i)
    end
end
