function write2vtk(u, timestep)
    mkpath("out")
    filename = timestep === nothing ? "out/data" : "out/data_$timestep"

    points = view(u, 1:3, :)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:size(points, 2)]

    vtk_grid(filename, points, cells) do vtk
        vtk["v"] = view(u, 4:6, :)
    end
end


function write2vtk(sol)
    for i in eachindex(sol)
        write2vtk(sol[i], i)
    end
end
