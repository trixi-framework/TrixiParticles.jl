function write2vtk(u, semi::SPHSemidiscretization{2}, timestep; show_boundaries=true)
    @unpack boundaries, cache = semi

    mkpath("out")
    filename = timestep === nothing ? "out/data" : "out/data_$timestep"

    points = view(u, 1:2, :)
    if show_boundaries && size(boundaries.coordinates, 1) != 0
        points = hcat(points, boundaries.coordinates)
    end
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(filename, points, cells) do vtk
        vtk["v"] = view(u, 3:4, :)
    end
end

function write2vtk(u, semi::SPHSemidiscretization{3}, timestep; show_boundaries=true)
    @unpack boundaries = semi

    mkpath("out")
    filename = timestep === nothing ? "out/data" : "out/data_$timestep"

    points = view(u, 1:3, :)
    if show_boundaries && size(boundaries.coordinates, 1) != 0
        points = hcat(points, boundaries.coordinates)
    end
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(filename, points, cells) do vtk
        vtk["v"] = view(u, 4:6, :)
    end
end


function write2vtk(sol, semi; show_boundaries=true)
    for i in eachindex(sol)
        write2vtk(sol[i], semi, i, show_boundaries=show_boundaries)
    end
end
