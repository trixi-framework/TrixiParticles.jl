function write2vtk(u, semi::WCSPHSemidiscretization{2}, timestep, path_name; show_boundaries=true)
    @unpack boundary_conditions, cache = semi
    path_out = joinpath(path_name)
    isdir(path_out) || mkpath(path_out)

    if length(splitpath(path_out))>1
        filename = timestep === nothing ? path_out*"/$(path_name[2])_data" : path_out*"/$(path_name[2])_data_$timestep"
    else
        filename = timestep === nothing ? path_out*"/data" : path_out*"/data_$timestep"
    end

    points = view(u, 1:2, :)
    if show_boundaries
        points = hcat(points, (boundary.coordinates for boundary in boundary_conditions)...)
    end
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(filename, points, cells) do vtk
        vtk["v"] = hcat(view(u, 3:4, :), zeros(2, size(points, 2) - size(u, 2)))
        if size(u, 1) >= 5
            vtk["density"] = vcat(view(u, 5, :), zeros(size(points, 2) - size(u, 2)))
            vtk["index"] = vcat(1:size(u, 2), zeros(size(points, 2) - size(u, 2)))
        end
    end
end
function write2vtk(u, semi::EISPHSemidiscretization{2}, timestep, path_name; show_boundaries=true)
    @unpack boundary_conditions, cache = semi
    path_out = joinpath(path_name)
    isdir(path_out) || mkpath(path_out)

    if length(splitpath(path_out))>1
        filename = timestep === nothing ? path_out*"/$(path_name[2])_data" : path_out*"/$(path_name[2])_data_$timestep"
    else
        filename = timestep === nothing ? path_out*"/data" : path_out*"/data_$timestep"
    end

    points = view(u, 1:2, :)
    if show_boundaries
        points = hcat(points, (boundary.coordinates for boundary in boundary_conditions)...)
    end
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(filename, points, cells) do vtk
        vtk["v"] = hcat(view(u, 3:4, :), zeros(2, size(points, 2) - size(u, 2)))
        if size(u, 1) >= 5
            vtk["density"] = vcat(view(u, 9, :), zeros(size(points, 2) - size(u, 2)))
            vtk["index"] = vcat(1:size(u, 2), zeros(size(points, 2) - size(u, 2)))
        end
    end
end

function write2vtk(u, semi::WCSPHSemidiscretization{3}, timestep; show_boundaries=true)
    @unpack boundary_conditions, cache = semi

    mkpath("out")
    filename = timestep === nothing ? "out/data" : "out/data_$timestep"

    points = view(u, 1:3, :)
    if show_boundaries
        points = hcat(points, (boundary.coordinates for boundary in boundary_conditions)...)
    end
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(filename, points, cells) do vtk
        vtk["v"] = view(u, 4:6, :)
        if size(u, 1) >= 7
            vtk["density"] = vcat(view(u, 7, :), zeros(size(points, 2) - size(u, 2)))
            vtk["index"] = vcat(1:size(u, 2), zeros(size(points, 2) - size(u, 2)))
        end
    end
end


function write2vtk(sol, semi; show_boundaries=true, path_out=[""])
    pushfirst!(path_out, "out")
    for i in eachindex(sol)
        write2vtk(sol[i], semi, i, path_out,show_boundaries=show_boundaries)
    end
end
