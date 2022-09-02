function write2vtk(u, semi::SPHSemidiscretization{2}, timestep; show_boundaries=true)
    @unpack boundary_conditions, cache = semi

    mkpath("out")
    filename = timestep === nothing ? "out/data" : "out/data_$timestep"

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

function write2vtk(u, semi::SPHSemidiscretization{3}, timestep; show_boundaries=true)
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


function write2vtk(sol, semi; show_boundaries=true)
    for i in eachindex(sol)
        write2vtk(sol[i], semi, i, show_boundaries=show_boundaries)
    end
end


function write2vtk(saving_callback)
    @unpack saved_values = saving_callback
    @unpack saveval = saved_values

    for timestep in eachindex(saveval)
        solution = saveval[timestep]

        mkpath("out")
        filename = timestep === nothing ? "out/data" : "out/data_$timestep"

        points = solution["coordinates"]
        cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

        vtk_grid(filename, points, cells) do vtk
            for (key, value) in solution
                if key != "coordinates"
                    vtk[key] = value
                end
            end
        end
    end
end
