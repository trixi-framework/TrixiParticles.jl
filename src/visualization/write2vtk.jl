function write2vtk(saved_values; output_directory="out")
    @unpack saveval = saved_values

    for timestep in eachindex(saveval)
        solution = saveval[timestep]

        mkpath(output_directory)
        filename = timestep === nothing ? "$output_directory/data" : "$output_directory/data_$timestep"

        points = solution[:coordinates]
        cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

        vtk_grid(filename, points, cells) do vtk
            for (key, value) in solution
                if key != :coordinates
                    vtk[string(key)] = value
                end
            end
        end
    end
end
