function pixie2vtk(solution_or_boundaries...; output_directory="out")
    for solution_or_boundary in solution_or_boundaries
        pixie2vtk(solution_or_boundary, output_directory=output_directory)
    end
end

function pixie2vtk(saved_values::SavedValues; output_directory="out")
    @unpack saveval = saved_values

    for timestep in eachindex(saveval)
        solution = saveval[timestep]

        mkpath(output_directory)
        filename = timestep === nothing ? "$output_directory/solution" : "$output_directory/solution_$timestep"

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

function pixie2vtk(boundary_conditions::Union{BoundaryConditionMonaghanKajtar, BoundaryConditionFrozenMirrored}; output_directory="out")
    @unpack coordinates = boundary_conditions

    mkpath(output_directory)
    filename = "$output_directory/boundaries"

    points = coordinates
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(filename, points, cells) do vtk
        nothing
    end

    return nothing
end
