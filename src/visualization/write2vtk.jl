function pixie2vtk(solution_or_boundaries...; output_directory="out", prefix="")
    for solution_or_boundary in solution_or_boundaries
        pixie2vtk(solution_or_boundary, output_directory=output_directory, prefix=prefix)
    end
end

function pixie2vtk(saved_values::SavedValues; output_directory="out", prefix="")
    @unpack saveval = saved_values

    for timestep in eachindex(saveval)
        solution = saveval[timestep]

        mkpath(output_directory)

        # For all containers
        for key in keys(solution)
            filename = prefix === "" ? "$output_directory/$key" : "$output_directory/$(prefix)_$key"
            filename *= timestep === nothing ? "" : "_$timestep"

            points = solution[key][:coordinates]
            cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

            vtk_grid(filename, points, cells) do vtk
                for (key, value) in solution[key]
                    if key != :coordinates
                        vtk[string(key)] = value
                    end
                end
            end
        end
    end
end

function pixie2vtk(container::BoundaryParticleContainer; output_directory="out", prefix="")
    @unpack initial_coordinates = container

    mkpath(output_directory)
    filename = prefix==="" ? "$output_directory/boundaries" : "$output_directory/$(prefix)_boundaries"

    points = initial_coordinates
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]

    vtk_grid(filename, points, cells) do vtk
        nothing
    end

    return nothing
end
