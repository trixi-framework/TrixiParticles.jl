vtkname(system::FluidSystem) = "fluid"
vtkname(system::SolidSystem) = "solid"
vtkname(system::BoundarySystem) = "boundary"

function system_names(systems)
    # Add `_i` to each system name, where `i` is the index of the corresponding
    # system type.
    # `["fluid", "boundary", "boundary"]` becomes `["fluid_1", "boundary_1", "boundary_2"]`.
    cnames = systems .|> vtkname
    filenames = [string(cnames[i], "_", count(==(cnames[i]), cnames[1:i]))
                 for i in eachindex(cnames)]
    return filenames
end

function get_git_hash()
    try
        return string(readchomp(Cmd(`git describe --tags --always --first-parent --dirty`,
                                    dir=pkgdir(@__MODULE__))))
    catch e
        return "Git is not installed or not accessible"
    end
end
