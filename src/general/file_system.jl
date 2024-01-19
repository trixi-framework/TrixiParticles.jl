# This function creates a unique filename by appending a number to the base name if needed.
function get_unique_filename(base_name, extension)
    filename = base_name * extension
    counter = 1

    while isfile(filename)
        filename = base_name * string(counter) * extension
        counter += 1
    end

    return filename
end

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
    cmd = `git rev-parse HEAD`
    out = IOBuffer()
    err = IOBuffer()

    # Run the command and wait for it to complete
    process = run(pipeline(cmd, stdout=out, stderr=err), wait=true)

    # Rewind the IOBuffers to read from the start
    seekstart(out)
    seekstart(err)

    git_hash = read(out, String)
    git_error = read(err, String)

    # Check the exit code of the process
    if process.exitcode == 0
        return chomp(git_hash)
    else
        if occursin("not a git repository", git_error)
            return "Not a Git repository"
        else
            return "Git is not installed or not accessible"
        end
    end
end


function get_julia_version()
    return string(VERSION)
end
