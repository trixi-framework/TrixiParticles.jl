# This function creates a unique filename by appending a number to the base name if needed.
function get_unique_filename(base_name, extension)
    # Ensure the extension starts with a dot
    extension = startswith(extension, ".") ? extension : "." * extension

    filename = base_name * extension
    counter = 1

    while isfile(filename)
        filename = base_name * string(counter) * extension
        counter += 1
    end

    return filename
end

function get_latest_unique_filename(dir_path, base_name, extension)
    files = readdir(dir_path)
    extension_regex = startswith(extension, ".") ? escape_string(extension) : "\\." * escape_string(extension)
    regex_pattern = "^" * escape_string(base_name) * "\\d*" * extension_regex * "\$"

    regex = Regex(regex_pattern)

    # Filter files based on the constructed regex
    matching_files = filter(x -> occursin(regex, x), files)

    if isempty(matching_files)
        println("No matching files found.")
        return ""
    end

    recent_file = sort(matching_files, by=x -> stat(joinpath(dir_path, x)).ctime, rev=true)[1]
    file_path = joinpath(dir_path, recent_file)
    return file_path
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
        return string(chomp(git_hash))
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
