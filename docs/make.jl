using Documenter
using TrixiParticles
using TrixiBase
using PointNeighbors

# Get TrixiParticles.jl root directory
trixiparticles_root_dir = dirname(@__DIR__)

# Copy files to not need to synchronize them manually
function copy_file(filename, replaces...;
                   new_file=joinpath(@__DIR__, "src", lowercase(filename)))
    source_path = joinpath(trixiparticles_root_dir, filename)

    if !isfile(source_path)
        error("File $filename not found. Ensure that you provide a path relative to the TrixiParticles.jl root directory.")
        return
    end

    content = read(source_path, String)
    content = replace(content, replaces...)

    header = """
    ```@meta
    EditURL = "https://github.com/trixi-framework/TrixiParticles.jl/blob/main/$filename"
    ```
    """
    content = header * content

    write(new_file, content)
end

function replace_with_code(filename)
    if !isfile(filename)
        cwd = pwd()
        error("Markdown file not found: $filename in directory: $cwd")
        return
    end

    # Define a regex pattern to match the include markers
    pattern = r"!!include:([^\s!]+\.jl)!!"

    function replace_include(match_::SubString{String})
        # Extract the filename using regex
        m = match(pattern, match_)
        if m === nothing
            error("Invalid include format in: $match")
        end
        file_to_include = joinpath(trixiparticles_root_dir, m.captures[1])

        try
            # Check if the Julia file exists
            if !isfile(file_to_include)
                error("File to include not found: $(file_to_include)")
            end

            # Read the content of the file to include
            return read(file_to_include, String)
        catch e
            # In case of any error
            error("Unable to include file $(file_to_include): $(e)")
        end
    end

    # Replace all occurrences in the markdown content
    filename_noext, extension = splitext(filename)
    copy_file(filename, new_file="$(filename_noext)_replaced$extension",
              pattern => replace_include)
end

replace_with_code("docs/src/tutorials/tut_setup.md")
replace_with_code("docs/src/tutorials/tut_dam_break.md")
replace_with_code("docs/src/tutorials/tut_beam.md")
replace_with_code("docs/src/tutorials/tut_falling.md")

copy_file("AUTHORS.md",
          "in the [LICENSE.md](LICENSE.md) file" => "under [License](@ref)")
copy_file("CONTRIBUTING.md",
          "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)",
          "[LICENSE.md](LICENSE.md)" => "[License](@ref)")
# Add section `# License` and add `>` in each line to add a quote
copy_file("LICENSE.md",
          "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)",
          "\n" => "\n> ", r"^" => "# License\n\n> ")
# Add section `# Code of Conduct` and add `>` in each line to add a quote
copy_file("CODE_OF_CONDUCT.md",
          "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)",
          "\n" => "\n> ", r"^" => "# Code of Conduct\n\n> ")
copy_file("NEWS.md")

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(TrixiParticles, :DocTestSetup, :(using TrixiParticles); recursive=true)

makedocs(sitename="TrixiParticles.jl",
         # Run doctests and check docs for the following modules
         modules=[TrixiParticles],
         format=Documenter.HTML(),
         # Explicitly specify documentation structure
         pages=[
             "Home" => "index.md",
             "News" => "news.md",
             "Installation" => "install.md",
             "Getting started" => "getting_started.md",
             "Development" => "development.md",
             "Tutorial" => "tutorial.md",
             "Examples" => "examples.md",
             "Visualization" => "visualization.md",
             "Preprocessing" => "preprocessing.md",
             "Components" => [
                 "Overview" => "overview.md",
                 "General" => [
                     "Semidiscretization" => joinpath("general", "semidiscretization.md"),
                     "Initial Condition and Setups" => joinpath("general",
                                                                "initial_condition.md"),
                     "Interpolation" => joinpath("general", "interpolation.md"),
                     "Density Calculators" => joinpath("general", "density_calculators.md"),
                     "Smoothing Kernels" => joinpath("general", "smoothing_kernels.md"),
                     "Neighborhood Search" => joinpath("general", "neighborhood_search.md"),
                     "Util" => joinpath("general", "util.md"),
                 ],
                 "Systems" => [
                     "Discrete Element Method (Solid)" => joinpath("systems",
                                                                   "dem.md"),
                     "Weakly Compressible SPH (Fluid)" => joinpath("systems",
                                                                   "weakly_compressible_sph.md"),
                     "Entropically Damped Artificial Compressibility for SPH (Fluid)" => joinpath("systems",
                                                                                                  "entropically_damped_sph.md"),
                     "Total Lagrangian SPH (Elastic Structure)" => joinpath("systems",
                                                                            "total_lagrangian_sph.md"),
                     "Boundary" => joinpath("systems", "boundary.md"),
                 ],
                 "Time Integration" => "time_integration.md",
                 "Callbacks" => "callbacks.md",
                 "TrixiBase.jl API Reference" => "reference-trixibase.md",
                 "PointNeighbors.jl API Reference" => "reference-pointneighbors.md",
             ],
             "Authors" => "authors.md",
             "Contributing" => "contributing.md",
             "Code of Conduct" => "code_of_conduct.md",
             "License" => "license.md",
         ])

deploydocs(repo="github.com/trixi-framework/TrixiParticles.jl",
           devbranch="main", push_preview=true)
