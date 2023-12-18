using Documenter, TrixiParticles

# Get TrixiParticles.jl root directory
trixiparticles_root_dir = dirname(@__DIR__)

# Copy files to not need to synchronize them manually
function copy_file(filename, replaces...)
    source_path = joinpath(trixiparticles_root_dir, filename)

    if !isfile(source_path)
        error("File $filename not found. Ensure that you are running make.jl from the TrixiParticles.jl root directory.")
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

    write(joinpath(@__DIR__, "src", lowercase(filename)), content)
end

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

makedocs(sitename="TrixiParticles.jl",
         # Explicitly specify documentation structure
         pages=[
             "Home" => "index.md",
             "News" => "news.md",
             "Installation" => "install.md",
             "Get Started" => "get_started.md",
             "Tutorial" => "tutorial.md",
             "Examples" => "examples.md",
             "Components" => [
                 "General" => [
                     "Semidiscretization" => joinpath("general", "semidiscretization.md"),
                     "Initial Condition and Setups" => joinpath("general",
                                                                "initial_condition.md"),
                     "Density Calculators" => joinpath("general", "density_calculators.md"),
                     "Smoothing Kernels" => joinpath("general", "smoothing_kernels.md"),
                     "Neighborhood Search" => joinpath("general", "neighborhood_search.md"),
                     "Util" => joinpath("general", "util.md"),
                 ],
                 "Systems" => [
                     "Weakly Compressible SPH (Fluid)" => joinpath("systems",
                                                                   "weakly_compressible_sph.md"),
                     "Entropically Damped Artificial Compressibility for SPH (Fluid)" => joinpath("systems",
                                                                                                  "entropically_damped_sph.md"),
                     "Total Lagrangian SPH (Elastic Solid)" => joinpath("systems",
                                                                        "total_lagrangian_sph.md"),
                     "Boundary" => joinpath("systems", "boundary.md"),
                 ],
                 "Time integration" => "time_integration.md",
                 "Callbacks" => "callbacks.md",
             ],
             "Authors" => "authors.md",
             "Contributing" => "contributing.md",
             "Code of Conduct" => "code_of_conduct.md",
             "License" => "license.md",
         ])
