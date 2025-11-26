using Documenter, DocumenterCitations, DocumenterMermaid
using TrixiParticles
using TrixiParticles.TrixiBase
using TrixiParticles.PointNeighbors
using Asciicast: Asciicast
using Literate: Literate

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

    # Use `replace` to make sure the path uses forward slashes for URLs
    filename_url = replace(filename, "\\" => "/")

    header = """
    ```@meta
    EditURL = "https://github.com/trixi-framework/TrixiParticles.jl/blob/main/$filename_url"
    ```
    """
    content = header * content

    write(new_file, content)
end

Literate.markdown(joinpath("docs", "literate", "src", "tut_setup.jl"),
                  joinpath("docs", "src", "tutorials"))
Literate.markdown(joinpath("docs", "literate", "src", "tut_custom_kernel.jl"),
                  joinpath("docs", "src", "tutorials"))
Literate.markdown(joinpath("docs", "literate", "src", "tut_packing.jl"),
                  joinpath("docs", "src", "tutorials"))

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

# Define environment variables to create plots without warnings
# https://discourse.julialang.org/t/test-plots-on-travis-gks-cant-open-display/9465/2
ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(sitename="TrixiParticles.jl",
         plugins=[bib],
         # Run doctests and check docs for the following modules
         modules=[TrixiParticles, TrixiBase],
         format=Documenter.HTML(; assets=Asciicast.assets()),
         # Explicitly specify documentation structure
         pages=[
             "Home" => "index.md",
             "News" => "news.md",
             "Installation" => "install.md",
             "Getting started" => "getting_started.md",
             "Development" => "development.md",
             "Tutorials" => [
                 "Overview" => "tutorial.md",
                 "General" => [
                     "Setting up your simulation from scratch" => joinpath("tutorials",
                                                                           "tut_setup.md"),
                     "Modifying or extending components of TrixiParticles.jl within a simulation file" => joinpath("tutorials",
                                                                                                                   "tut_custom_kernel.md")
                 ],
                 "Preprocessing" => [
                     "Particle packing tutorial" => joinpath("tutorials",
                                                             "tut_packing.md")
                 ]
             ],
             "Examples" => "examples.md",
             "Visualization" => "visualization.md",
             "Preprocessing" => [
                 "Sampling of Geometries" => joinpath("preprocessing", "preprocessing.md")
             ],
             "GPU Support" => "gpu.md",
             "API Reference" => [
                 "Overview" => "overview.md",
                 "General" => [
                     "Semidiscretization" => joinpath("general", "semidiscretization.md"),
                     "Initial Condition and Setups" => joinpath("general",
                                                                "initial_condition.md"),
                     "Interpolation" => joinpath("general", "interpolation.md"),
                     "Density Calculators" => joinpath("general", "density_calculators.md"),
                     "Smoothing Kernels" => joinpath("general", "smoothing_kernels.md"),
                     "Neighborhood Search" => joinpath("general", "neighborhood_search.md"),
                     "Util" => joinpath("general", "util.md")
                 ],
                 "Systems" => [
                     "Fluid Models" => [
                         "Overview" => joinpath("systems", "fluid.md"),
                         "Weakly Compressible SPH (Fluid)" => joinpath("systems",
                                                                       "weakly_compressible_sph.md"),
                         "Entropically Damped Artificial Compressibility for SPH (Fluid)" => joinpath("systems",
                                                                                                      "entropically_damped_sph.md"),
                         "Implicit Incompressible SPH (Fluid)" => joinpath("systems",
                                                                           "implicit_incompressible_sph.md")
                     ],
                     "Discrete Element Method (Solid)" => joinpath("systems", "dem.md"),
                     "Total Lagrangian SPH (Elastic Structure)" => joinpath("systems",
                                                                            "total_lagrangian_sph.md"),
                     "Boundary" => joinpath("systems", "boundary.md")
                 ],
                 "Time Integration" => "time_integration.md",
                 "Callbacks" => "callbacks.md",
                 "TrixiBase.jl API Reference" => "reference-trixibase.md",
                 "PointNeighbors.jl API Reference" => "reference-pointneighbors.md"
             ],
             "Authors" => "authors.md",
             "Contributing" => "contributing.md",
             "Code of Conduct" => "code_of_conduct.md",
             "License" => "license.md",
             "References" => "references.md"
         ])

deploydocs(repo="github.com/trixi-framework/TrixiParticles.jl",
           devbranch="main", push_preview=true)
