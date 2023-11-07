using Documenter, TrixiParticles

# Get TrixiParticles.jl root directory
trixiparticles_root_dir = dirname(@__DIR__)

# Copy list of authors to not need to synchronize it manually
authors_text = read(joinpath(trixiparticles_root_dir, "AUTHORS.md"), String)
authors_text = replace(authors_text,
                       "in the [LICENSE.md](LICENSE.md) file" => "under [License](@ref)")
write(joinpath(@__DIR__, "src", "authors.md"), authors_text)

makedocs(sitename="TrixiParticles.jl",
         # Explicitly specify documentation structure
         pages=[
             "Home" => "index.md",
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
                     "Weakly Compressible SPH (Fluid)" => joinpath("systems", "weakly_compressible_sph.md"),
                     "Entropically Damped Artificial Compressibility for SPH (Fluid)" => joinpath("systems", "entropically_damped_sph.md"),
                     "Total Lagrangian SPH (Elastic Solid)" => joinpath("systems", "total_lagrangian_sph.md"),
                     "Boundary" => joinpath("systems", "boundary.md"),
                 ],
                 "Time integration" => "time_integration.md",
                 "Callbacks" => "callbacks.md",
             ],
             "Authors" => "authors.md",
             "Contributing" => "contributing.md",
             #  "Code of Conduct" => "code_of_conduct.md",
             "License" => "license.md",
         ])
