# [Development](@id development)


## Preview of the documentation

To generate the Documentation, first instantiate the `docs` environment
by executing the following command from the TrixiParticles.jl root directory:
```bash
julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"
```
This command only has to be run once. After that, maintain the `docs` environment
as described under [Installation](@ref Common issues).

With an instantiated `docs` environment, generate the docs with the following command (again from the TrixiParticles.jl root directory):
```bash
julia --project=docs --color=yes docs/make.jl
```
You can then open the generated files in `docs/build` with your webbrowser.
Alternatively, run
```bash
python3 -m http.server -d docs/build
```
and open `localhost:8000` in your webbrowser.