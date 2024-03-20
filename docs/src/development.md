# [Development](@id development)


## Documentation
To generate the Documentation first instantiate the docs environment
```
cd /path/to/TrixiParticles.jl/
julia --project=./docs
julia > ] develop .
julia > ] instantiate
```
than we can generated the docs for example by using the following command

```
cd /path/to/TrixiParticles.jl/ && julia --project=./docs ./docs/make.jl && cd docs/build && python3 -m http.server
```
using your webbrowser one can than navigate to localhost:8000.