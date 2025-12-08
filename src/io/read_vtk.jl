"""
	vtk2trixi(file::String)

Load VTK file and convert data to an [`InitialCondition`](@ref).

# Arguments
- `file`: Name of the VTK file to be loaded.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.

# Example
```jldoctest; output = false
# Create a rectangular shape
rectangular = RectangularShape(0.1, (10, 10), (0, 0), density=1.5, velocity=(1.0, -2.0),
                               pressure=1000.0)

# Write the `InitialCondition` to a vtk file
trixi2vtk(rectangular; filename="rectangular", output_directory="out")

# Read the vtk file and convert it to `InitialCondition`
ic = vtk2trixi(joinpath("out", "rectangular.vtu"))

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ InitialCondition                                                                                 │
│ ════════════════                                                                                 │
│ #dimensions: ……………………………………………… 2                                                                │
│ #particles: ………………………………………………… 100                                                              │
│ particle spacing: ………………………………… 0.1                                                              │
│ eltype: …………………………………………………………… Float64                                                          │
│ coordinate eltype: ……………………………… Float64                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
````
"""
function vtk2trixi(file)
    vtk_file = ReadVTK.VTKFile(file)

    # Retrieve data fields (e.g., pressure, velocity, ...)
    point_data = ReadVTK.get_point_data(vtk_file)
    field_data = ReadVTK.get_field_data(vtk_file)

    # Retrieve fields
    ndims = first(ReadVTK.get_data(field_data["ndims"]))
    coordinates = ReadVTK.get_points(vtk_file)[1:ndims, :]

    fields = ["velocity", "density", "pressure", "mass", "particle_spacing"]
    results = Dict{String, Array{Float64}}()

    for field in fields
        # Look for any key that contains the field name
        all_keys = keys(point_data)
        idx = findfirst(k -> occursin(field, k), all_keys)
        if idx !== nothing
            results[field] = ReadVTK.get_data(point_data[all_keys[idx]])
        else
            # Use zeros as default values when a field is missing
            results[field] = field in ["mass"] ?
                             zeros(size(coordinates, 2)) : zero(coordinates)
            @info "No '$field' field found in VTK file. Will be set to zero."
        end
    end

    particle_spacing = allequal(results["particle_spacing"]) ?
                       first(results["particle_spacing"]) :
                       results["particle_spacing"]

    return InitialCondition(; coordinates, particle_spacing=particle_spacing,
                            velocity=results["velocity"],
                            mass=results["mass"],
                            density=results["density"],
                            pressure=results["pressure"])
end
