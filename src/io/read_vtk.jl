"""
	vtk2trixi(file::String; custom_quantities...)

Load VTK file and convert data to a NamedTuple.

# Arguments
- `file`:                 Name of the VTK file to be loaded.
- `custom_quantities...`: Additional custom quantities to be loaded from the VTK file.
                          Each custom quantity must be explicitly listed in the
                          `custom_quantities` during the simulation to ensure it is
                          included in the VTK output and can be successfully loaded.
                          See [Custom Quantities](@ref custom_quantities) for details.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.

# Example
```jldoctest; output = false
# Create a rectangular shape
rectangular = RectangularShape(0.1, (10, 10), (0, 0), density=1.5, velocity=(1.0, -2.0),
                               pressure=1000.0)

# Write the `InitialCondition` with custom quantity to a vtk file
trixi2vtk(rectangular; filename="rectangular", output_directory="out",
          my_custom_quantity=3.0)

# Read the vtk file and convert the data to an `NamedTuple`
data = vtk2trixi(joinpath("out", "rectangular.vtu");
                 my_custom_quantity="my_custom_quantity")

# output
NamedTuple{data...}
"""
function vtk2trixi(file; custom_quantities...)
    vtk_file = ReadVTK.VTKFile(file)

    # Retrieve data fields (e.g., pressure, velocity, ...)
    point_data = ReadVTK.get_point_data(vtk_file)
    field_data = ReadVTK.get_field_data(vtk_file)

    results = Dict{Symbol, Any}()

    # Retrieve fields
    ndims = first(ReadVTK.get_data(field_data["ndims"]))
    coordinates = ReadVTK.get_points(vtk_file)[1:ndims, :]

    fields = [:velocity, :density, :pressure, :mass, :particle_spacing]
    for field in fields
        # Look for any key that contains the field name
        all_keys = keys(point_data)
        idx = findfirst(k -> occursin(string(field), k), all_keys)
        if idx !== nothing
            results[field] = ReadVTK.get_data(point_data[all_keys[idx]])
        else
            # Use zeros as default values when a field is missing
            results[field] = string(field) in ["mass"] ?
                             zeros(size(coordinates, 2)) : zero(coordinates)
            @info "No '$field' field found in VTK file. Will be set to zero."
        end
    end

    results[:particle_spacing] = allequal(results[:particle_spacing]) ?
                                 first(results[:particle_spacing]) :
                                 results[:particle_spacing]
    results[:coordinates] = coordinates
    results[:time] = "time" in keys(field_data) ?
                     first(ReadVTK.get_data(field_data["time"])) : 0.0

    for (key, quantity) in custom_quantities
        if quantity in keys(point_data)
            results[key] = ReadVTK.get_data(point_data[quantity])
        end
        if quantity in keys(field_data)
            results[key] = first(ReadVTK.get_data(field_data[quantity]))
        end
    end

    return NamedTuple(results)
end
