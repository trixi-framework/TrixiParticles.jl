"""
	vtk2trixi(file::String; element_type=:default, coordinates_eltype=Float64,
              custom_quantities...)

Load VTK file and convert data to a `NamedTuple`.

# Arguments
- `file`:                 Name of the VTK file to be loaded.
- `custom_quantities...`: Additional custom quantities to be loaded from the VTK file.
                          Each custom quantity must be explicitly listed in the
                          `custom_quantities` during the simulation to ensure it is
                          included in the VTK output and can be successfully loaded.
                          See [Custom Quantities](@ref custom_quantities) for details.

# Keywords
- `element_type`: Element type for particle fields (`:default` keeps the type
    stored in the VTK file, otherwise converted to the given type).
- `coordinates_eltype`: Element type for particle coordinates (defaults to `Float64`).

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.

# Example
```jldoctest; output = false, filter = r"density = \\[.*\\]|pressure = \\[.*\\]|mass = \\[.*\\]|velocity = \\[.*\\]|coordinates = \\[.*\\]"
# Create a rectangular shape
rectangular = RectangularShape(0.1, (10, 10), (0, 0), density=1.5, velocity=(1.0, -2.0),
                               pressure=1000.0)

# Write the `InitialCondition` with custom quantity to a VTK file
trixi2vtk(rectangular; filename="rectangular", output_directory="out",
          my_custom_quantity=3.0)

# Read the VTK file and convert the data to a `NamedTuple`
data = vtk2trixi(joinpath("out", "rectangular.vtu");
                 my_custom_quantity="my_custom_quantity")

# output
(particle_spacing = 0.1, density = [...], time = 0.0, pressure = [...], mass = [...], my_custom_quantity = 3.0, velocity = [...], coordinates = [...])
```
"""
function vtk2trixi(file; element_type=:default, coordinates_eltype=Float64,
                   custom_quantities...)
    vtk_file = ReadVTK.VTKFile(file)

    # Retrieve data fields (e.g., pressure, velocity, ...)
    point_data = ReadVTK.get_point_data(vtk_file)
    field_data = ReadVTK.get_field_data(vtk_file)
    point_coords = ReadVTK.get_points(vtk_file)

    cELTYPE = coordinates_eltype
    ELTYPE = element_type === :default ? eltype(point_coords) : element_type

    results = Dict{Symbol, Any}()

    # Retrieve fields
    ndims = first(ReadVTK.get_data(field_data["ndims"]))
    coordinates = convert.(cELTYPE, point_coords[1:ndims, :])

    fields = [:velocity, :density, :pressure, :mass, :particle_spacing]
    for field in fields
        # Look for any key that contains the field name
        all_keys = keys(point_data)
        idx = findfirst(k -> occursin(string(field), k), all_keys)
        if idx !== nothing
            results[field] = convert.(ELTYPE, ReadVTK.get_data(point_data[all_keys[idx]]))
        else
            # Use zeros as default values when a field is missing
            results[field] = string(field) in ["mass"] ?
                             zeros(ELTYPE, size(coordinates, 2)) : zero(coordinates)
            @info "No '$field' field found in VTK file. Will be set to zero."
        end
    end

    results[:particle_spacing] = allequal(results[:particle_spacing]) ?
                                 first(results[:particle_spacing]) :
                                 results[:particle_spacing]
    results[:coordinates] = coordinates
    results[:time] = "time" in keys(field_data) ?
                     first(ReadVTK.get_data(field_data["time"])) : zero(ELTYPE)

    for (key, quantity_) in custom_quantities
        quantity = string(quantity_)
        if quantity in keys(point_data)
            results[key] = ReadVTK.get_data(point_data[quantity])
        elseif quantity in keys(field_data)
            results[key] = first(ReadVTK.get_data(field_data[quantity]))
        else
            throw(ArgumentError("Custom quantity '$quantity' not found in VTK file. " *
                                "Make sure it was included during the simulation."))
        end
    end

    return NamedTuple(results)
end
