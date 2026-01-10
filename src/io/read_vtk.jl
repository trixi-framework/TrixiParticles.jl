"""
	vtk2trixi(file::String; element_type=nothing, coordinates_eltype=nothing,
              create_initial_condition=true, custom_quantities...)

Read a VTK file and return a `NamedTuple` with keys
`:coordinates`, `:velocity`, `:density`, `:pressure`, `:particle_spacing`, `:time`,
plus any requested custom quantities.
Missing fields are zero-filled; `:particle_spacing` is scalar if constant, otherwise per-particle.

# Arguments
- `file`: Name of the VTK file to be loaded.

# Keywords
- `element_type`: Element type for particle fields. By default, the type
                  stored in the VTK file is used.
                  Otherwise, data is converted to the specified type.
- `coordinates_eltype`: Element type for particle coordinates. By default, the type
                        stored in the VTK file is used.
                        Otherwise, data is converted to the specified type.
- `create_initial_condition`: If `true`, an `InitialCondition` object is created
                              and included in the returned `NamedTuple` under
                              the key `:initial_condition`. Default is `true`.
- `custom_quantities...`: Keyword arguments to load additional quantities from the VTK file.
                          Each keyword becomes a key in the returned `NamedTuple`, with its
                          string value specifying the VTK field name to read.
                          Example: `my_data="field_name"` loads VTK field `"field_name"`
                          as `:my_data` in the result.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.

# Example
```jldoctest; output = false, filter = r"density = \\[.*\\]|pressure = \\[.*\\]|velocity = \\[.*\\]|coordinates = \\[.*\\]"
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
(particle_spacing = 0.1, density = [...], time = 0.0, pressure = [...], my_custom_quantity = 3.0, velocity = [...], coordinates = [...], initial_condition = InitialCondition{Float64, Float64}())
```
"""
function vtk2trixi(file; element_type=nothing, coordinates_eltype=nothing,
                   create_initial_condition=true, custom_quantities...)
    vtk_file = ReadVTK.VTKFile(file)

    # Retrieve data fields (e.g., pressure, velocity, ...)
    point_data = ReadVTK.get_point_data(vtk_file)
    field_data = ReadVTK.get_field_data(vtk_file)
    point_coords = ReadVTK.get_points(vtk_file)

    cELTYPE = isnothing(coordinates_eltype) ? eltype(point_coords) : coordinates_eltype
    ELTYPE = isnothing(element_type) ?
             eltype(first(ReadVTK.get_data(point_data["pressure"]))) : element_type

    results = Dict{Symbol, Any}()

    # Retrieve fields
    ndims = first(ReadVTK.get_data(field_data["ndims"]))
    coordinates = convert.(cELTYPE, point_coords[1:ndims, :])

    fields = [:velocity, :density, :pressure, :particle_spacing]
    for field in fields
        # Look for any key that contains the field name
        all_keys = keys(point_data)
        idx = findfirst(k -> occursin(string(field), k), all_keys)
        if idx !== nothing
            results[field] = convert.(ELTYPE, ReadVTK.get_data(point_data[all_keys[idx]]))
        else
            # Use zeros as default values when a field is missing
            results[field] = string(field) in ["velocity"] ?
                             zero(coordinates) : zeros(ELTYPE, size(coordinates, 2))
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

    results = NamedTuple(results)

    if create_initial_condition
        ic = InitialCondition(; coordinates=results.coordinates,
                              particle_spacing=results.particle_spacing,
                              velocity=results.velocity, density=results.density,
                              pressure=results.pressure)

        return merge(results, (initial_condition=ic,))
    else
        return results
    end
end
