"""
    vtk2trixi(file::String)

Convert data from VTK-file to InitialCondition 

# Arguments
- `file`: Name of the file to be loaded. 
"""
function vtk2trixi(file)
    vtk_file = VTKFile(file)

    # Retrieve data fields (e.g., pressure, velocity, ...)
    point_data = ReadVTK.get_point_data(vtk_file)
    field_data = get_field_data(vtk_file)

    # Retrieve fields
    ndims = get_data(field_data["ndims"])
    coordinates = get_points(vtk_file)[1:ndims[1], :]

    fields = ["velocity", "density", "pressure", "mass"]
    results = Dict{String, Array{Float64}}()

    for field in fields
        found = false
        for k in keys(point_data)
            if !isnothing(match(Regex("$field"), k))
                results[field] = get_data(point_data[k])
                found = true
                break
            end
        end
        if !found
            # Set fields to zero if not found
            if field in ["density", "pressure", "mass"]
                results[field] = zeros(size(coordinates, 2))
            else
                results[field] = zero(coordinates)
            end
            @info "No '$field' field found in VTK file. Will be set to zero."
        end
    end

    return InitialCondition(
                            ; coordinates,
                            velocity=results["velocity"],
                            mass=results["mass"],
                            density=results["density"],
                            pressure=results["pressure"])
end