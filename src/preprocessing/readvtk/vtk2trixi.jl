"""
    vtk2trixi(file)

Convert data from VTK-file to InitialCondition 
"""
function vtk2trixi(file)
    vtk_file = VTKFile(file)

    # Retrieve point data fields (e.g., pressure, velocity, ...)
    point_data = get_point_data(vtk_file)

    # Retrieve fields
    velocity = nothing
    coordinates = nothing
    try
        velocity = get_data(point_data[first(k
                                             for k in keys(point_data)
                                             if occursin(r"(?i)velocity", k))])

        # Coordinates are stored as a 3xN array
        # Adjust the dimension of coordinates to match the velocity field
        coordinates = get_points(vtk_file)[axes(velocity, 1), :]
    catch
        field_data = get_field_data(vtk_file)
        ndims = get_data(field_data["ndims"])

        # If no velocity field was found, adjust the coordinates to the dimensions
        coordinates = get_points(vtk_file)[1:ndims[1], :]

        # Make sure that the 'InitialCondition' has a velocity field
        @warn "No 'velocity' field found in VTK file. Velocity is set to zero."
        velocity = zeros(size(coordinates))
    end

    density = get_data(point_data[first(k
                                        for k in keys(point_data)
                                        if occursin(r"(?i)density", k))])

    pressure = get_data(point_data[first(k
                                         for k in keys(point_data)
                                         if occursin(r"(?i)pressure", k))])

    # mass = get_data(point_data[first(k
    #                                   for k in keys(point_data)
    #                                   if occursin(r"(?i)mass", k))])
    # TODO: Read out mass correctly as soon as mass is written out in vtu-file by Trixi

    return InitialCondition(
                            ; coordinates,
                            velocity,
                            mass=zeros(size(coordinates, 2)),
                            density,
                            pressure)
end