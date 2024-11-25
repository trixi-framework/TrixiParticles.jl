# TODO: Write documentation
# TODO: Write tests
"""
    Convert data from VTK-file to InitialCondition 
"""
function vtk2trixi(file; iter=0, input_directory="out", prefix="")
    vtk_file = VTKFile(file)

    # Retrieve point data fields (e.g., pressure, velocity, ...)
    point_data = get_point_data(vtk_file)

    # Retrieve fields
    pressure = get_data(point_data["pressure"])
    # mass = get_data(point_data["mass"]) # TODO: Read out mass correctly as soon as mass is written out in vtu-file by Trixi

    density = try
        get_data(point_data["density"])
    catch
        get_data(point_data["hydrodynamic_density"])
    end

    velocity = try
        get_data(point_data["velocity"])
    catch
        try
            get_data(point_data["wall_velocity"])
        catch
            get_data(point_data["initial_velocity"])
        end
    end

    # Coordinates are stored as a 3xN array
    # Adjust the dimension of coordinates to match the velocity field
    coordinates = get_points(vtk_file)[axes(velocity, 1), :]

    return InitialCondition(
                            ; coordinates,
                            velocity,
                            mass=zeros(size(coordinates, 2)),
                            density,
                            pressure)
end
