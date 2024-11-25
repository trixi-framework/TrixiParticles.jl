# TODO: Write documentation
# TODO: Write tests
"""
    Convert data from VTK-file to InitialCondition 
"""
function vtk2trixi(file; iter=0, input_directory="out", prefix="")
    vtk_file = VTKFile(file)

    # Retrieve point data fields (e.g., pressure, velocity, ...)
    point_data = get_point_data(vtk_file)
    meta_data = get_field_data(vtk_file)
    # TODO: Shapes created directly with write2vtk do not have 'meta_data'. Add this feature

    NDIMS = get_data(meta_data["ndims"])
    coordinates = get_points(vtk_file)[1:NDIMS[1], :]

    # Retrieve fields 
    density = try
        get_data(point_data["density"])
    catch
        try
            get_data(point_data["hydrodynamic_density"])
        catch
            zeros(size(coordinates, 2))
        end
    end

    pressure = try
        get_data(point_data["pressure"])
    catch
        zeros(size(coordinates, 2))
    end

    velocity = try
        get_data(point_data["velocity"])
    catch
        try
            get_data(point_data["initial_velocity"])
        catch
            zeros(size(coordinates))
        end
    end

    # TODO: Read out mass correctly as soon as mass is written out in vtu-file by Trixi
    # mass = try
    #     get_data(point_data["mass"])
    # catch
    #     zeros(size(coordinates, 2))
    # end

    return InitialCondition(
                            ; coordinates,
                            velocity,
                            mass=ones(size(coordinates, 2)),
                            density,
                            pressure)
end
