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

    # Meta data only written out in simulations
    NDIMS = get_data(meta_data["ndims"])
    coordinates = get_points(vtk_file)[1:NDIMS[1], :]

    # Retrieve fields
    pressure = get_data(point_data["pressure"])

    density = try
        get_data(point_data["density"])
    catch
        get_data(point_data["hydrodynamic_density"])
    end

    velocity = try
        get_data(point_data["velocity"])
    catch
        try
            get_data(point_data["initial_velocity"])
        catch
            try
                get_data(point_data["wall_velocity"])
            catch
                # Case is only used for 'boundary_systems' in simulations where velocity is not written out
                zeros(size(coordinates))
            end
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
