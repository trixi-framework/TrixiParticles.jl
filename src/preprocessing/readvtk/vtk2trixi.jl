"""
    Convert data from VTK-file to InitialCondition 
"""
# TODO: Write documentation
# TODO: Write tests

function vtk2trixi(file, system_type)
    # Validate the system type
    if !(system_type in ["fluid", "boundary"])
        throw(ArgumentError("Invalid system_type argument. Must be 'fluid' or 'boundary'."))
    end

    vtk_file = VTKFile(file)

    # Retrieve point data fields (e.g., pressure, velocity, ...)
    point_data = get_point_data(vtk_file)

    coordinates = get_points(vtk_file)
    # Check for 2D or 3D coordinates
    if all(coordinates[3, :] .== 0)
        # If the third row is all zeros, reduce to 2xN
        coordinates = coordinates[1:2, :]
    end

    if system_type == "fluid"

        # Required fields
        required_fields = ["density", "pressure", "velocity"] #, "mass"]
        missing_fields = [field for field in required_fields if field ∉ keys(point_data)]

        # Throw an error if any required field is missing
        if !isempty(missing_fields)
            throw(ArgumentError("The following required fields are missing in the VTK file: $(missing_fields)"))
        end

        # Retrieve required field
        density = get_data(point_data["density"])
        pressure = get_data(point_data["pressure"])
        velocity = get_data(point_data["velocity"])
        # TODO: Read out mass correctly as soon as mass is written out in vtu-file by Trixi

        return InitialCondition(; coordinates, velocity, mass=ones(size(coordinates, 2)),
                                density,
                                pressure)

    else
        # Required fields
        required_fields = ["hydrodynamic_density", "pressure"] #, "mass"]
        missing_fields = [field for field in required_fields if field ∉ keys(point_data)]

        # Throw an error if any required field is missing
        if !isempty(missing_fields)
            throw(ArgumentError("The following required fields are missing in the VTK file: $(missing_fields)"))
        end

        # Retrieve required field
        hydrodynamic_density = get_data(point_data["hydrodynamic_density"])
        pressure = get_data(point_data["pressure"])
        # TODO: Read out mass correctly as soon as mass is written out in vtu-file by Trixi

        return InitialCondition(; coordinates, velocity=zeros(size(coordinates)),
                                mass=ones(size(coordinates, 2)),
                                density=hydrodynamic_density,
                                pressure)
    end
end
