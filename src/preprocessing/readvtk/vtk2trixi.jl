"""
    Convert data from VTK-file to InitialCondition 
"""
# TODO: write documentation
# TODO: write tests

function vtk2trixi(file)
    vtk_file = VTKFile(file)

    # retrieve point data fields (e.g., pressure, velocity, ...)
    vtk_point_data = get_point_data(vtk_file)

    # create field data arrays
    density = get_data(vtk_point_data["density"])

    pressure = get_data(vtk_point_data["pressure"])

    velocity = get_data(vtk_point_data["velocity"])

    # retrieve particle coordinates
    # point coordinates are stored in a 3xN matrix, but velocity can be stored either as 3xN or 2xN matrix
    coordinates = get_points(vtk_file)[axes(velocity, 1), :]

    mass = ones(size(coordinates, 2))
    # TODO: read out mass as soon as mass is written out in vtu-file by Trixi

    # TODO: get custom_quantities from vtk file

    # TODO: include the cases, that flieds like velocity are not stored in the vtk file

    return InitialCondition(; coordinates, velocity, mass, density, pressure)
end
