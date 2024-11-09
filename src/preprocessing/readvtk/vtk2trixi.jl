using ReadVTK

"""
    Convert data from VTK-file to InitialCondition

    # FluidSystem data only
"""

function vtk2trixi(filename)
    vtk_file = VTKFile(filename)

    # Retrieve particle coordinates
    coordinates = get_points(vtk_file)

    # Retrieve point data fields (e.g., pressure, velocity, ...)
    vtk_point_data = get_point_data(vtk_file)

    # create field data arrays
    density = get_data(vtk_point_data["density"])

    pressure = get_data(vtk_point_data["pressure"])

    velocity = get_data(vtk_point_data["velocity"])
    if size(velocity, 1) == 2
        # If velocity is 2D, add 0.0 for the z component
        velocity = vcat(velocity, zeros(1, size(velocity, 2)))
    end

    mass = ones(size(coordinates, 2))

    return InitialCondition(; coordinates, velocity, mass, density, pressure)
end

# TODO: edit the mass array --> InitialCondition needs a mass or a particle_spacing
# TODO: example file in folder examples/readvtk
# TODO: make it work with 2D velocity --> In ParaView the velocity vecor is 3D after trixi2vtk. It should be 2D if the initial data is 2D.