using WriteVTK

output_directory = "out_vtk"
mkpath(output_directory)

file = joinpath(output_directory, "two_points")

# Define the points array in the expected format: 3 rows (x, y, z), 2 columns (for 2 points)
points = [1.0 3.0  # x-coordinates
          2.0 4.0  # y-coordinates
          0.0 0.0] # z-coordinates (both points on the z=0 plane)

# Create VTK_VERTEX cells for both points
cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (1,)),  # Cell for first point
    MeshCell(VTKCellTypes.VTK_VERTEX, (2,))]  # Cell for second point

# Initialize the VTK grid with the points and cells
vtk_file = vtk_grid(file, points, cells)

# Assign scalar data to the two points
vtk_file["scalar"] = [100.0, 200.0]  # Scalar values for the two points

# Assign vector data (e.g., velocity) to the two points
vtk_file["velocity"] = [1.0, 2.0, 3.0,   # Velocity for first point
    4.0, 5.0, 6.0]    # Velocity for second point

# Save the VTK file
vtk_save(vtk_file)
