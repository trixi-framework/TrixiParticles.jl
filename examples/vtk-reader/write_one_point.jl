using WriteVTK

output_directory = "out_vtk"
mkpath(output_directory)

file = joinpath(output_directory, "one_point")

# Define the points array in the expected format: 3 rows (x, y, z), 1 column (for 1 point)
points = [1.0  # x-coordinate
          2.0  # y-coordinate
          0.0] # z-coordinate

# Create a VTK_VERTEX cell for the single point
cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (1,))]  # Cell for the single point

# Initialize the VTK grid with the point and cell
vtk_file = vtk_grid(file, points, cells)

# Assign scalar data to the point
vtk_file["scalar"] = [100.0]  # Scalar value for the point

# Assign vector data (e.g., velocity) to the point
vtk_file["velocity"] = [10.0, 0.0, 0.0]  # Velocity for the point

# Save the VTK file
vtk_save(vtk_file)

# Surprisingly, the code works not for one point, but for >=2 points.