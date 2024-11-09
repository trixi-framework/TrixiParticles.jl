using WriteVTK

# Function to create n points and their associated data
function create_vtk_file(n::Int, coords::Matrix{Float64}, scalars::Vector{Float64},
                         velocities::Matrix{Float64}, filename::String)
    # Ensure the coordinates matrix has the correct size (3 x n)
    @assert size(coords)==(3, n) "coords should be a 3 x n matrix, where n is the number of points"
    # Ensure scalars is a vector of length n
    @assert length(scalars)==n "scalars should be a vector of length n"
    # Ensure velocities is a matrix of size (3 x n)
    @assert size(velocities)==(3, n) "velocities should be a 3 x n matrix, where n is the number of points"

    output_directory = "out_vtk"
    mkpath(output_directory)

    file = joinpath(output_directory, filename)

    # Create VTK_VERTEX cells for all points
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:n]

    # Initialize the VTK grid with the points and cells
    vtk_file = vtk_grid(file, coords, cells)

    # Assign scalar data to the points
    vtk_file["scalar"] = scalars

    # Flatten velocities into a 1D array for VTK
    vtk_file["velocity"] = vec(velocities)

    # Save the VTK file
    vtk_save(vtk_file)
end

# Example usage with n points
n = 3  # Number of points

# Define coordinates for n points (3 rows for x, y, z; n columns for the points)
coords = [1.0 2.0 3.0;  # x-coordinates
          1.0 2.0 3.0;  # y-coordinates
          0.0 0.0 0.0]  # z-coordinates (all on the z=0 plane)

# Define scalar values for each point
scalars = [100.0, 200.0, 300.0]

# Define velocity vectors for each point (3 rows for vx, vy, vz; n columns for the points)
velocities = [10.0 0.0 5.0;  # x-components
              0.0 5.0 2.0;  # y-components
              0.0 0.0 0.0]  # z-components (all on the z=0 plane)

# Create the VTK file
create_vtk_file(n, coords, scalars, velocities, "n_points")
