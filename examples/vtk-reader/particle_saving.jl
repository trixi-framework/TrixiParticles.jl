using ReadVTK

struct Particle
    index::Int64
    coordinates::Tuple{Float64, Float64, Float64}  # Assuming 3D coordinates
    density::Float64
    velocity::Tuple{Float64, Float64, Float64}  # Assuming 3D velocity
    pressure::Float64
end

vtk_file = VTKFile("out/fluid_1_1.vtu")
#vtk_file = VTKFile("out_vtk/rectangle_of_water.vtu")

# Retrieve particle coordinates
coords = get_points(vtk_file)

# Retrieve point data fields (e.g., pressure, velocity, ...)
vtk_point_data = get_point_data(vtk_file)

# Dynamically get all available field names from point_data
field_data = Dict()
for field_name in keys(vtk_point_data)
    field_data[field_name] = get_data(vtk_point_data[field_name])
end

# Create an array of Particle instances
particles = Vector{Particle}(undef, size(coords, 2))

for i in 1:size(coords, 2)
    # Retrieve required field "index"
    index = field_data["index"][i]

    # Coordinates
    coordinates = (coords[1, i], coords[2, i], coords[3, i])

    # Retrieve each required field directly, assuming all are present
    density = field_data["density"][i]
    pressure = field_data["pressure"][i]

    velocity = if size(field_data["velocity"], 1) == 2
        # If velocity is 2D, add 0.0 for the z component
        (field_data["velocity"][1, i], field_data["velocity"][2, i], 0.0)
    else
        # If velocity is 3D, use all three components
        (field_data["velocity"][1, i], field_data["velocity"][2, i],
         field_data["velocity"][3, i])
    end

    # Create a new Particle instance
    particles[i] = Particle(index, coordinates, density, velocity, pressure)
end

# Display some particles for verification
for particle in particles[1:2]
    println(particle)
end

println("Coords: ", size(coords))
println("velocity ", size(field_data["velocity"]))

# TODO FieldData 