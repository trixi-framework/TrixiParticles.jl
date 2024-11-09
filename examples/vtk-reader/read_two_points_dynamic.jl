using ReadVTK

#vtk = VTKFile("out/fluid_1_0.vtu")
#vtk = VTKFile("out_vtk/two_points.vtu")
vtk = VTKFile("out_vtk/rectangle_of_water.vtu")

point_data_file = get_point_data(vtk)

point_names = keys(point_data_file)

for point_name in point_names
    point_array = point_data_file[point_name]

    point_data = get_data(point_array)

    println(point_name)
    println(point_data)
end

coords = get_points(vtk)
for i in 1:size(coords, 2)
    println("Coords $i: (", coords[1, i], ", ", coords[2, i], ", ", coords[3, i], ")")
end
