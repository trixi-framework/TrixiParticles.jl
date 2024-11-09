using ReadVTK

vtk = VTKFile("out_vtk/two_points.vtu")

"""
# Fields of vtk

- `filename`: original path to the VTK file that has been read in
- `xml_file`: object that represents the XML file
- `file_type`: currently only `"UnstructuredGrid"` or `"ImageData"` are supported
- `version`: VTK XML file format version
- `byte_order`: can be `LittleEndian` or `BigEndian` and must currently be the same as the system's
- `compressor`: can be empty (no compression) or `vtkZLibDataCompressor`
- `appended_data`: in case of appended data (see XML documentation), the data is stored here for
                   convenient retrieval (otherwise it is empty)
- `n_points`: number of points in the VTK file
- `n_cells`: number of cells in the VTK file`
"""

cell_data = get_cell_data(vtk)

point_data = get_point_data(vtk)
# create an vector with the names of the point data like "velocity"
names = point_data.names

# create an vector with the information about the Data like "offset"
data_arrays = point_data.data_arrays

# information about the file like "UnstructedGrid"
vtk_file = point_data.vtk_file

# gets the data of the vtk-file
appended_data = vtk.appended_data

# gets the coords of all the points
coords = get_points(vtk)

# gets the information about the cells
cells = get_cells(vtk)
