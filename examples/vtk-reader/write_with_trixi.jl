using TrixiParticles

rectangle = RectangularShape(0.1, (1, 2), (0.0, 0.0), velocity=[10.0, 0.0], density=1000.0)

custom_quantity = ones(nparticles(rectangle))

trixi2vtk(rectangle; filename="rectangle_of_water", output_directory="out_vtk",
          custom_quantity=custom_quantity)