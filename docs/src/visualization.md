# Visualization

## Export VTK files
You can export particle data as VTK files by using the [`SolutionSavingCallback`](@ref).
All our [predefined examples](examples.md) are already using this callback to export VTK files to the `out` directory (relative to the directory that you are running Julia from).
VTK files can be read by visualization tools like [ParaView](https://www.paraview.org/) and [VisIt](https://visit.llnl.gov/).

### ParaView

Follow these steps to view the exported VTK files in ParaView:

1. Click `File -> Open`.
2. Navigate to the `out` directory (relative to the directory that you are running Julia from).
3. Open both `boundary_1.pvd` and `fluid_1.pvd`.
4. Click "Apply", which by default is on the left pane below the "Pipeline Browser".
5. Hold the left mouse button to move the solution around.

You will now see the following:
![image](https://github.com/user-attachments/assets/383d323a-3020-4232-9dc3-682b0afe8653)

It is useful to make the particles larger.
For this, **first** make sure you have "fluid_1.pvd" highlighted in the "Pipeline Browser" then in the "Properties" window in the bottom left change "Point Size" to a larger value.
![image](https://github.com/user-attachments/assets/6e975d2c-82ed-4d53-936b-bb0beafaf515)

To now view the result variables **first** make sure you have "fluid_1.pvd" highlighted in the "Pipeline Browser" then select them in the variable selection combo box (see picture below).
Let's, for example, pick "density". To now view the time progression of the result hit the "play button" (see picture below).
![image](https://github.com/user-attachments/assets/10dcf7eb-5808-4d4d-9db8-4beb25b5e51a)

## API

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("visualization", file), readdir(joinpath("..", "src", "visualization")))
```
