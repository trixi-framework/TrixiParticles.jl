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
![image](https://github.com/svchb/TrixiParticles.jl/assets/10238714/45c90fd2-984b-4eee-b130-e691cefb33ab)

To now view the result variables **first** make sure you have "fluid_1.pvd" highlighted in the "Pipeline Browser" then select them in the variable selection combo box (see picture below).
Let's, for example, pick "density". To now view the time progression of the result hit the "play button" (see picture below).
![image](https://github.com/svchb/TrixiParticles.jl/assets/10238714/7565a13f-9532-4a69-9f81-e79505400b1c)