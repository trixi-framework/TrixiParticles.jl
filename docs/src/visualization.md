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

It is useful to make the dot size dependent on the actual particle size.
For this, first make sure you have "fluid_1.pvd" highlighted in the "Pipeline Browser".
Then, in the Properties panel (bottom left), adjust the following settings:
1. "Representation" to "Point Gaussian".
2. Choose the right "Shader Preset": "Plain Circle" for 2D and "Sphere" for 3D.
3. Activate "Scale by Array" and select "`particle_spacing`" in "Gaussian Scale Array".
4. Deactivate "Use Scale Function".
5. Set the "Gaussian Radius" to "`0.5`".
![image](https://github.com/user-attachments/assets/194d9a09-5937-4ee4-b229-07078afe3ff0)

#### Visualization with Macro
To simplify the visualization of your particle data in ParaView, you can use a macro.
This macro automates the manual steps in the previous section to a single click of a button.
Install the macro as follows.

1. **Save the macro code** (see below) as a `.py` file, e.g. `PointGaussianMacro.py`.
2. Open **ParaView** and go to the top menu:
   **Macros** → **Import New Macro...** → Select the saved `.py` file.
3. The macro will now appear in the **Macros** menu and can optionally be pinned to the **toolbar**.
4. **Load your dataset** into ParaView.
5. **Select the dataset** in the Pipeline Browser.
6. Click on the macro name in the **Macros** menu (or toolbar, if pinned) to run it.
7. The Point Gaussian representation with `particle_spacing` scaling will be applied automatically.


---

#### Macro Code

```python
# trace generated using paraview version 5.13.1
#from paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

# get active source
source = GetActiveSource()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# get display properties
sourceDisplay = GetDisplayProperties(source, view=renderView1)

# change representation type
sourceDisplay.SetRepresentationType('Point Gaussian')

# modified display properties
sourceDisplay.ShaderPreset = 'Plain circle' # for 2D, change to 'Sphere' for 3D
sourceDisplay.ScaleByArray = 1
sourceDisplay.SetScaleArray = ['POINTS', 'particle_spacing']
sourceDisplay.UseScaleFunction = 0
sourceDisplay.GaussianRadius = 0.5
```

#### Show results
To now view the result variables **first** make sure you have "fluid_1.pvd" highlighted in the "Pipeline Browser" then select them in the variable selection combo box (see picture below).
Let's, for example, pick "density". To now view the time progression of the result hit the "play button" (see picture below).
![image](https://github.com/user-attachments/assets/10dcf7eb-5808-4d4d-9db8-4beb25b5e51a)

## API

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("io", file), readdir(joinpath("..", "src", "io")))
```
