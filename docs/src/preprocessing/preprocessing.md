# [Sampling of Geometries](@id sampling_of_geometries)

Generating the initial configuration of a simulation requires filling volumes (3D) or surfaces (2D) of complex geometries with particles.
The algorithm to sample a complex geometry should be robust and fast,
since for large problems (large numbers of particles) or complex geometries (many geometry faces),
generating the initial configuration is not trivial and can be very expensive in terms of computational cost.
We therefore use a [winding number](https://en.wikipedia.org/wiki/Winding_number) approach for an inside-outside segmentation of an object.
The winding number ``w(\mathbf{p})`` is a signed integer-valued function of a point ``\mathbf{p}`` and is defined as

```math
w(\mathbf{p}) = \frac{1}{2 \pi} \sum^n_{i=1} \Theta_i.
```

Here, ``\Theta_i`` is the *signed* angle between ``\mathbf{c}_i - \mathbf{p}`` and ``\mathbf{c}_{i+1} - \mathbf{p}`` where ``\mathbf{c}_i`` and ``\mathbf{c}_{i+1}`` are two consecutive vertices on a curve.
In 3D, we refer to the solid angle of an *oriented* triangle with respect to ``\mathbf{p}``.

We provide the following methods to calculate ``w(\mathbf{p})``:
- [Hormann et al. (2001)](@cite Hormann2001) evaluate the winding number combined with an even-odd rule, but only for 2D polygons (see [WindingNumberHormann](@ref)).
- Naive winding: [Jacobson et al. (2013)](@cite Jacobson2013) generalized the winding number so that the algorithm can be applied for both 2D and 3D geometries (see [WindingNumberJacobson](@ref)).
- Hierarchical winding: [Jacobson et al. (2013)](@cite Jacobson2013) also introduced a fast hierarchical evaluation of the winding number. For further information see the description below.

## [Hierarchical Winding](@id hierarchical_winding)
According to [Jacobson et al. (2013)](@cite Jacobson2013) the winding number with respect to a polygon (2D) or triangle mesh (3D) is the sum of the winding numbers with respect to each edge (2D) or face (3D).
We can show this with the following example in which we determine the winding number for each edge of a triangle separately and sum them up:

```julia
using TrixiParticles
using Plots

triangle = [125.0 375.0 250.0 125.0;
            175.0 175.0 350.0 175.0]

# Delete all edges but one
edge1 = deleteat!(TrixiParticles.Polygon(triangle), [2, 3])
edge2 = deleteat!(TrixiParticles.Polygon(triangle), [1, 3])
edge3 = deleteat!(TrixiParticles.Polygon(triangle), [1, 2])

algorithm = WindingNumberJacobson()

grid = hcat(([x, y] for x in 1:500, y in 1:500)...)

_, w1 = algorithm(edge1, grid; store_winding_number=true)
_, w2 = algorithm(edge2, grid; store_winding_number=true)
_, w3 = algorithm(edge3, grid; store_winding_number=true)

w = w1 + w2 + w3

heatmap(1:500, 1:500, reshape(w1, 500, 500)', color=:coolwarm, showaxis=false,
        tickfontsize=12, size=(570, 500), margin=6 * Plots.mm)
heatmap(1:500, 1:500, reshape(w2, 500, 500)', color=:coolwarm, showaxis=false,
        tickfontsize=12, size=(570, 500), margin=6 * Plots.mm)
heatmap(1:500, 1:500, reshape(w3, 500, 500)', color=:coolwarm, showaxis=false,
        tickfontsize=12, size=(570, 500), margin=6 * Plots.mm)
heatmap(1:500, 1:500, reshape(w, 500, 500)', color=:coolwarm, showaxis=false,
        tickfontsize=12, size=(570, 500), margin=6 * Plots.mm, clims=(-1, 1))

```

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/bf491b2d-740e-4136-8a7b-e321f26f86fd" alt="triangle"/>
</figure>
```

This summation property has some interesting consequences that we can utilize for an efficient computation of the winding number.
Let ``\mathcal{S}`` be an open surface and ``\bar{\mathcal{S}}`` an arbitrary closing surface, such that

```math
\partial \bar{\mathcal{S}} = \partial \mathcal{S}
```

and ``\mathcal{B} = \bar{\mathcal{S}} \cup \mathcal{S}`` is some closed oriented surface.
For any query point ``\mathbf{p}`` outside of ``\mathcal{B}``, we know that

```math
w_{\mathcal{S}}(\mathbf{p}) + w_{\bar{\mathcal{S}}}(\mathbf{p}) = w_{\mathcal{B}}(\mathbf{p}) = 0.
```

This means

```math
w_{\mathcal{S}}(\mathbf{p}) = - w_{\bar{\mathcal{S}}}(\mathbf{p}),
```

regardless of how ``\bar{\mathcal{S}}`` is constructed (as long as ``\mathbf{p}`` is outside of ``\mathcal{B}``).

We can use this property in the discrete case to efficiently compute the winding number of a query point
by partitioning the polygon or mesh in a "small" part (as in consisting of a small number of edges/faces) and a "large" part.
For the small part we just compute the winding number, and for the large part we construct a small closing and compute its winding number.
The partitioning is based on a hierarchical construction of bounding boxes.

### Bounding volume hierarchy

To efficiently find a "small part" and a "large part" as mentioned above, we construct a hierarchy of bounding boxes by starting with the whole domain and recursively splitting it in two equally sized boxes.
The resulting hierarchy is a binary tree.

The algorithm by Jacobsen et al. (Algorithm 2, p. 5) traverses this binary tree recursively until we find the leaf in which the query point is located.
The recursion stops with the following criteria:

- if the bounding box ``T`` is a leaf then ``T.\mathcal{S} = \mathcal{S} \cap T``, the part of ``\mathcal{S}``
  that lies inside ``T``, is the "small part" mentioned above, so evaluate the winding number naively as ``w(\mathbf{p}, T.\mathcal{S})``.
- else if ``\mathbf{p}`` is outside ``T`` then ``T.\mathcal{S}`` is the "large part", so evaluate the winding number naively
  as ``-w(\mathbf{p}, T.\bar{\mathcal{S}})``, where ``T.\bar{\mathcal{S}}`` is the closing surface of ``T.\mathcal{S}``.

#### Continuous example

Now consider the following continuous (not discretized to a polygon) 2D example.
We compute the winding number of the point ``\mathbf{p}`` with respect to ``\mathcal{S}`` using the depicted hierarchy of bounding boxes.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/0ca2f475-6dd5-43f9-8b0c-87a0612ecdf4" alt="continuous closing"/>
</figure>
```

(1):
- Recurse left: ``w_{\text{left}} = \text{\texttt{hierarchical\_winding}} (\mathbf{p}, T.\text{left})``
- Recurse right: ``w_{\text{right}} = \text{\texttt{hierarchical\_winding}} (\mathbf{p},T.\text{right})``

(2):
- Query point ``\mathbf{p}`` is outside bounding box ``T``, so don't recurse deeper.
- Compute ``w_{\mathcal{S}}(\mathbf{p}) = - w_{\bar{\mathcal{S}}}(\mathbf{p})`` with the closure ``T.\bar{\mathcal{S}}``, which is generally much smaller (fewer edges in the discrete version) than ``T.\mathcal{S}``:

```math
w_{\text{left}} = -\text{\texttt{naive\_winding}} (\mathbf{p}, T.\bar{\mathcal{S}})
```

(3):
- Bounding box ``T`` is a leaf. Use open surface ``T.\mathcal{S}``:

```math
w_{\text{right}} = \text{\texttt{naive\_winding}} (\mathbf{p}, T.\mathcal{S})
```

The reconstructed surface will then look as in the following image.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/920bb4f1-1336-4e77-b06d-d5b46ca0d8d5" alt="reconstructed surface"/>
</figure>
```

We finally sum up the winding numbers

```math
w = w_{\text{left}} + w_{\text{right} } = -w_{T_{\text{left}}.\bar{\mathcal{S}}} + w_{T_{\text{right}}.\mathcal{S}}
```

#### Discrete example

We will now go through the discrete version of the example above.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/a9b59cc3-5421-40af-b0b0-f4c18a5a7078" alt="discrete geometry"/>
</figure>
```

To construct the hierarchy for the discrete piecewise-linear example in (1), we have to do the following.

(2):
Each edge is distributed to the child whose box contains the edge's barycenter (red dots in (2)).
Splitting stops when the number of a box's edges slips below a
threshold (usually ``\approx 100`` faces in 3D, here: 6 edges).

(3):
For the closure, [Jacobson et al. (2013)](@cite Jacobson2013) define *exterior vertices* (*exterior edges* in 3D)
as boundary vertices of such a segmentation (red dots in (3)).
To find them, we traverse around each edge (face in 3D) in order, and
increment or decrement for each vertex (edge) a specific counter.

```julia
v1 = edge_vertices_ids[edge][1]
v2 = edge_vertices_ids[edge][2]

vertex_count[v1] += 1
vertex_count[v2] -= 1
```

In 2D, a vertex is declared as exterior if `vertex_count(vertex) != 0`, so there is not the same amount of edges in this box going into versus out of the vertex.
To construct the closing surface, the exterior vertices are then connected to one arbitrary
exterior vertex using appropriately oriented line segments:

```julia
edge = vertex_count[v] > 0 ? (closing_vertex, v) : (v, closing_vertex)
```

The resulting closed surface ``T.S \cup T.\bar{S}`` then has the same number of edges going into and out of each vertex.

#### Incorrect evaluation

If we follow the algorithm, we know that recursion stops if

- the bounding box ``T`` is a leaf or
- the query point ``\mathbf{p}`` is outside the box.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/7bae164a-8d5b-4761-9d54-9abf99fca94a" alt="incorrect evaluation"/>
</figure>
```

(1): The query point ``\mathbf{p}`` is outside the box, so we calculate the winding number with the (red) closure of the box.

(2): The query point ``\mathbf{p}`` is inside the box, so we use the (blue) edges distributed to the box.

(3): In this case, it leads to an incorrect evaluation of the winding number.
The query point is clearly inside the box, but not inside the reconstructed surface.
This is because the property ``w_{\mathcal{S}}(\mathbf{p}) = - w_{\bar{\mathcal{S}}}(\mathbf{p})``
only holds when ``\mathbf{p}`` is outside of ``\mathcal{B}``, which is not the case here.

#### Correct evaluation
[Jacobson et al. (2013)](@cite Jacobson2013) don't mention this problem or provide a solution to it.
We contacted the authors and found that they know about this problem and solve it
by resizing the bounding box to fully include the closing surface
of the neighboring box, since it doesn't matter if the boxes overlap.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/097f01f4-1f37-48e4-968a-4c0970548b24" alt="correct evaluation resizing"/>
</figure>
```

To avoid resizing, we take a different approach and calculate the closure of the bounding box differently:
- Exclude intersecting edges in the calculation of the exterior vertices.
- This way, all exterior vertices are inside the bounding box, and so will be the closing surface.
- The intersecting edges are later added with flipped orientation,
  so that the closing is actually a closing of the exterior plus intersecting edges.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/a8ff9a7e-e6d6-44d1-9a29-7debddf2803d" alt="correct evaluation intersecting" width=60%/>
</figure>
```

The evaluation then looks as follows.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/9bb2d2ad-14e8-4bd0-a9bd-3c824932affd" alt="correct evaluation intersecting 2"/>
</figure>
```

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("preprocessing", "point_in_poly", "winding_number_hormann.jl")]
```

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("preprocessing", "point_in_poly", "winding_number_jacobson.jl")]
```
# [Read geometries from file](@id read_geometries_from_file)
Geometries can be imported using the [`load_geometry`](@ref) function.
- For 3D geometries, we support the binary and ASCII (`.stl`) format.
- For 2D geometries, the recommended format is DXF (`.dxf`), with optional support for a simple ASCII (`.asc`) format.

For 3D geometries, a convenience function [`extrude_geometry`](@ref) is provided to create a 3D triangulated surface by extruding a planar mesh.

## ASCII Format (.asc)
An .asc file contains a list of 2D coordinates, space-delimited, one point per line,
where the first column are the x values and the second the y values.
For example:
```plaintext
# ASCII
0.0 0.0
1.0 0.0
1.0 1.0
0.0 1.0
```
It is the user’s responsibility to ensure the points are ordered correctly.
This format is easy to generate and inspect manually.

## DXF Format (.dxf) – recommended
The DXF (Drawing Exchange Format) is a widely-used CAD format for 2D and 3D data.
We recommend this format for defining 2D polygonal.
Only DXF entities of type `LINE` or `POLYLINE` are supported.
To create DXF files from scratch, we recommend using the open-source tool [FreeCAD](https://www.freecad.org/).
For a less technical approach, we recommend using [Inkscape](https://inkscape.org/) to create SVG files and then export them as DXF.

### Creating DXF Files with FreeCAD

- Open FreeCAD and create a new document.
- Switch to the Sketcher workbench and create a new sketch.
- Choose the XY-plane orientation and draw your geometry.
- Select the object to be exported.
- Go to "File > Export..." and choose "AutoDesk DXF (*.dxf)" as the format.
- Ensure the following Import-Export options are enabled:
    - "Use legacy Python exporter".
    - "Project exported objects along current view direction".

### Creating DXF Files with Inkscape
SVG vector graphics can also be used as a basis for geometry.
Use Inkscape to open or create the SVG.
You can simply draw a Bezier curve by pressing "B" on your keyboard.
We reommend the mode "Create spiro paths" for a smooth curve.
Select the relevant path and export it as DXF:
- Go to "File > Save As...".
- Choose "Desktop Cutting Plotter (AutoCAD DXF R12)(*.dxf)" format.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("preprocessing", "geometries", "io.jl")]
```

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("preprocessing", "geometries", "triangle_mesh.jl")]
```


# [Particle Packing](@id particle_packing)
To obtain a body-fitted and isotropic particle distribution, an initial configuration (see [Sampling of Geometries](@ref sampling_of_geometries)) is first generated. This configuration is then packed using a [`ParticlePackingSystem`](@ref) following the steps introduced in [Neher2026](@cite).
For a hands-on tutorial with complete examples, see the [particle packing tutorial](@ref tut_packing).
The preprocessing pipeline consists of the following steps:

- Load geometry: Fig. 1, [`load_geometry`](@ref).
- Compute the signed distance field (SDF): Fig. 2, [`SignedDistanceField`](@ref).
- Generate boundary particles: Fig. 3, [`sample_boundary`](@ref).
- Initial sampling of the interior particles with inside-outside segmentation: Fig. 4, [`ComplexShape`](@ref).
- Pack the initial configuration of interior and boundary particles (Fig. 5): Fig. 6, [`ParticlePackingSystem`](@ref).

The input data can either be a 3D triangulated surface mesh represented in STL format or a 2D polygonal traversal of the geometry (see [`load_geometry`](@ref)).
The second step involves generating the SDF (see [`SignedDistanceField`](@ref)), which is necessary for the final packing step as it requires a surface detection.
The SDF is illustrated in Fig. 2, where the distances to the surface of the geometry are visualized as a color map.
As shown, the SDF is computed only within a narrow band around the geometry’s surface, enabling  a face-based neighborhood search (NHS) to be used exclusively during this step.
In the third step, the initial configuration of the boundary particles is generated (orange particles in Fig. 3).
Boundary particles are created by copying the positions of SDF points located outside the geometry but within a predefined boundary thickness (see [`sample_boundary`](@ref)).
In the fourth step, the initial configuration of the interior particles (green particles in Fig. 4) is generated using the hierarchical winding number approach (see [Hierarchical Winding](@ref hierarchical_winding)).
After steps **1** through **4**, the initial configuration of both interior and boundary particles is obtained, as illustrated in Fig. 5.
The interface of the geometry surface is not well resolved with the initial particle configuration.
Thus, in the final step, a packing algorithm by Zhu et al. [Zhu2021](@cite) is applied utilizing the SDF to simultaneously optimize the positions of both interior and boundary particles,
yielding an isotropic distribution while accurately preserving the geometry surface, as illustrated in Fig. 6.

```@raw html
<div style="display: flex; gap: 16px; flex-wrap: wrap;">
  <figure style="margin: 0; text-align: center;">
    <img src="https://github.com/user-attachments/assets/7fe9d1f1-1633-4377-8b97-a4d1778aee07" alt="geometry" style="max-width: 200px;">
    <figcaption>(1) Geometry representation</figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="https://github.com/user-attachments/assets/2b79188c-3148-49f1-8337-718721851bf5" alt="sdf" style="max-width: 200px;">
    <figcaption>(2) Signed distances to the surface</figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="https://github.com/user-attachments/assets/1501718f-d1f5-4f14-b1bc-2a2e581db476" alt="boundary" style="max-width: 200px;">
    <figcaption>(3) Boundary particles</figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="https://github.com/user-attachments/assets/f7376b15-324a-4da1-bb59-db01c7bd6620" alt="interior" style="max-width: 200px;">
    <figcaption>(4) Interior particles</figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="https://github.com/user-attachments/assets/4be889d6-e70a-4c5e-bef2-0071ea4d898c" alt="initial_config" style="max-width: 200px;">
    <figcaption>(5) Initial configuration</figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="https://github.com/user-attachments/assets/0f7aba29-3cf7-4ec1-8c95-841e72fe620d" alt="packed_config" style="max-width: 200px;">
    <figcaption>(6) Packed configuration</figcaption>
  </figure>
</div>
```


```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("preprocessing", "particle_packing", file), readdir(joinpath("..", "src", "preprocessing", "particle_packing")))
```
