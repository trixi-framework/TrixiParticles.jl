# Sample Geometry

Generating the initial configuration of a simulation requires filling volumes (3D) or surfaces (2D) of complex geometries with particles.
The algorithm to sample a complex geometry should be robust and fast,
since for large problems (high numbers of particles) or complex geometries (many geometry faces),
generating the initial configuration is not trivial and can be very expensive in terms of computational cost.
We therefore use a [winding number](https://en.wikipedia.org/wiki/Winding_number) approach for an inside-outside segmentation of an object.
The winding number $w(\mathbf{p})$ is a signed integer-valued property of a point $\mathbf{p}$ and is defined as

$w(\mathbf{p}) = \frac{1}{2 \pi} \sum^n_{i=1} \Theta_i,$

where  $\Theta_i$ is the **signed** angle between vectors from two consecutive vertices to $\mathbf{p}$.
In 3D, we refer to the solid angle of an *oriented* triangle with respect to $\mathbf{p}$

We provide the following methods to calculate $w(\mathbf{p})$:
- Horman et al. (2001) evaluated the winding number combined with an even-odd rule and is only for 2D polygons (see [WindingNumberHorman](@ref)).
- Naive winding: Jacobson et al. (2013) generalized the winding number so that the algorithm can be applied for 2D and also for 3D geometries (see [WindingNumberJacobson](@ref)).
- Hierarchical winding: Jacobson et al. (2013) also introduced a fast hierarchical evaluation of the winding number. For further information see the description below.

## [Hierarchical Winding](@id hierarchical_winding)
According to Jacobson et al. (2013) the winding number is the sum of harmonic functions defined for each edge (2D) or face (3D).
We can show this with the following example in which we determine the winding number for each edge of a triangle separately and sum them up:

```julia
using TrixiParticles

particle_spacing = 0.025

triangle = stack([[0.0, 0.0], [1.0, 0.0], [0.5, 0.7], [0.0, 0.0]])

edge1 = deleteat!(TrixiParticles.Polygon(triangle), [2, 3])
edge2 = deleteat!(TrixiParticles.Polygon(triangle), [1, 3])
edge3 = deleteat!(TrixiParticles.Polygon(triangle), [1, 2])

algorithm = WindingNumberJacobson()

grid = RectangularShape(particle_spacing, (80, 80), (-0.5, -0.7); tlsph=true, density=1.0)

inpoly, w1 = algorithm(edge1, grid.coordinates; store_winding_number=true)
inpoly, w2 = algorithm(edge2, grid.coordinates; store_winding_number=true)
inpoly, w3 = algorithm(edge3, grid.coordinates; store_winding_number=true)

w = w1 + w2 + w3

trixi2vtk(grid, w=w, w1=w1, w2=w2, w3=w3)
```

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/89f0b7e3-5d47-4eba-9d29-34cf54f1b246" alt="triangle"/>
</figure>
```

That means, if we want to calculate the winding number only for a specific area of interest, e.g. the top right square of a circle, we can use the closure of this area as we can see in the following image.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/8e1c2a6d-3b28-4f75-991f-0d0ecc9f6db1" alt="circle"/>
</figure>
```

For a continuous problem, we then have an open surface $\mathcal{S}$ and an arbitrary closing surface $\bar{\mathcal{S}}$, where $\partial \bar{\mathcal{S}} = \partial \mathcal{S}$,
such that $\mathcal{B} = \bar{\mathcal{S}} \cup \mathcal{S}$ is some closed oriented surface.
If an arbitrary query point $\mathbf{p}$ is outside the convex hull of $\mathcal{B}$, we know that

$w_{\mathcal{S}}(\mathbf{p}) + w_{\bar{\mathcal{S}}}(\mathbf{p}) = w_{\mathcal{B}}(\mathbf{p}) = 0.$

This means

$w_{\mathcal{S}}(\mathbf{p}) = - w_{\bar{\mathcal{S}}}(\mathbf{p}) ,$

regardless of how $\bar{\mathcal{S}}$ is constructed.

### Bounding volume hierarchy

If we construct a bounding volume hierarchy $T$ and evaluate $w$ hierarchical,
we can perform asymptotically better than the naive computation of $w$ for the whole geometry as is shown by Jacobson et al. (2013).

The algorithm behind (Jacobson et al., Algorithm 2, p. 5) calls itself recursively, where the recursion stops with the following criteria.

- if $T$ is a leaf then evaluate the winding number naively $w(\mathbf{p}, T.\mathcal{S})$ where $T.\mathcal{S}$ is the open surface in $T$.
- else if $\mathbf{p}$ is outside $T$ then evaluate the winding number naively $-w(\mathbf{p}, T.\bar{\mathcal{S}})$ where $T.\bar{\mathcal{S}}$ is the closing surface of $T$.

#### Continuous geometry

For an arbitrary continuous geometry, the steps are as follows.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/66c423ae-28ce-4d48-aa52-777726a3a677" alt="continuous closing"/>
</figure>
```

(1):
- Recurse left: $w_{\text{left}} = \text{\texttt{hierarchical\_winding}} (\mathbf{p}, T.\text{left})$
- Recurse right: $w_{\text{right}} = \text{\texttt{hierarchical\_winding}} (\mathbf{p},T.\text{right})$

(2):
- Query point $\mathbf{p}$ is outside bounding box $T$.
- Use closure $T.\bar{\mathcal{S}}$.
- Note the negative winding $\rightarrow w_{\mathcal{S}}(\mathbf{p}) = - w_{\bar{\mathcal{S}}}(\mathbf{p})$

$w_{\text{left}} = -\text{\texttt{naive\_winding}} (\mathbf{p}, T.\bar{\mathcal{S}})$

(3):
- Bounding box $T$ is a leaf. Use open surface $T.\mathcal{S}$

$w_{\text{right}} = \text{\texttt{naive\_winding}} (\mathbf{p}, T.\mathcal{S})$

The reconstructed surface will then look as in the following image.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/920bb4f1-1336-4e77-b06d-d5b46ca0d8d5" alt="reconstructed surface"/>
</figure>
```

We finally sum up the winding numbers

$w = w_{\text{left}} + w_{\text{right} } = -w_{T_{\text{left}}.\bar{\mathcal{S}}} + w_{T_{\text{right}}.\mathcal{S}}$

#### Discrete geometry

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/a9b59cc3-5421-40af-b0b0-f4c18a5a7078" alt="discrete geometry"/>
</figure>
```

To construct the hierarchy for discrete piecewise-linear surfaces (1), we have to do the following.

(2):
Each edge is distributed to the child whose box contains the edge's barycenter.
Splitting stops when the number of a box's edges slips below a
threshold (usually $\approx 100$ faces in 3D, here: 6 edges).

(3):
For the closure, Jacobson et al. (2013) define *exterior vertices* (*exterior edges* in 3D)
as boundary vertices of such a segmentation.
To find them, we traverse around each edge (face in 3D) in order, and
increment or decrement for each vertex (edge) a specific counter.

```julia
  # 2D
  v1 = edge_vertices_ids[edge][1]
  v2 = edge_vertices_ids[edge][2]

  vertex_count[v1] += 1
  vertex_count[v2] -= 1
```
```julia
  # 3D
  edge1 = face_edges_ids[face][1]

  if edge_vertices_ids[edge1] == (v1, v2)
      directed_edges[edge1] += 1
  else
      directed_edges[edge1] -= 1
  end
```

In 2D, a vertex is declared as exterior if `vertex_count(vertex) != 0`.
The exterior vertices are then connected to one arbitrary
exterior vertex using appropriately oriented line segments

```julia
  edge = vertex_count[v] > 0 ? (closing_vertex, v) : (v, closing_vertex)
```

In 3D, an edge is declared as exterior if `directed_edges[edge] != 0`
The exterior edges are triangulated with one exterior vertex $k$ with orientation
- ``\{i, j, k\}`` if `directed_edges[edge] < 0`
- ``\{j, i, k\}`` if `directed_edges[edge] > 0`

#### Incorrect evaluation

If we follow the algorithm, we know that recursion stops if

- The bounding box $T$ is a leaf
- The query point $\mathbf{p}$ is outside the box

(1): If $\mathbf{p}$ is outside the box, we calculate the winding number with the (red) closure of the box.

(2): If $\mathbf{p}$ is inside the box, we use the (blue) edges distributed to the box.

(3): In this case, it leads to an incorrect evaluation of the winding number.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/7bae164a-8d5b-4761-9d54-9abf99fca94a" alt="incorrect evaluation"/>
</figure>
```

#### Correct evaluation
Jacobson et al. (2013) therefore resize the bounding box, since it doesn't matter if the boxes overlap.

```@raw html
<figure>
  <img src="https://github.com/user-attachments/assets/097f01f4-1f37-48e4-968a-4c0970548b24" alt="correct evaluation resizing"/>
</figure>
```

To avoid resizing, we calculate the closure of the bounding box differently:
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
Pages = [joinpath("preprocessing", "point_in_poly", "winding_number_horman.jl")]
```

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("preprocessing", "point_in_poly", "winding_number_jacobson.jl")]
```

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("preprocessing", "geometries", "io.jl")]
```

### [References](@id references_complex_shape)
- Alec Jacobson, Ladislav Kavan, and Olga Sorkine-Hornung "Robust inside-outside segmentation using generalized winding numbers".
  In: ACM Transactions on Graphics, 32.4 (2013), pages 1--12.
  [doi: 10.1145/2461912.2461916](https://igl.ethz.ch/projects/winding-number/robust-inside-outside-segmentation-using-generalized-winding-numbers-siggraph-2013-jacobson-et-al.pdf)
- Kai Horman, Alexander Agathos "The point in polygon problem for arbitrary polygons".
  In: Computational Geometry, 20.3 (2001), pages 131--144.
  [doi: 10.1016/s0925-7721(01)00012-8](https://doi.org/10.1016/S0925-7721(01)00012-8)
