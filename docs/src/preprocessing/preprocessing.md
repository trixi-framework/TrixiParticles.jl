# Sample Geometry
Generating the initial configuration of a simulation requires filling volumes (3D) or surfaces (2D) of complex geometries with particles.
The algorithm to sample a complex geometry should be robust and fast,
since for large problems (high number of particles) or complex geometries (many geometry-faces),
generating the initial configuration is not trivial and can be very expensive in terms of computational costs.
We therefore use a winding number approach for an inside-outside segmentation of an object.

We provide the following methods to calculate the [winding number](https://en.wikipedia.org/wiki/Winding_number):
- Horman et al. (2001) evaluated the winding number combined with an even-odd rule and is only for 2D polygons (see [WindingNumberHorman](@ref)).
- Naive winding: Jacobson et al. (2013) generalized the winding number so that the algorithm can be applied for 2D and also for 3D geometries (see [WindingNumberJacobson](@ref)).
- Hierarchical winding: Jacobson et al. (2013) also introduced a fast hierarchical evaluation of the winding number. For further information see description below.


## Hierarchical Winding
According to Jacobson et al. (2013) the winding number is the sum of harmonic functions defined for each edge (2D) or face (3D).
We can show this with the following example in which we determine the winding number for each edge of a triangle separately and sum them up (TODO: insert code):

$w = w_1 + w_2 + w_3$

![alt text](image.png)

That means, if we want to calculate the winding number only for a specific area of interest, e.g. the top right square of a circle, we can use the closure of this area as we can see in the following image.

![alt text](image-2.png)

For a continuous problem, we then have an open surface $\mathcal{S}$ and an arbitrary closing surface $\bar{\mathcal{S}}$, where $\partial \bar{\mathcal{S}} = \partial \mathcal{S}$,
such that $\mathcal{B} = \bar{\mathcal{S}} \cup \mathcal{S}$ is some closed oriented surface.
If an arbitrary query point $\mathbf{p}$ is outside the convex hull of $\mathcal{B}$, we know that

$w_{\mathcal{S}}(\mathbf{p}) + w_{\bar{\mathcal{S}}}(\mathbf{p}) = w_{\mathcal{B}}(\mathbf{p}) = 0.$

This means

$w_{\mathcal{S}}(\mathbf{p}) = - w_{\bar{\mathcal{S}}}(\mathbf{p}) ,$

regardless of how $\bar{\mathcal{S}}$ is constructed.

If we construct a bounding volume hierarchy $T$ and evaluate $w$ hierarchical,
we can perform asymptotically better than the naive winding computing $w$ for the whole geometry as is shown by Jacobson et al. (2013).

The algorithm behind (Jacobson et al., Algorithm 2, p. 5) calls itself recursively, where the recursion stops with the following criteria.

- if $T$ is a leaf then evaluate the winding number naively $w(\mathbf{p}, T.\mathcal{S})$ where $T.\mathcal{S}$ is the open surface in $T$.
- else if $p$ is outside $T$ then evaluate the winding number naively $-w(\mathbf{p}, T.\bar{\mathcal{S}})$ where $T.\bar{\mathcal{S}}$ is the closing surface of $T$.

### Continuos Geometry

For an arbitrary continuous geometry, the steps are as follows.

![alt text](image-3.png)

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

![alt text](image-4.png)

We finally sum up the winding numbers

$w = w_{\text{left}} + w_{\text{right} } = -w_{T_{\text{left}}.\bar{\mathcal{S}}} + w_{T_{\text{right}}.\mathcal{S}}$

### Discrete Geometry

![alt text](image-5.png)

To construct the hierarchy for discrete piecewise-linear surfaces (1), we have to do the following.

(2):
- Each edge is distributed to the child whose box contains the edge's barycenter.
- Splitting stops when the number of a box's edges slips below a
  threshold (usually $\approx 100$ faces in 3D, here: 6 edges).

(3)
- For the closure, Jacobson et al. (2013) define \textit{exterior vertices} (\textit{exterior edges} in 3D)
  as boundary vertices of such a segmentation.
- To find them, we traverse around each edge (face in 3D) in order, and
  increment or decrement for each edge a specific counter.
- An edge (face in 3D) is declared as exterior if `count(edge) != 0`.
- In 2D, the exterior vertices are then connected to one arbitrary
  exterior vertex using appropriately oriented (depending on the sign of the `count`) line segments.
- In 3D, exterior edges are triangulated with one vertex

#### Incorrect evaluation

If we follow the algorithm, we know that recursion stops if

- The bounding box $T$ is a leaf
- The query point $\mathcal{p}$ is outside the box

(1): If $\mathbf{p}$ is outside the box, we calculate the winding number with the (red) closure of the box.

(2): If $\mathbf{p}$ is inside the box, we use the (blue) edges distributed to the box.

(3): In this case, it leads to an incorrect evaluation of the winding number.

![alt text](image-6.png)

#### Correct evaluation
Jacobson et al. (2013) therefore resize the bounding box, since it doesn't matter if the boxes overlap.

![alt text](image-7.png)


To avoid resizing, we calculate the closure of the bounding box differently:
- Exclude intersecting edges in the calculation of the exterior vertices
- This way, all exterior vertices are inside the bounding box, and so will be the closing surface
- The intersecting edges are later added with flipped orientation,
  so that the closing is actually a closing of the exterior plus intersecting edges.

![alt text](image-8.png)

The evaluation then looks as follows.

![alt text](image-9.png)

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
