@doc raw"""
    PPEExplicitLiu(eta)

Solving the pressure with the pressure Poisson's equation (PPE). 
The PPE can be written as 
```math
\sum_{j=1}^{N} m_j \frac{8 (P_i^{n+1}-P_j^{n+1}) \vec{r_{ij}} \cdot \nabla_i W_{ij} }{ (\rho_i + \rho_j)^2(r_{ij}^2 + \eta^2) } = 
    - \frac{1}{\Delta t} \sum_{j=1}^N \frac{m_j}{\rho_j} \vec{u_{ij}}^* \cdot \nabla_i W_{ij}
```
where ``\vec{r}``,  ``\rho``, ``P``, ``m`` are the position, density, pressure and mass of the particle, respectively. 
``W`` is the kernel-function and ``\eta`` is to keep tje denominator nonzero and is usually set to ``\eta = 0.1h``.
The subscript ``i`` and ``j`` denotes to fluid particle and its neighbor, respectively (e.g. ``\vec{r_{ij}} = \vec{r_{i}}-\vec{r_{j}}`` ) and ``n`` 
and ``n+1`` denote the time steps. It should be noted that the density always satisfies ``\rho_i = \rho_j = \rho_0``.

The SPH projection technique (Cummins, 1994) employs an intermediate velocity ``\vec{u}^*`` in a prediction step and can be calculated as:
```math
u^{*} = \vec{u}^n + \left( \nu \nabla^2 \vec{u}^n + \vec{g} \right) \Delta t
```
where ``\vec{u}``,  ``\nu``, ``\vec{g}`` are the velocity of the particle, the kinematic viscosity and the gravitational acceleration, respectively. 

Derived from (Rafiee et al, 2009), the pressure can be explicitly calculated as
```math
P_i^{n+1} = \frac{\sum_{j=1}^N A_{ij} P_{ij}^n + B_i}{\sum_{j=1}^N A_{ij} }
```
where
```math
A_{ij} = \frac{8 m_j \vec{r_{ij}} \cdot \nabla_i W_{ij} }{  (\rho_i + \rho_j)^2(r_{ij}^2 + \eta^2)  }
```

```math
B_{i} = - \frac{1}{\Delta t} \sum_{j=1}^N \frac{ m_{j} }{ \rho_j } \vec{u_{ij}^*} \cdot \nabla_i W_{ij}
```

The pressure of the particle near the free surface maybe negative or much larger than zero. (Lu Liu et al, 2022) employed the density at the intermediate 
step ``\rho^*`` in the PPE to check if the particle is near a surface or not. They force the pressure to zero if ``\rho^*`` is lower than ``0.7 \rho_0``.

!!! note "Note"
    The intermediate density ``\rho^*`` can based on both, the density summation or the velocity divergence.  


References:
- Lu Liue, Jie Wu, Shunying Ji. "DEM–SPH coupling method for the interaction between irregularly shaped granular materials and ﬂuids".
  In: Powder Technology Vol. 400 (2022).
  [doi: 10.1016/j.powtec.2022.117249](https://doi.org/10.1016/j.powtec.2022.117249)
- Ashkan Rafiee, Krish P. Thiagrajan. "An SPH projection method for simulating fluid-hypoelastic structure interaction".
  In: Computer Methods Appl. Engrg. 198 (2009), pages 2785-2795.
  [doi: :10.1016/j.cma.2009.04.001](https://doi.org/:10.1016/j.cma.2009.04.001)
- Sven J. Cummins, Murray Rudman. "An SPH Projection Method".
  In: Journal of Computational Physics Vol. 152 (1999), pages 584-604.
  [doi: :10.1006/jcph.1999.6246](https://doi.org/10.1006/jcph.1999.6246)
"""
struct PPEExplicitLiu{ELTYPE}
    eta     :: ELTYPE 
    function PPEExplicitLiu(eta)
        new{typeof(eta)}(eta)
    end
end


function (PressurePoissonEquation::PPEExplicitLiu)(u, semi, particle)
    @unpack smoothing_kernel, smoothing_length, density_calculator, neighborhood_search, cache = semi
    @unpack mass, prior_pressure, dt = cache
    @unpack eta = PressurePoissonEquation
    density_particle = get_particle_density(u, cache, density_calculator, particle)
    # TBD: Better solution to store the intermediate density. 
    intermediate_density = u[end, particle]
    
    if intermediate_density > 0.9 * density_particle
        particle_coords = get_particle_coords(u, semi, particle)

        vel_intermediate_particle = get_intermediate_vel(u, semi, particle, dt[1])

        term_sum_A = 0.0
        term_sum_B = 0.0
        term_sum_AP = 0.0

        for neighbor in eachneighbor(particle, u, neighborhood_search, semi)

            neighbor_coords = get_particle_coords(u, semi, neighbor)
            density_neighbor = get_particle_density(u, cache, density_calculator, neighbor)
            vel_intermediate_neighbor = get_intermediate_vel(u, semi, neighbor, dt[1])

            distance = norm(particle_coords - neighbor_coords)
            pos_diff = particle_coords - neighbor_coords
            vel_diff = vel_intermediate_particle - vel_intermediate_neighbor

            if eps() < distance <= compact_support(smoothing_kernel, smoothing_length)
                grad_kernel = kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

                dot_prod = sum(pos_diff .* grad_kernel)
                A_ij = 8 * mass[neighbor] * dot_prod / ((density_particle + density_neighbor)^2 * (distance^2 + eta^2))

                term_sum_A += A_ij
                term_sum_AP += A_ij * prior_pressure[neighbor]
                term_sum_B += mass[neighbor] * sum(vel_diff .* grad_kernel) / density_neighbor
            end
        end

        return (term_sum_AP - term_sum_B / dt[1]) / term_sum_A

    else
        return 0.0
    end

end






@inline function get_intermediate_coords(u, semi, particle, dt)
    r = get_particle_coords(u, semi, particle)
    intermediate_vel = get_intermediate_vel(u, semi, particle, dt)
    return r + intermediate_vel*dt
end

@inline function get_intermediate_vel(u, semi, particle, dt)
    @unpack gravity, cache = semi
    dv_viscosity = get_dv_viscosity(semi, particle)
    u = get_particle_vel(u, semi, particle)
    return u +(dv_viscosity + gravity)*dt
end


@inline function get_dv_viscosity(semi, particle)
    @unpack cache = semi
    return SVector(ntuple(@inline(dim -> cache.dv_viscosities[dim, particle]), Val(ndims(semi))))
end