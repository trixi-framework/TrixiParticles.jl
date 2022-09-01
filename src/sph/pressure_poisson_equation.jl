@doc raw"""
    PPEExplicitLiu(eta)

test
```math
a+b
```
where ``a`` ...
"""
struct PPEExplicitLiu{ELTYPE}
    eta     :: ELTYPE 
    nu      :: ELTYPE # kinematic viscosity
    function PPEExplicitLiu(eta, nu)
        new{typeof(eta)}(eta, nu)
    end
end


function (PressurePoissonEquation::PPEExplicitLiu)(u, semi, particle)
    @unpack smoothing_kernel, smoothing_length, density_calculator, neighborhood_search, cache = semi
    @unpack mass, pressure = cache
    @unpack eta = PressurePoissonEquation
    density_particle = get_particle_density(u, cache, density_calculator, particle)
    particle_coords = get_particle_coords(u, semi, particle)

    vel_intermediate_particle = get_intermediate_vel(u, semi, particle)

    term_sum_A = 0.0
    term_sum_B = 0.0
    term_sum_AP = 0.0

    # TBD get timestep from integrator.dt
    dt = 1e-03

    for neighbor in eachneighbor(particle, u, neighborhood_search, semi)

        neighbor_coords = get_particle_coords(u, semi, neighbor)
        density_neighbor = get_particle_density(u, cache, density_calculator, neighbor)
        vel_intermediate_neighbor = get_intermediate_vel(u, semi, neighbor)

        distance = norm(particle_coords - neighbor_coords)       
        pos_diff = particle_coords - neighbor_coords
        vel_diff = vel_intermediate_particle - vel_intermediate_neighbor
        
        grad_kernel = kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

        dot_prod = sum(pos_diff .* grad_kernel)
        A_ij = 8 * mass[neighbor] * dot_prod / ((density_particle + density_neighbor)^2 * (distance^2 + eta^2))
        
        term_sum_A += A_ij
        term_sum_AP += A_ij*pressure[neighbor]
        term_sum_B +=  -mass[neighbor] * sum(vel_diff .* grad_kernel) / density_neighbor

    end

    return (term_sum_AP + term_sum_B/dt)/term_sum_A

end
