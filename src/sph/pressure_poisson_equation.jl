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
    function PPEExplicitLiu(eta)
        new{typeof(eta)}(eta)
    end
end


function (PressurePoissonEquation::PPEExplicitLiu)(u, semi, particle)
    @unpack smoothing_kernel, smoothing_length, density_calculator, neighborhood_search, cache = semi
    @unpack mass, prior_pressure, dt = cache
    @unpack eta = PressurePoissonEquation
    density_particle = get_particle_density(u, cache, density_calculator, particle)
    intermediate_density = u[end, particle]
    
    if intermediate_density > 0.7 * density_particle
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