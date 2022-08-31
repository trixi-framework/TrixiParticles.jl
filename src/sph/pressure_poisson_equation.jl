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


function (pressure_poisson_equation::PPEExplicitLiu)(u, semi, particle, neighbor, pos_diff, distance, density)
    @unpack smoothing_kernel, smoothing_length, cache = semi
    @unpack mass, pressure = cache
    @unpack eta = pressure_poisson_equation

    density_particle = get_particle_density(u, cache, density_calculator, particle)
    density_neighbor = get_particle_density(u, cache, density_calculator, neighbor)

    dt =  integrator.dt

    dot_prod = sum(pos_diff .* (kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance))
    A_ij = 8*mass[neighbor]*dot_prod/((density_particle+density_neighbor)^2*(distance^2+eta^2))

    B_i = mass[neighbor]*

    return density
end



function (get_intermediate_velocity::PPEExplicitLiu)(u, semi, particle, neighbor, )
    vel = get_particle_vel(u, semi, particle)
    intermediate_velocity = vel + 

end