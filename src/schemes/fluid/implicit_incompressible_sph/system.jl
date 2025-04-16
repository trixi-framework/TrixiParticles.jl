"""
ImplicitIncompressibleSPHSystem(initial_condition,
                                smoothing_kernel, smoothing_length;
                                viscosity=nothing,
                                acceleration=ntuple(_ -> 0.0,ndims(smoothing_kernel)))

System for particles of a fluid.
The implicit incompressible SPH (IISPH) scheme is used, wherein a linear systems gets solved 
iteratively in order to compute the correct pressure values such that the density deviation 
from the rest density is (almost) zero.
See [Implicit Incompressible SPH](@ref iisph) for more details on the method.

# Arguments
- `initial_condition`:  [`InitialCondition`](@ref) representing the system's particles.
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).

# Keyword Arguments
- `viscosity`:                  Viscosity model for this system (default: no viscosity).
                                See [`ArtificialViscosityMonaghan`](@ref) or [`ViscosityAdami`](@ref).
- `acceleration`:               Acceleration vector for the system. (default: zero vector)
"""

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
struct ImplicitIncompressibleSPHSystem{NDIMS, ELTYPE <: Real, IC, MA, P, K,
                                   V, PF, SRFN, Dens, preDens, PV, D, A, SD, S} <:
       FluidSystem{NDIMS, IC}
    initial_condition                 :: IC 
    mass                              :: MA     # Array{ELTYPE, 1}
    pressure                          :: P      # Array{ELTYPE, 1} 
    smoothing_kernel                  :: K 
    smoothing_length                  :: ELTYPE 
    acceleration                      :: SVector{NDIMS, ELTYPE} 
    viscosity                         :: V 
    pressure_acceleration_formulation :: PF
    transport_velocity                :: Nothing # TODO
    surface_normal_method             :: SRFN
    density                           :: Dens    # Array{ELTYPE, 1}
    predicted_density                 :: preDens # Array{ELTYPE, 1}
    v_adv                             :: PV      # Array{ELTYPE, NDIMS}
    d                                 :: D       # Array{ELTYPE, NDIMS}
    a                                 :: A       # Array{ELTYPE, 1}
    sum_dj                            :: SD      # Array{ELTYPE, NDIMS}
    s_term                            :: S       # Array{ELTYPE, 1}
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function ImplicitIncompressibleSPHSystem(initial_condition,
                                        smoothing_kernel, smoothing_length;
                                        viscosity=nothing,
                                        acceleration=ntuple(_ -> 0.0, ndims(smoothing_kernel)),
                                        reference_particle_spacing=0.0)

    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)
    n_particles = nparticles(initial_condition)

    mass = copy(initial_condition.mass)
    pressure = copy(initial_condition.pressure)

    if ndims(smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    # Make acceleration an SVector
    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    pressure_acceleration = pressure_acceleration_summation_density
    
    density = copy(initial_condition.density)
    predicted_density = zeros(ELTYPE, n_particles)
    a = zeros(ELTYPE, n_particles)
    d = zeros(ELTYPE, NDIMS, n_particles)
    v_adv = zeros(ELTYPE, NDIMS, n_particles)
    sum_dj = zeros(ELTYPE, NDIMS, n_particles)
    s_term = zeros(ELTYPE, n_particles)
    return ImplicitIncompressibleSPHSystem(initial_condition, mass, pressure,
                                       smoothing_kernel, smoothing_length,
                                       acceleration_, viscosity,
                                       pressure_acceleration, nothing,
                                       nothing, density, predicted_density, v_adv, d, a, sum_dj, s_term)
end


function reset_callback_flag!(system::ImplicitIncompressibleSPHSystem)
    return system
end


function Base.show(io::IO, system::ImplicitIncompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "ImplicitIncompressibleSPHSystem{", ndims(system), "}(")
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.acceleration)
    print(io, ", ", system.source_terms)
    print(io, ") with ", nparticles(system), " particles")
end


function Base.show(io::IO, ::MIME"text/plain", system::ImplicitIncompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "ImplicitIncompressibleSPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "viscosity", system.viscosity)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

@inline each_moving_particle(system :: ImplicitIncompressibleSPHSystem) = Base.OneTo(n_moving_particles(system))
@inline active_coordinates(u, system :: ImplicitIncompressibleSPHSystem) = current_coordinates(u, system)
@inline active_particles(system :: ImplicitIncompressibleSPHSystem) = eachparticle(system)

@propagate_inbounds function particle_pressure(v, system::ImplicitIncompressibleSPHSystem,
                                               particle)
    return system.pressure[particle]
end

@propagate_inbounds function particle_density(v, system::ImplicitIncompressibleSPHSystem, particle)
    return system.density[particle]
end

#TODO: Was machen mit dem Soundspeed? Wird für viscosity benötigt
@inline system_sound_speed(system::ImplicitIncompressibleSPHSystem) = 1000.0


@propagate_inbounds function predicted_velocity(system::ImplicitIncompressibleSPHSystem, particle)
    return extract_svector(system.v_adv, system, particle)
end

@propagate_inbounds function predicted_velocity(system::BoundarySystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@propagate_inbounds function get_updated_velocity(system::ImplicitIncompressibleSPHSystem, particle, v_updated)
    return extract_svector(v_updated, system, particle)
end

@propagate_inbounds function get_updated_velocity(system::BoundarySystem, particle, v_updated)
    return zero(SVector{ndims(system), eltype(system)})
end

@propagate_inbounds function get_current_pressure_value(system::ImplicitIncompressibleSPHSystem, particle, pressure)
   return pressure[particle]
end   

@propagate_inbounds function get_current_pressure_value(system::BoundarySystem, particle, pressure)
    return 0
 end    

 @propagate_inbounds function get_current_pressure_value_pair(system, neighbor_system::ImplicitIncompressibleSPHSystem, particle, neighbor, pressure)
    p_a = get_current_pressure_value(system, particle, pressure)
    p_b = get_current_pressure_value(neighbor_system, neighbor, pressure)
    return p_a, p_b
 end    

 @propagate_inbounds function get_current_pressure_value_pair(system, neighbor_system::BoundarySystem, particle, neighbor, pressure)
    p_a = get_current_pressure_value(system, particle, neighbor)
    return p_a, 0
 end    

@propagate_inbounds function get_predicted_density(system::ImplicitIncompressibleSPHSystem, particle, predicted_density)
    return predicted_density[particle]
end

@propagate_inbounds function get_predicted_density(system::BoundarySystem, particle, predicted_density)
    return 1000.0
end

@propagate_inbounds function get_predicted_density_pair(system, neighbor_system::ImplicitIncompressibleSPHSystem, particle, neighbor, predicted_density) 
    rho_a = get_predicted_density(system, particle, predicted_density)
    rho_b = get_predicted_density(neighbor_system, neighbor, predicted_density)
    return rho_a, rho_b
end

@propagate_inbounds function get_predicted_density_pair(system, neighbor_system::BoundarySystem, particle, neighbor, predicted_density) 
    rho_a = get_predicted_density(system, particle, predicted_density)
    return rho_a, 1000.0
end

@propagate_inbounds function get_d(system::ImplicitIncompressibleSPHSystem, d, particle)
    return extract_svector(d, system, particle)
end

@propagate_inbounds function get_d(system::BoundarySystem, d, particle)
    return extract_svector(d, system, particle)
end

@propagate_inbounds function get_sum_dj(system::ImplicitIncompressibleSPHSystem, sum_dj, particle)
    return extract_svector(sum_dj, system, particle)
end

@propagate_inbounds function get_sum_dj(system::BoundarySystem, sum_dj, particle)
    return extract_svector(sum_dj, system, particle)
end

# Calculates a summand for the calculation of the d_ii values 
function calculate_dii(system::ImplicitIncompressibleSPHSystem, m_b, rho_a, grad_kernel, time_step)
    return -time_step^2 * m_b / rho_a^2 * grad_kernel
end

# Calculates a summand for the calculation of the d_ii values 
function calculate_dii(system::BoundarySystem, m_b, rho_a, grad_kernel, time_step)
    return calculate_dii(system::BoundarySystem, system.boundary_model, m_b, rho_a, grad_kernel, time_step)
end

# Calculates a summand for the calculation of the d_ii values 
function calculate_dii(system, boundary_model::BoundaryModelDummyParticles, m_b, rho_a, grad_kernel, time_step)
    return calculate_dii(system, boundary_model, boundary_model.density_calculator, m_b, rho_a, grad_kernel, time_step)
end

# Calculates a summand for the calculation of the d_ii values 
function calculate_dii(system, boundary_model, density_calculator::PressureMirroring ,m_b, rho_a, grad_kernel, time_step)
    return -time_step^2 * m_b / rho_a^2 * grad_kernel #for pressure mirroring
end

# Calculates a summand for the calculation of the d_ii values 
function calculate_dii(system, boundary_model, density_calculator::PressureZeroing, m_b, rho_a, grad_kernel, time_step)
    return 0 #for pressure zeroing
end

# Calculates the d_ij value for a particle i and his neighbor j from the equation 9 in IHMSEN et al
function calculate_dij(system::ImplicitIncompressibleSPHSystem, particle, density, grad_kernel, time_step)
    return SVector(-time_step^2 * hydrodynamic_mass(system, particle) / density[particle]^2 * grad_kernel)
end

# Calculates the d_ij value for a particle i and his neighbor j from the equation 9 in IHMSEN et al
function calculate_dij(system::BoundarySystem, particle, density, grad_kernel, time_step)
    return SVector(-time_step^2 * hydrodynamic_mass(system, particle) / density[particle]^2 * grad_kernel)
end

# Calculates the sum d_ij*p_j over all j for a given particle i (IHMSEN et al section 3.1.1)
function calculate_sum_dj(system :: ImplicitIncompressibleSPHSystem, particle, density, pressure, grad_kernel, time_step)
    p_b = pressure[particle]
    dij = calculate_dij(system, particle, density, grad_kernel, time_step)
    return SVector(dij * p_b)
end

# Calculates the sum d_ij*p_j over all j for a given particle i (IHMSEN et al section 3.1.1)
function calculate_sum_dj(system :: BoundarySystem, particle, density, pressure, grad_kernel, time_step)
    return calculate_sum_dj(system, system.boundary_model, particle, density, pressure, grad_kernel, time_step)
end

# Calculates the sum d_ij*p_j over all j for a given particle i (IHMSEN et al section 3.1.1)
function calculate_sum_dj(system, boundary_model::BoundaryModelDummyParticles,  particle, density, pressure, grad_kernel, time_step)
    return calculate_sum_dj(system, boundary_model, boundary_model.density_calculator, particle, density, pressure, grad_kernel, time_step)
end

# Calculates the sum d_ij*p_j over all j for a given particle i (IHMSEN et al section 3.1.1)
function calculate_sum_dj(system, boundary_model, density_calculator::PressureMirroring, particle, density, pressure, grad_kernel, time_step)
    #pressure = particle_pressure(0 ,system, particle) #TODO In case of using different density calculators, the boundary particles have own pressure values which just need to be determined in this line, and then they contribute to the pressure value of the fluid particles 
    return zero(SVector{ndims(system), eltype(system)})
end

# Calculates the sum d_ij*p_j over all j for a given particle i (IHMSEN et al section 3.1.1)
function calculate_sum_dj(system, boundary_model, density_calculator::PressureZeroing, particle, density, pressure, grad_kernel, time_step)
    #pressure = particle_pressure(0, system, particle) #TODO s.o.
    return zero(SVector{ndims(system), eltype(system)})
end

# Calculates the sum term in equation (13) from IHMSEN et al
function calculate_sum_term(system, neighbor_system:: ImplicitIncompressibleSPHSystem, particle, neighbor, density, pressure, sum_dj, d, grad_kernel_ab, grad_kernel_ba, time_step)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    sum_da = get_sum_dj(system, sum_dj, particle)
    d_b = get_d(neighbor_system, d, neighbor)
    p_a = pressure[particle]
    p_b = pressure[neighbor]
    sum_db = get_sum_dj(neighbor_system, sum_dj, neighbor)
    dba = -time_step^2 * hydrodynamic_mass(system, particle) / density[particle]^2 * grad_kernel_ba
    return m_b * dot(sum_da - d_b * p_b - (sum_db - dba * p_a), grad_kernel_ab)
end

# Calculates the sum term in equation (13) from IHMSEN et al
function calculate_sum_term(system, neighbor_system:: BoundarySystem, particle, neighbor, density, pressure, sum_dj, d, grad_kernel_ab, grad_kernel_ba, time_step)
    sum_da = get_sum_dj(system, sum_dj, particle)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    return m_b * dot(sum_da, grad_kernel_ab)
end


# Calculates the pressure values by solving a linear system with a relaxed jacobi scheme
function update_quantities!(system::ImplicitIncompressibleSPHSystem, v, u,
                            v_ode, u_ode, semi, t)

    # fixed time step size 
    time_step = 0.001
    
    density = system.density 
    predicted_density = system.predicted_density 
    d = system.d
    a = system.a
    v_adv = system.v_adv
    pressure = system.pressure
    v_updated = copy(v_adv)
    #v_updated = zeros(ELTYPE, ndims(system), n_particles)
    updated_density = copy(density)

    set_zero!(d)
    sound_speed = system_sound_speed(system)
    set_zero!(a)
    summation_density!(system, semi,  u, u_ode, density)
    set_zero!(predicted_density)
    set_zero!(updated_density)


    v_particle_system = wrap_v(v_ode, system, semi)
    v_adv .= v_particle_system

    # Calculate the predicted velocity by adding all non-pressure accelerations
    @trixi_timeit timer() "Predict Advection" foreach_system(semi) do neighbor_system
        # get neighbor system u and v values 
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        # get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
        # get neighborhood_search
        nhs = get_neighborhood_search(system, neighbor_system, semi)
        
        foreach_point_neighbor(system, neighbor_system,
                                system_coords, neighbor_system_coords,
                                nhs;
                                points=each_moving_particle(system)) do particle,
                                                                        neighbor,
                                                                        pos_diff,
                                                                        distance
            #TODO: Was tun mit dem sound speed?

            m_a = hydrodynamic_mass(system, particle)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)

            rho_a = @inbounds particle_density(v_particle_system, system, particle)
            rho_b = @inbounds particle_density(v_neighbor_system, neighbor_system, neighbor)
 
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)

            dv_viscosity_ = @inbounds dv_viscosity(system, neighbor_system,
                                v_particle_system, v_neighbor_system,
                                particle, neighbor, pos_diff, distance,
                                sound_speed, m_a, m_b, rho_a, rho_b,
                                grad_kernel)

            # calculate d_ii with formula in eq. 9 from IHMSEN et al
            for i in 1:ndims(system)
                d[i, particle] += calculate_dii(neighbor_system, m_b, rho_a, grad_kernel[i], time_step)
            end

            # calculate predicted velocities
            for i in 1:ndims(system)
                v_adv[i, particle] += time_step * (dv_viscosity_[i] + system.acceleration[i])
            end
        end
    end

    # Calculation of the a_ii-values according to equation 12 from IHMSEN et al
    @trixi_timeit timer() "Calculate matrix diagonal a_ii" foreach_system(semi) do neighbor_system
        # set neighbor system u and v values 
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        # get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
        # get neighborhood_search
        nhs = get_neighborhood_search(system, neighbor_system, semi)
        
        foreach_point_neighbor(system, neighbor_system,
                                system_coords, neighbor_system_coords,
                                nhs;
                                points=each_moving_particle(system)) do particle,
                                                                        neighbor,
                                                                        pos_diff,
                                                                        distance

            # set pressure p0 to a half of the previous/current pressure value
            pressure[particle] = 0.5 * pressure[particle]

            grad_kernel_ji = smoothing_kernel_grad(system, -pos_diff, distance)
            # compute dji
            dji = calculate_dij(neighbor_system, neighbor, density, grad_kernel_ji, time_step)
                               
            m_b = hydrodynamic_mass(neighbor_system, neighbor)

            # calculate a_ii values
            a[particle] += m_b * dot((get_d(system, d, particle) - dji), smoothing_kernel_grad(system, pos_diff, distance))
        end
    end

    foreach_system(semi) do neighbor_system
        # get neighbor system u and v values 
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        # get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
        # get neighborhood_search
        nhs = get_neighborhood_search(system, neighbor_system, semi)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords, nhs,
                                points=each_moving_particle(system)) do particle, neighbor,
                                                                        pos_diff, distance
            
            # Calculate predicted density through predicted velocity
            v_adv_diff = predicted_velocity(system, particle) - predicted_velocity(neighbor_system, neighbor)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)
            
            predicted_density[particle] = density[particle] + time_step * m_b * dot(v_adv_diff, grad_kernel)
        end
    end

    # relaxiaion parameter
    w = 0.5

    sum_dj = system.sum_dj
    s_term = system.s_term
    sum_dj .= 0.0
    #updated_density .= predicted_density

    # Calculate pressure values with iterative pressure solver (relaxed jacobi scheme)
    l = 0
    while l < 15 #TODO: Abbruchbedingung
        set_zero!(sum_dj)
        updated_density .= predicted_density
        v_updated .= v_adv
        @trixi_timeit timer() "Pressure solve - calculate Sum over d_j's" foreach_system(semi) do neighbor_system
            # Get neighbor system u and v values 
            u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
            # get coordinates
            system_coords = current_coordinates(u, system)
            neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
            # get neighborhood_search
            nhs = get_neighborhood_search(system, neighbor_system, semi)

            foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords, nhs;
            points=each_moving_particle(system)) do particle,
                                                    neighbor,
                                                    pos_diff,
                                                    distance

                grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)
                sum_dj_ = calculate_sum_dj(neighbor_system, neighbor, density, pressure, grad_kernel, time_step)  # only gives a value for fluid particles (0 for boundary particles) 
                for i in 1:ndims(system)
                    sum_dj[i, particle] += sum_dj_[i] 
                end
            end
        end

        s_term .= 0.0
        @trixi_timeit timer() "Pressure solve - calculate pressure values" foreach_system(semi) do neighbor_system
            # Get neighbor system u and v values 
            u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
            # get coordinates
            system_coords = current_coordinates(u, system)
            neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
            # get neighborhood_search
            nhs = get_neighborhood_search(system, neighbor_system, semi)
            @trixi_timeit timer() "calc s_term loop" foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords, nhs;
                                    points=each_moving_particle(system)) do particle,
                                                                            neighbor,
                                                                            pos_diff,
                                                                            distance
                
                grad_kernel_ab = smoothing_kernel_grad(system, pos_diff, distance)
                grad_kernel_ba = smoothing_kernel_grad(system, -pos_diff, distance)
                s_term[particle] += calculate_sum_term(system, neighbor_system, particle, neighbor, density, pressure, sum_dj, d, grad_kernel_ab, grad_kernel_ba, time_step)                      
            end
        end
        rest_density = 1000.0
        # pressure update
        for particle in eachparticle(system)
            # Removing instability by avoiding to divide through very low numbers for a
            # This is not mentioned in the paper but it's what they do in SPlisHSPlasH
            if abs(a[particle]) > 1e-9
                pressure[particle] = max((1-w) * pressure[particle] + w * 1/a[particle] * (rest_density - predicted_density[particle] - s_term[particle]), 0) #version with pressure clamping (no negative pressure values)
            else
                pressure[particle] = 0
            end
        end
        #=
        #TODO: Abbruchbedingung
            foreach_system(semi) do neighbor_system
            # Get neighbor system u and v values 
            u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
            # get coordinates
            system_coords = current_coordinates(u, system)
            neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
            # get neighborhood_search
            nhs = get_neighborhood_search(system, neighbor_system, semi)

                foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords, nhs;
                points=each_moving_particle(system)) do particle,
                                                        neighbor,
                                                        pos_diff,
                                                        distance
                    
                    grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)

                    m_a = hydrodynamic_mass(system, particle)
                    m_b = hydrodynamic_mass(neighbor_system, neighbor)

                    p_a, p_b = get_current_pressure_value_pair(system, neighbor_system, particle, neighbor, pressure)

                    rho_a, rho_b  = get_predicted_density_pair(system, neighbor_system, particle, neighbor, predicted_density) 

                    dv_pressure = pressure_acceleration(system, neighbor_system, neighbor, m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                                        distance, grad_kernel, nothing)
                    # update velocity

                    for i in 1:ndims(system)
                        @inbounds v_updated[i, particle] += time_step * dv_pressure[i]

                    end
                end
            end
            foreach_system(semi) do neighbor_system
                # Get neighbor system u and v values 
                u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
                # get coordinates
                system_coords = current_coordinates(u, system)
                neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
                # get neighborhood_search
                nhs = get_neighborhood_search(system, neighbor_system, semi)
    
                foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords, nhs;
                points=each_moving_particle(system)) do particle,
                                                        neighbor,
                                                        pos_diff,
                                                        distance
                    #TODO update velocity

                    v_updated_diff = get_updated_velocity(system, particle, v_updated) - get_updated_velocity(neighbor_system, neighbor, v_updated)
                    m_b = hydrodynamic_mass(neighbor_system, neighbor)
                    grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)
                    
                    updated_density[particle] += time_step * m_b * dot(v_updated_diff, grad_kernel)
                end
                                    
            end
            mean_updated_density = sum(updated_density) / nparticles(system)
            density_deviation_abs = mean_updated_density - rest_density
            density_deviation_rel = density_deviation_abs / rest_density
            println("l = ", l)
            println("Mean:", mean_updated_density)
            println("Rest_density:", rest_density)
            println("Absolut Deviation:", density_deviation_abs)
            println("Relative Deviation:", density_deviation_rel)
            #TODO calculate deviation
            =#
        l += 1
    end
    return system
end



function write_v0!(v0, system::ImplicitIncompressibleSPHSystem)
    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(system.initial_condition.velocity)
    copyto!(v0, indices, system.initial_condition.velocity, indices)
    return v0
end
