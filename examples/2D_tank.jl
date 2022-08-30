using Pixie
using OrdinaryDiffEq
setup = ["tank_2D",                   
        "BC_crespo"]  
width = 2.0
water_height = 0.9
container_height = 1.0
particle_spacing = 0.02
smoothing_length = 1.2 * particle_spacing

mass = 1000 * particle_spacing^2

# Particle data
n_particles_per_dimension = (Int((width) / particle_spacing),
                             Int(water_height / particle_spacing))
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = mass * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = x * particle_spacing
    particle_coordinates[2, particle] = y * particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
end

# Boundary particle data
boundary_particle_spcng = smoothing_length/1.3
function init_boundaries!(bound_spcng, HEIGHT, WIDTH; staggered=false)
    i = 0
    if staggered
        x_points_staggered  = collect(-bound_spcng*3/2:bound_spcng:WIDTH+bound_spcng*6/2)
        y_points_staggered  = collect(-bound_spcng*3/2:bound_spcng:HEIGHT+bound_spcng*6/2)
    else
        x_points_staggered         = Vector{Float64}(undef, 0)
        y_points_staggered         = Vector{Float64}(undef, 0)
    end
    x_points = collect(-bound_spcng:bound_spcng:WIDTH+2*bound_spcng)
    y_points = collect(-bound_spcng:bound_spcng:HEIGHT+2*bound_spcng)
    array_length       = length(x_points)+length(y_points)*2 +length(x_points_staggered)+length(y_points_staggered)*2
    boundary           = Array{Float64, 2}(undef, 2, array_length);
    for x in x_points
        i += 1
            boundary[1, i] = x
            boundary[2, i] = y_points[1]
    end
    for x = [x_points[1]; x_points[end]]
        for y in y_points
            i += 1
                boundary[1, i] = x
                boundary[2, i] = y
        end
    end

    for x in x_points_staggered
        i += 1
            boundary[1, i] = x
            boundary[2, i] =  y_points_staggered[1]
    end
    for x = [x_points_staggered[1]; x_points_staggered[end]]
        for y in y_points_staggered
            i += 1
                boundary[1, i] = x
                boundary[2, i] = y
        end
    end
    return boundary
end
boundary_coordinates = init_boundaries!(boundary_particle_spcng, container_height, width; staggered=true);
boundary_masses = mass * ones(Float64, size(boundary_coordinates, 2))

c = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)

smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

K = 9.81 * water_height
boundary_conditions = Pixie.BoundaryConditionCrespo(boundary_coordinates, boundary_masses, c, neighborhood_search=SpatialHashingSearch{2}(search_radius))

# Create semidiscretization
semi = SPHSemidiscretization{2}(particle_masses,
                                ContinuityDensity(), state_equation,
                                smoothing_kernel, smoothing_length,
                                viscosity=ArtificialViscosityMonaghan(0.5, 0.0),
                                boundary_conditions=boundary_conditions,
                                gravity=(0.0, -9.81),
                                neighborhood_search=SpatialHashingSearch{2}(search_radius))

tspan = (0.0, 1.0)
ode = semidiscretize(semi, particle_coordinates, particle_velocities, particle_densities, tspan)

alive_callback = AliveCallback(alive_interval=10)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
#sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()), callback=alive_callback);
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()), dt=1e-5, saveat=0.02, callback=alive_callback);
