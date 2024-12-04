using TrixiParticles
using OrdinaryDiffEq
using Random  # Import Random module for generating random positions and radii

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.00125

boundary_layers = 4
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 0.5)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.0, 0.0)
tank_size = (2.0, 0.5)

fluid_density = 1000.0
sound_speed = 100
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

box = RectangularShape(fluid_particle_spacing, (Int(0.25/fluid_particle_spacing), Int(0.1/fluid_particle_spacing)), (0.5*tank_size[1], 0.0),
                       density=fluid_density)

# Sphere radius range
sphere_radius_min = 0.005  # Minimum radius
sphere_radius_max = 0.025  # Maximum radius
sphere_radius_range = (sphere_radius_min, sphere_radius_max)

# Set random seed for reproducibility (optional)
Random.seed!(1234)

# Define the tank boundaries to avoid placing spheres too close to the walls
x_range = (0.0, tank_size[1])
y_range = (0.1, tank_size[2]+0.1)

# Function to generate non-overlapping spheres with random radii
function generate_non_overlapping_spheres(num_spheres, radius_range, x_range, y_range)
    spheres = []
    positions_radii = []
    attempt = 0
    max_attempts = 1000  # Prevent infinite loops
    while length(spheres) < num_spheres && attempt < max_attempts
        attempt += 1
        # Generate random radius within the specified range
        radius = rand() * (radius_range[2] - radius_range[1]) + radius_range[1]
        # Adjust x and y ranges to account for sphere radius and avoid walls
        x_min = x_range[1] + radius + 0.01  # Add small buffer to avoid wall overlap
        x_max = x_range[2] - radius - 0.01
        y_min = y_range[1] + radius + 0.01
        y_max = y_range[2] - radius - 0.01
        if x_min >= x_max || y_min >= y_max
            continue  # Skip if adjusted ranges are invalid
        end
        # Generate random position within adjusted ranges
        position = (rand() * (x_max - x_min) + x_min,
                    rand() * (y_max - y_min) + y_min)
        # Check for overlap with existing spheres
        no_overlap = true
        for (pos_existing, radius_existing) in positions_radii
            dist = sqrt((position[1] - pos_existing[1])^2 + (position[2] - pos_existing[2])^2)
            if dist < (radius + radius_existing) + 0.01  # Add small buffer
                no_overlap = false
                break
            end
        end
        if no_overlap
            # Create the sphere
            sphere = SphereShape(fluid_particle_spacing, radius, position,
                                 fluid_density, sphere_type=VoxelSphere(), velocity=(0.0, -1.0))
            push!(spheres, sphere)
            push!(positions_radii, (position, radius))
        end
    end
    if length(spheres) < num_spheres
        error("Could not generate the required number of non-overlapping spheres within the maximum attempts.")
    end
    return spheres
end

num_spheres = 20
spheres = generate_non_overlapping_spheres(num_spheres, sphere_radius_range, x_range, y_range)

# Combine all spheres using the union function
combined_spheres = reduce(union, spheres)

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 3.5 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()

nu = 0.0025
viscosity = ViscosityMorris(nu=nu)
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

sphere_surface_tension = EntropicallyDampedSPHSystem(combined_spheres, fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     sound_speed, viscosity=viscosity,
                                                     density_calculator=ContinuityDensity(),
                                                     acceleration=(0.0, -gravity),
                                                     reference_particle_spacing=fluid_particle_spacing,
                                                     surface_tension=SurfaceTensionMorris(surface_tension_coefficient=0.0728),
                                                     surface_normal_method=ColorfieldSurfaceNormal(ideal_density_threshold=0.9,
                                                                                      interface_threshold=0.001, boundary_contact_threshold=0.0))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
wall_viscosity = nu
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length,
                                             viscosity=ViscosityAdami(nu=wall_viscosity),
                                             correction=MixedKernelGradientCorrection())


boundary_system = BoundarySPHSystem(tank.boundary, boundary_model,
                                    surface_normal_method=StaticNormals((0.0, 1.0)))

box_model = BoundaryModelDummyParticles(box.density, box.mass,
                                    state_equation=state_equation,
                                    boundary_density_calculator,
                                    fluid_smoothing_kernel, fluid_smoothing_length,
                                    viscosity=ViscosityAdami(nu=wall_viscosity),
                                    correction=MixedKernelGradientCorrection())

box_system = BoundarySPHSystem(box, box_model,
                                    surface_normal_method=StaticNormals((0.0, 1.0)))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(sphere_surface_tension, boundary_system, box_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.01, output_directory="out",
                                         prefix="", write_meta_data=true)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error-based) time step size control.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6
            reltol=1e-4, # Default reltol is 1e-3
            dt=1e-6,
            save_everystep=false, callback=callbacks);
