include("../validation_util.jl")

############################################################################################
# Case "Elastic plate under a hydrostatic water column" as described in
# "A fluid–structure interaction model for free-surface flows and flexible structures
# using smoothed particle hydrodynamics on a GPU" by J. O'Connor and B.D. Rogers
# published in Journal of Fluids and Structures
# https://doi.org/10.1016/j.jfluidstructs.2021.103312
############################################################################################

using TrixiParticles
using OrdinaryDiffEq
using JSON

# ============================================================================
# Experiment Parameters
# ============================================================================
n_particles_plate_y = 3                # paper uses 5 to 40 (3 for short runtime in CI)
boundary_layers = 3
spacing_ratio = 1

gravity = 9.81
tspan = (0.0, 0.3)     # recommended: run at least to 0.5 (paper runs to 1.0); using 0.3 for speed in CI

initial_fluid_size = (1.0, 2.0)
plate_size = (1.0, 0.05)

fluid_density = 1000.0
solid_density = 2700.0

E = 67.5e9
nu = 0.3
sound_speed = 50

# Particle spacings (solid and fluid share same spacing)
solid_particle_spacing = plate_size[2] / (n_particles_plate_y - 1)
fluid_particle_spacing = solid_particle_spacing
boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# Analytical solution (constant)
D = E * plate_size[2]^3 / (12 * (1 - nu^2))
analytical_value = -0.0026 * gravity *
                   (fluid_density * initial_fluid_size[2] +
                    solid_density * plate_size[2]) / D

# Filename suffix based on resolution (here, number of particles in y)
formatted_string = replace(string(n_particles_plate_y), "." => "")

# ============================================================================
# Geometry Definitions: Tank, Beam, and Fixed Particles
# ============================================================================
n_particles_plate_x = round(Int, plate_size[1] / solid_particle_spacing + 1)
n_particles_per_dimension = (n_particles_plate_x, n_particles_plate_y)

plate = RectangularShape(solid_particle_spacing, n_particles_per_dimension,
                         (0.0, -plate_size[2]), density=solid_density, tlsph=true)

fixed_particles_1 = RectangularShape(solid_particle_spacing, (3, n_particles_plate_y),
                                     (-3 * solid_particle_spacing, -plate_size[2]),
                                     density=solid_density, tlsph=true)
fixed_particles_2 = RectangularShape(solid_particle_spacing, (3, n_particles_plate_y),
                                     (plate_size[1] + solid_particle_spacing,
                                      -plate_size[2]), density=solid_density, tlsph=true)
fixed_particles = union(fixed_particles_1, fixed_particles_2)

solid_geometry = union(plate, fixed_particles)

# ============================================================================
# Smoothing Kernel, Boundary, and Related Quantities
# ============================================================================
smoothing_kernel = WendlandC2Kernel{2}()
smoothing_length_solid = 2 * sqrt(2) * solid_particle_spacing
smoothing_length_fluid = 2 * sqrt(2) * fluid_particle_spacing

hydrodynamic_densities = fluid_density * ones(size(solid_geometry.density))
hydrodynamic_masses = hydrodynamic_densities * solid_particle_spacing^ndims(solid_geometry)
boundary_density_calculator = AdamiPressureExtrapolation()

# ============================================================================
# Sensor Function (for Postprocessing)
# ============================================================================
function y_deflection(v, u, t, system::TotalLagrangianSPHSystem)
    # Choose the particle in the middle of the beam along x.
    particle_id = Int(n_particles_per_dimension[1] * (n_particles_plate_y + 1) / 2 -
                      (n_particles_per_dimension[1] + 1) / 2 + 1)
    return TrixiParticles.current_coords(u, system, particle_id)[2] + plate_size[2] / 2
end
y_deflection(v, u, t, system) = nothing

# ============================================================================
# Run Simulations and Compute Errors
# ============================================================================
errors = Dict{String, Tuple{Float64, Float64}}()

for method in ["edac", "wcsph"]
    local_state_equation = method == "edac" ? nothing :
                           StateEquationCole(; sound_speed, reference_density=fluid_density,
                                             exponent=7, clip_negative_pressure=false)

    tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, (plate_size[1], 3.0),
                           min_coordinates=(0.0, fluid_particle_spacing / 2),
                           fluid_density, n_layers=boundary_layers,
                           spacing_ratio=spacing_ratio,
                           faces=(true, true, false, false),
                           acceleration=(0.0, -gravity),
                           state_equation=local_state_equation)

    if method == "edac"
        fluid_system = EntropicallyDampedSPHSystem(tank.fluid, smoothing_kernel,
                                                   smoothing_length_fluid, sound_speed,
                                                   acceleration=(0.0, -gravity))
    else
        fluid_density_calculator = ContinuityDensity()
        density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
        fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                                   local_state_equation, smoothing_kernel,
                                                   smoothing_length_fluid,
                                                   density_diffusion=density_diffusion,
                                                   acceleration=(0.0, -gravity))
    end

    boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                                 state_equation=local_state_equation,
                                                 boundary_density_calculator,
                                                 smoothing_kernel, smoothing_length_fluid)
    boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)
    boundary_model_solid = BoundaryModelDummyParticles(hydrodynamic_densities,
                                                       hydrodynamic_masses,
                                                       state_equation=local_state_equation,
                                                       boundary_density_calculator,
                                                       smoothing_kernel,
                                                       smoothing_length_solid)
    solid_system = TotalLagrangianSPHSystem(solid_geometry, smoothing_kernel,
                                            smoothing_length_solid,
                                            E, nu, boundary_model=boundary_model_solid,
                                            n_fixed_particles=nparticles(fixed_particles),
                                            acceleration=(0.0, -gravity))

    semi = Semidiscretization(solid_system, fluid_system, boundary_system)
    ode = semidiscretize(semi, tspan)

    pp_filename = "validation_result_hyd_" * method * "_" * formatted_string *
                  "_postprocess.jl"
    pp = PostprocessCallback(; dt=0.0025, filename=pp_filename, y_deflection,
                             kinetic_energy)
    info_callback = InfoCallback(interval=1000)
    saving_callback = SolutionSavingCallback(dt=0.5, prefix="")
    callbacks = CallbackSet(info_callback, saving_callback, pp)
    sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks)

    # Load the run JSON file and add the analytical solution as a single point.
    run_filename = joinpath("out", pp_filename)
    run_data = JSON.parsefile(run_filename)
    run_data["analytical_solution"] = Dict(
        "n_values" => 1,
        "time" => [tspan[2]],
        "values" => [analytical_value],
        "datatype" => "Float64",
        "type" => "point"
    )
    open(run_filename, "w") do io
        JSON.print(io, run_data, 2)
    end

    # Compute errors using data from t ∈ [0.25, 0.5] for the sensor starting with "y_deflection".
    sensor_key = first(filter(k -> startswith(k, "y_deflection"), keys(run_data)))
    time_vals = run_data[sensor_key]["time"]
    sim_vals = run_data[sensor_key]["values"]
    inds = findall(t -> 0.25 <= t <= 0.5, time_vals)
    avg_sim = sum(sim_vals[inds]) / length(sim_vals[inds])
    abs_error = abs(avg_sim - analytical_value)
    rel_error = abs_error / abs(analytical_value)
    errors[method] = (abs_error, rel_error)
end
