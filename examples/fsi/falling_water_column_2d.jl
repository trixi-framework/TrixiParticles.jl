# ==========================================================================================
# 2D Falling Water Column on an Elastic Beam (FSI)
#
# This example simulates a column of water falling under gravity and impacting
# an elastic beam, which is fixed at one end (cantilever).
# It demonstrates Fluid-Structure Interaction where the fluid deforms the solid structure.
## ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Solid Beam Parameters (Loaded from `oscillating_beam_2d.jl`)
# The `trixi_include` below will define `particle_spacing` (for solid), `material`,
# `solid` (particles), `fixed_particles`, `nparticles`, etc.
# We need to specify the thickness and number of particles for the beam here.
beam_thickness_param = 0.05 # meters
beam_num_particles_y_param = 5 # Number of particles across beam thickness (y-direction for beam)

# Fluid Parameters
# Fluid resolution is set relative to the solid particle spacing from the included file.
# This `fluid_particle_spacing` will be defined *after* `particle_spacing` (solid) is loaded.
# fluid_particle_spacing_factor = 3.0 # fluid_dp = factor * solid_dp

# Physical Parameters
gravity_magnitude = 9.81
system_acceleration_vec = (0.0, -gravity_magnitude)
simulation_tspan = (0.0, 1.0) # seconds

# Fluid Properties
fluid_density_ref = 1000.0 # kg/m^3
# `initial_fluid_height` will be from `initial_fluid_size` defined later.
# sound_speed_fluid = 10 * sqrt(gravity_magnitude * initial_fluid_height)
fluid_exponent_eos = 7

# ------------------------------------------------------------------------------
# Load Solid Beam Setup
# ------------------------------------------------------------------------------
# Include the oscillating beam setup to define the solid structure.
# `sol=nothing` prevents the included solid-only simulation from running.
# Variables like `solid`, `fixed_particles`, `material`, `particle_spacing` (solid),
# `solid_smoothing_kernel`, `solid_smoothing_length` will be available after this.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "solid", "oscillating_beam_2d.jl"),
              thickness=beam_thickness_param,
              n_particles_y=beam_num_particles_y_param,
              sol=nothing, ode=nothing) # Don't run, just load setup

# Now that `particle_spacing` (solid) is loaded, define fluid particle spacing.
fluid_particle_spacing = 3.0 * particle_spacing # Fluid particles are 3x larger than solid

# ------------------------------------------------------------------------------
# Experiment Setup: Fluid Column
# ------------------------------------------------------------------------------
# Dimensions and initial position of the water column
initial_fluid_width = 0.525
initial_fluid_height = 1.0125
initial_fluid_size = (initial_fluid_width, initial_fluid_height)

# Position the fluid column. Ensure it's above the beam.
# Beam's fixed end is at x=0, extends in +x. Beam thickness is `beam_thickness_param`.
# Assuming beam is thin and near y=0. Fluid needs to be placed above this.
fluid_initial_min_x = 0.1
fluid_initial_min_y = 0.2 # Ensure this is above the beam's initial position
fluid_initial_position = (fluid_initial_min_x, fluid_initial_min_y)

# Sound speed calculation now that `initial_fluid_height` is known.
sound_speed_fluid = 10 * sqrt(gravity_magnitude * initial_fluid_height)
fluid_state_equation = StateEquationCole(sound_speed=sound_speed_fluid,
                                         reference_density=fluid_density_ref,
                                         exponent=fluid_exponent_eos)

# Create fluid particles for the water column
fluid_column_particles = RectangularShape(fluid_particle_spacing,
                                          round.(Int, initial_fluid_size ./ fluid_particle_spacing),
                                          fluid_initial_position,
                                          density=fluid_density_ref)

# ------------------------------------------------------------------------------
# Fluid System Setup (Weakly Compressible SPH)
# ------------------------------------------------------------------------------
fluid_smoothing_length = 1.2 * fluid_particle_spacing # Common choice for WCSPH
fluid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
fluid_viscosity_model = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(fluid_column_particles,
                                           fluid_density_calculator,
                                           fluid_state_equation,
                                           fluid_smoothing_kernel,
                                           fluid_smoothing_length,
                                           viscosity=fluid_viscosity_model,
                                           acceleration=system_acceleration_vec,
                                           reference_particle_spacing=fluid_particle_spacing)

# ------------------------------------------------------------------------------
# Solid System Setup (Elastic Beam - Reconfigured for FSI)
# ------------------------------------------------------------------------------
# The `solid` ParticleSet, `material` properties (E, nu), `solid_smoothing_kernel`,
# `solid_smoothing_length` are inherited from the included `oscillating_beam_2d.jl`.

# FSI Boundary Model for the Solid Beam (Monaghan-Kajtar)
# This model defines how fluid pressure is applied to the solid.
k_factor_monaghan_kajtar = gravity_magnitude * initial_fluid_height # Characteristic pressure scale
# Spacing ratio for Monaghan-Kajtar model: fluid_dp / solid_dp
spacing_ratio_fluid_solid_mk = fluid_particle_spacing / particle_spacing # `particle_spacing` is solid's here

# Hydrodynamic properties needed for the FSI boundary model on the solid
hydrodynamic_densities_for_beam = fluid_density_ref .* ones(size(solid.density))
# Mass for Monaghan-Kajtar is typically particle volume * fluid density.
hydrodynamic_masses_for_beam = hydrodynamic_densities_for_beam .* particle_spacing^2 # solid_dp^2

solid_fsi_boundary_model_beam = BoundaryModelMonaghanKajtar(k_factor_monaghan_kajtar,
                                                            spacing_ratio_fluid_solid_mk,
                                                            particle_spacing, # Solid particle spacing
                                                            hydrodynamic_masses_for_beam)

# Re-define the solid system from `oscillating_beam_2d.jl` to include the FSI boundary model
# and ensure correct acceleration and fixed particles.
# `nparticles(fixed_particles)` gets the number of fixed particles from the included file.
solid_beam_fsi_system = TotalLagrangianSPHSystem(solid, # `solid` particles from included file
                                                 solid_smoothing_kernel, # from included file
                                                 solid_smoothing_length, # from included file
                                                 material.E, material.nu, # from included file
                                                 boundary_model=solid_fsi_boundary_model_beam,
                                                 n_fixed_particles=nparticles(fixed_particles), # from included file
                                                 acceleration=system_acceleration_vec, # Consistent gravity
                                                 reference_particle_spacing=particle_spacing) # Solid's particle_spacing

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
# Note: This setup does not include fixed tank walls for the fluid,
# so the fluid will fall and spread freely after impacting the beam.
semi = Semidiscretization(fluid_system, solid_beam_fsi_system,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, simulation_tspan)

# Callbacks
info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.005, prefix="fsi_water_beam_2d")
callbacks = CallbackSet(info_callback, saving_callback)

# Solve the ODE system.
sol = solve(ode, RDPK3SpFSAL49(), # Solver choice from original file
            abstol=1e-6,
            reltol=1e-4,
            dtmax=1e-3, # Limit stepsize for stability
            save_everystep=false,
            callback=callbacks)
