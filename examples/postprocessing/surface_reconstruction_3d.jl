# ==========================================================================================
# 3D Surface Reconstruction Example
#
# This example demonstrates the `SurfaceReconstructionCallback` in TrixiParticles.jl,
# which reconstructs a closed, volume-corrected water surface from the fluid particles
# of a 3D dam-break simulation and writes it as a VTK PolyData (`.vtp`) time series.
# After the simulation, the final frame is reconstructed once more offline with
# `reconstruct_surface`.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.05

boundary_layers = 3
spacing_ratio = 1
boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 0.3)

initial_fluid_size = (0.8, 0.5, 0.4)
tank_size = (floor(1.4 / boundary_particle_spacing) * boundary_particle_spacing, 0.8, 0.5)

fluid_density = 1000.0

state_equation = StateEquationCole(; reference_density=fluid_density,
                                   sound_speed=10 * sqrt(gravity * 0.5), exponent=7)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density;
                       n_layers=boundary_layers, spacing_ratio,
                       acceleration=(0.0, -gravity, 0.0))

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.5 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{3}()

# `ContinuityDensity` stores the density in the solution vector, so the per-particle
# volumes `m_i / rho_i` of the reconstruction are always up to date.
fluid_density_calculator = ContinuityDensity()

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel, smoothing_length,
                                           density_calculator=fluid_density_calculator,
                                           state_equation,
                                           acceleration=(0.0, -gravity, 0.0))

# ==========================================================================================
# ==== Boundary
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             smoothing_length; state_equation,
                                             clip_negative_pressure=true,
                                             reference_particle_spacing=fluid_particle_spacing)

boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

# ==========================================================================================
# ==== Surface Reconstruction
# The production configuration: voxel size h/2, Gaussian width 0.9h, isovalue correction
# to 0.1% within at most 8 iterations, warm-started between frames. Passing `tank_size`
# pins the reconstruction grid, so the internal workspace is reused every frame.
# Isolated droplets and splashes below the grid resolution are filled with
# volume-equivalent spheres by the sparse-component fallback.
reconstruction = SurfaceReconstruction(; particle_spacing=fluid_particle_spacing,
                                       tank_size=tank_size,
                                       sparse_component_fallback=true)

# Reconstruct every 50 accepted time steps into `out/surface_fluid_1_*.vtp`
# (collected in `out/surface_fluid_1.pvd`) together with per-vertex normals.
surface_callback = SurfaceReconstructionCallback(reconstruction, semi; interval=50,
                                                 output_directory="out",
                                                 prefix="surface")

info_callback = InfoCallback(interval=100)
callbacks = CallbackSet(info_callback, surface_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# ==========================================================================================
# ==== Offline Reconstruction
# Reconstruct the final frame once more (e.g., for statistics or custom postprocessing)
# and inspect the result. This is skipped when the solve itself was suppressed.
if !isnothing(sol)
    mesh, stats = reconstruct_surface(semi, sol)
    println("Reconstructed surface: ", length(mesh.vertices), " vertices, ",
            length(mesh.faces), " faces, effective isovalue ",
            round(stats.effective_isovalue; digits=4), ", enclosed volume ",
            round(stats.volume; digits=6), " m^3")
end
