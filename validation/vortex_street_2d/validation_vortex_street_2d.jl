using TrixiParticles

tspan = (0.0, 20.0)

# In Tafuni et al. (2018), the resolution is `0.01` (5M particles).
# Results in 1.3M particles and acceptable results compared to Tafuni et al. (2018).
# resolution_factor = 0.02 # (runtime: ~6-10h)
# Results in 100k particles and much noisier results compared to Tafuni et al. (2018).
resolution_factor = 0.05

reynolds_number = 200
cylinder_diameter = 0.1
domain_size = (25 * cylinder_diameter, 20 * cylinder_diameter)

open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method=ZerothOrderMirroring())

# The vortex shedding is chaotic, so the results depend on the order of the floating point
# operations. Use `SerialUpdate()` to obtain consistent results with multiple threads.
update_strategy = ParallelUpdate()

# Set this to a GPU backend like `CUDABackend()` to run the simulation on a GPU.
parallelization_backend = PolyesterBackend()

# Import variables into scope without running the simulation.
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              reynolds_number=reynolds_number, saving_callback=nothing,
              open_boundary_model=open_boundary_model, update_strategy=update_strategy,
              particle_spacing_factor=resolution_factor,
              domain_size=domain_size, tspan=tspan, sol=nothing)

shifting_technique = TransportVelocityAdami(background_pressure=5 * fluid_density *
                                                                sound_speed^2)

# ==========================================================================================
# ==== Postprocessing
surface_point_spacing = 0.002
circle = SphereShape(surface_point_spacing, cylinder_diameter / 2, cylinder_center,
                     fluid_density, n_layers=1, sphere_type=RoundSphere())

# Points for pressure interpolation, located at the wall interface
data_points = copy(circle.coordinates)
center = SVector(cylinder_center)
# Arc length per surface point
ds = pi * cylinder_diameter / nparticles(circle)

# Use `let` block to define the function with the *current values* of the global variables,
# instead of reading the globals every time it is called, which would make it slow.
let fluid_density = fluid_density, prescribed_velocity = prescribed_velocity,
    cylinder_diameter = cylinder_diameter, data_points = data_points, center = center,
    ds = ds

    global function force_coefficient(system, v_ode, u_ode, semi, component)
        force = zero(SVector{ndims(system), eltype(system)})
        values = interpolate_points(data_points, semi, system, v_ode, u_ode;
                                    cut_off_bnd=false, clip_negative_pressure=false)

        for i in axes(data_points, 2)
            point = TrixiParticles.current_coords(data_points, system, i)
            normal = TrixiParticles.normalize(point - center)
            force -= values.pressure[i] * ds * normal
        end

        return 2 * force[component] /
               (fluid_density * prescribed_velocity^2 * cylinder_diameter)
    end
end

lift_force(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function lift_force(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode,
                    v_ode, u_ode, semi, t)
    return force_coefficient(system, v_ode, u_ode, semi, 2)
end

drag_force(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function drag_force(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode,
                    v_ode, u_ode, semi, t)
    return force_coefficient(system, v_ode, u_ode, semi, 1)
end

# The force coefficients are computed from the pressure at the cylinder surface, which is
# very noisy at low resolutions. A velocity sensor in the wake of the cylinder yields a much
# cleaner signal of the vortex shedding, so we additionally record the transverse velocity
# `v_y` at a single point on the wake centerline.
# On the centerline, `v_y` oscillates with the shedding frequency itself
# (whereas `v_x` oscillates with twice the shedding frequency).
velocity_sensor_point = [cylinder_center[1] + 2 * cylinder_diameter; cylinder_center[2];;]

wake_velocity_y(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

let velocity_sensor_point = velocity_sensor_point
    global function wake_velocity_y(system::TrixiParticles.AbstractFluidSystem,
                                    dv_ode, du_ode, v_ode, u_ode, semi, t)
        values = interpolate_points(velocity_sensor_point, semi, system, v_ode, u_ode;
                                    cut_off_bnd=false)

        return values.velocity[2, 1]
    end
end

# Tag the output file with the resolution, e.g. `resulting_force_dp20.json` for a particle
# spacing of `d/20`.
resolution_name = round(Int, 1 / resolution_factor)

pp_callback = PostprocessCallback(; dt=0.02,
                                  lift_force, drag_force, wake_velocity_y,
                                  filename="resulting_force_dp$(resolution_name)",
                                  write_csv=false, write_file_interval=10)

# ======================================================================================
# ==== Run the simulation
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              parallelization_backend=parallelization_backend,
              reynolds_number=reynolds_number,
              open_boundary_model=open_boundary_model, update_strategy=update_strategy,
              shifting_technique=shifting_technique,
              particle_spacing_factor=resolution_factor, domain_size=domain_size,
              tspan=tspan,
              extra_callback=pp_callback, saving_callback=nothing)
