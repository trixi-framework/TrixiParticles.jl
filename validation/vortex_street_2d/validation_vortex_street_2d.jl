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

# Import variables into scope without running the simulation.
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              reynolds_number=reynolds_number, saving_callback=nothing,
              open_boundary_model=open_boundary_model, factor_d=resolution_factor,
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
        values = interpolate_points(data_points, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                    clip_negative_pressure=false)

        for i in axes(data_points, 2)
            point = TrixiParticles.current_coords(data_points, system, i)
            normal = TrixiParticles.normalize(point - center)
            force -= values.pressure[i] * ds * normal
        end

        return 2 * force[component] / (fluid_density * prescribed_velocity^2 * cylinder_diameter)
    end
end

calculate_lift_force(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function calculate_lift_force(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode,
                              v_ode, u_ode, semi, t)
    return force_coefficient(system, v_ode, u_ode, semi, 2)
end

calculate_drag_force(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function calculate_drag_force(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode,
                              v_ode, u_ode, semi, t)
    return force_coefficient(system, v_ode, u_ode, semi, 1)
end

pp_callback = PostprocessCallback(; dt=0.02,
                                  f_l=calculate_lift_force, f_d=calculate_drag_force,
                                  filename="resulting_force",
                                  write_csv=true, write_file_interval=10)

# ======================================================================================
# ==== Run the simulation
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              parallelization_backend=PolyesterBackend(), reynolds_number=reynolds_number,
              open_boundary_model=open_boundary_model,
              shifting_technique=shifting_technique,
              factor_d=resolution_factor, domain_size=domain_size, tspan=tspan,
              extra_callback=pp_callback)
