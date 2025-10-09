using TrixiParticles
using FFTW
using CSV, DataFrames
using Test
# using ThreadPinning; pinthreads(:numa)

# Results in [90k particles, 340k particles, 1.2M particles, 5M particles]
# In the Tafuni et al. (2018), the resolution is `0.01` (5M particles).
resolution_factor = 0.02 # [0.08, 0.04, 0.02, 0.01]

reynolds_number = 200

open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method=ZerothOrderMirroring())
# open_boundary_model = BoundaryModelDynamicalPressureZhang()

model = nameof(typeof(open_boundary_model))
output_directory = joinpath(validation_dir(), "vortex_street_2d", "tvf",
                            "$(model)_dp_$(resolution_factor)D_Re_$reynolds_number")

# ==========================================================================================
# ==== Postprocessing
trixi_include(joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              reynolds_number=reynolds_number, saving_callback=nothing,
              open_boundary_model=open_boundary_model,
              output_directory=output_directory, factor_d=resolution_factor, sol=nothing)

circle = SphereShape(0.001, cylinder_diameter / 2,
                     cylinder_center, fluid_density, n_layers=1,
                     sphere_type=RoundSphere())

# Points for pressure interpolation, located at the wall interface
const data_points = copy(circle.coordinates)
const center = SVector(cylinder_center)
# Arc length per surface point
const ds = pi * cylinder_diameter / nparticles(circle)

calculate_lift_force(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function calculate_lift_force(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode,
                              v_ode, u_ode, semi, t)
    force = zero(SVector{ndims(system), eltype(system)})

    values = interpolate_points(data_points, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                clip_negative_pressure=false)
    pressure = Array(values.pressure)

    for i in axes(data_points, 2)
        point = TrixiParticles.current_coords(data_points, system, i)

        # F = ∑ -p_i * A_i * n_i
        force -= pressure[i] * ds * TrixiParticles.normalize(point - center)
    end

    return 2 * force[2] / (fluid_density * prescribed_velocity^2 * cylinder_diameter)
end

calculate_drag_force(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function calculate_drag_force(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode,
                              v_ode,
                              u_ode, semi, t)
    force = zero(SVector{ndims(system), eltype(system)})

    values = interpolate_points(data_points, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                clip_negative_pressure=false)
    pressure = Array(values.pressure)

    for i in axes(data_points, 2)
        point = TrixiParticles.current_coords(data_points, system, i)

        # F = ∑ -p_i * A_i * n_i
        force -= pressure[i] * ds * TrixiParticles.normalize(point - center)
    end

    return 2 * force[1] / (fluid_density * prescribed_velocity^2 * cylinder_diameter)
end

pp_callback = PostprocessCallback(; dt=0.02,
                                  f_l=calculate_lift_force, f_d=calculate_drag_force,
                                  output_directory, filename="resulting_force",
                                  write_csv=true, write_file_interval=10)

shifting_technique = TransportVelocityAdami(background_pressure=5 * fluid_density *
                                                                sound_speed^2)

# ======================================================================================
# ==== Run the simulation
trixi_include(joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              parallelization_backend=PolyesterBackend(), reynolds_number=reynolds_number,
              saving_callback=nothing, output_directory=output_directory,
              open_boundary_model=open_boundary_model,
              shifting_technique=shifting_technique,
              factor_d=resolution_factor, tspan=(0.0, 20.0), extra_callback=pp_callback)
