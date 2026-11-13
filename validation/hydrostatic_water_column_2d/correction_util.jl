# Diagnostics shared by the correction-enabled hydrostatic validation cases.
if !isdefined(@__MODULE__, :HydrostaticCorrectionMethods)
    include("correction_methods.jl")
end

function hydrostatic_gradient_correction(case)
    gradient = HydrostaticCorrectionMethods.definition(case).gradient
    return gradient == :kernel ? KernelCorrection() :
           gradient == :gradient ? GradientCorrection() :
           gradient == :blended ? BlendedGradientCorrection(0.5) :
           gradient == :mixed ? MixedKernelGradientCorrection() : nothing
end

function hydrostatic_reference(system, height)
    depth = max(hydrostatic_surface_height - height, 0.0)
    if system isa WeaklyCompressibleSPHSystem
        # Integrate dp/d(depth) = ρ(p)g for the Cole equation of state, rather than
        # treating a compressible column as a constant-density one.
        equation = system.state_equation
        exponent = equation.exponent
        c2 = TrixiParticles.sound_speed(equation)^2
        log_density_ratio = exponent == 1 ? gravity * depth / c2 :
                            log1p((exponent - 1) * gravity * depth / c2) / (exponent - 1)
        density = equation.reference_density * exp(log_density_ratio)
        pressure = equation.reference_density * c2 / exponent *
                   expm1(exponent * log_density_ratio) + equation.background_pressure
        return (; density, pressure)
    end
    return (; density=fluid_density, pressure=fluid_density * gravity * depth)
end

hydrostatic_errors(system, dv, du, v, u, semi, t) = nothing
function hydrostatic_errors(system::TrixiParticles.AbstractFluidSystem,
                            dv, du, v_ode, u_ode, semi, t)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    density = collect(TrixiParticles.current_density(v, system))
    pressure = [TrixiParticles.current_pressure(v, system, particle)
                for particle in TrixiParticles.eachparticle(system)]
    reference = [hydrostatic_reference(system,
                                       TrixiParticles.current_coords(u, system, particle)[2])
                 for particle in TrixiParticles.eachparticle(system)]
    exact_pressure = getproperty.(reference, :pressure)
    exact_density = getproperty.(reference, :density)

    return [norm(density - exact_density) / norm(exact_density),
        norm(pressure - exact_pressure) / norm(exact_pressure),
        sum(density) / length(density), sum(pressure) / length(pressure)]
end

function initial_hydrostatic_balance(ode, fluid)
    semi = ode.p.semi
    v_ode, u_ode = ode.u0.x
    dv_ode = zero(v_ode)
    TrixiParticles.kick!(dv_ode, v_ode, u_ode,
                         (; semi, split_integration_data=nothing), first(ode.tspan))
    dv = TrixiParticles.wrap_v(dv_ode, fluid, semi)
    u = TrixiParticles.wrap_u(u_ode, fluid, semi)
    coordinates = TrixiParticles.current_coordinates(u, fluid)
    support = TrixiParticles.compact_support(fluid, fluid)
    interior = [i
                for i in TrixiParticles.eachparticle(fluid)
                if 2support < coordinates[1, i] < plate_size[1] - 2support &&
                   2support < coordinates[2, i] < hydrostatic_surface_height - 2support]
    boundary = setdiff(collect(TrixiParticles.eachparticle(fluid)), interior)
    acceleration_error(particles) = norm(dv[1:2, particles]) /
                                    (sqrt(length(particles)) * gravity)

    # Independently test the local corrected gradient of the affine field f(x,y)=y.
    # Unlike the conservative pressure force, this is the operator whose first moment
    # MixedKernelGradientCorrection is constructed to reproduce exactly.
    gradient = zeros(eltype(v_ode), 2, nparticles(fluid))
    TrixiParticles.foreach_system_wrapped(semi, v_ode,
                                          u_ode) do neighbor, v_neighbor, u_neighbor
        TrixiParticles.has_system_interaction(fluid, neighbor, semi) || return
        neighbor_coordinates = TrixiParticles.current_coordinates(u_neighbor, neighbor)
        TrixiParticles.foreach_point_neighbor(fluid, neighbor, coordinates,
                                              neighbor_coordinates,
                                              semi) do particle, other, displacement,
                                                       distance
            volume = TrixiParticles.hydrodynamic_mass(neighbor, other) /
                     TrixiParticles.current_density(v_neighbor, neighbor, other)
            grad_W = TrixiParticles.smoothing_kernel_grad(fluid, displacement, distance,
                                                          particle)
            difference = neighbor_coordinates[2, other] - coordinates[2, particle]
            gradient[:, particle] .+= volume * difference .* grad_W
        end
    end
    gradient[2, :] .-= 1
    errors = hydrostatic_errors(fluid, dv_ode, nothing, v_ode, u_ode, semi,
                                first(ode.tspan))
    return (; initial_density_error=errors[1], initial_pressure_error=errors[2],
            initial_momentum_error=acceleration_error(TrixiParticles.eachparticle(fluid)),
            initial_momentum_error_interior=acceleration_error(interior),
            initial_momentum_error_boundary=acceleration_error(boundary),
            initial_affine_gradient_error=norm(gradient) / sqrt(nparticles(fluid)))
end

function density_reinit_audit()
    return Ref((; count=0, volume_residual=0.0, frozen_volume_difference=0.0,
                evolved_density_change=0.0, eos_residual=0.0, edac_pressure_change=0.0))
end

# The counterfactual freezes fluid volumes only; boundary volumes remain at the
# current hydrodynamic density in both sums, isolating the evolved-fluid-volume role.
function evolved_volume_reference(fluid, v_ode, u_ode, semi)
    u = TrixiParticles.wrap_u(u_ode, fluid, semi)
    coordinates = TrixiParticles.current_coordinates(u, fluid)
    numerator = zeros(eltype(v_ode), nparticles(fluid))
    evolved = zero(numerator)
    frozen = zero(numerator)
    TrixiParticles.foreach_system_wrapped(semi, v_ode,
                                          u_ode) do neighbor, v_neighbor, u_neighbor
        TrixiParticles.has_system_interaction(fluid, neighbor, semi) || return
        TrixiParticles.foreach_point_neighbor(fluid, neighbor, coordinates,
                                              TrixiParticles.current_coordinates(u_neighbor,
                                                                                 neighbor),
                                              semi) do a, b, r, distance
            weight = TrixiParticles.hydrodynamic_mass(neighbor, b) *
                     TrixiParticles.smoothing_kernel(fluid, distance, a)
            density = TrixiParticles.current_density(v_neighbor, neighbor, b)
            initial = neighbor isa TrixiParticles.AbstractFluidSystem ?
                      neighbor.initial_condition.density[b] : density
            numerator[a] += weight
            evolved[a] += weight / density
            frozen[a] += weight / initial
        end
    end
    return (; expected=numerator ./ evolved, frozen=numerator ./ frozen)
end

function scheduled_density_reinit!(callback, integrator, audit=nothing)
    # Split integration may have advanced the structure after the last fluid RHS.
    # Rebuild the current geometry/boundary state before using evolved-density volumes.
    v, u = integrator.u.x
    TrixiParticles.update_systems_and_nhs(v, u, integrator.p.semi, integrator.t)
    isnothing(audit) &&
        return TrixiParticles.apply_reinitialization!((callback.affect!,), integrator)
    fluid = integrator.p.semi.systems[callback.affect!.system_index]
    reference = evolved_volume_reference(fluid, v, u, integrator.p.semi)
    state = TrixiParticles.wrap_v(v, fluid, integrator.p.semi)
    initial = fluid.initial_condition.density
    density_change = norm(TrixiParticles.current_density(state, fluid) - initial) /
                     norm(initial)
    pressure_before = [TrixiParticles.current_pressure(state, fluid, i)
                       for i in TrixiParticles.eachparticle(fluid)]
    result = TrixiParticles.apply_reinitialization!((callback.affect!,), integrator)
    actual = TrixiParticles.current_density(state, fluid)
    pressure_after = [TrixiParticles.current_pressure(state, fluid, i)
                      for i in TrixiParticles.eachparticle(fluid)]
    eos = fluid isa WeaklyCompressibleSPHSystem ?
          maximum(abs.(pressure_after - fluid.state_equation.(actual))) /
          (fluid_density * gravity * hydrostatic_surface_height) : 0.0
    pressure_change = fluid isa EntropicallyDampedSPHSystem ?
                      norm(pressure_after - pressure_before) /
                      max(norm(pressure_before), eps()) : 0.0
    previous = audit[]
    audit[] = (; count=(previous.count + 1),
               volume_residual=max(previous.volume_residual,
                                   norm(actual - reference.expected) /
                                   norm(reference.expected)),
               frozen_volume_difference=max(previous.frozen_volume_difference,
                                            norm(reference.expected - reference.frozen) /
                                            norm(reference.expected)),
               evolved_density_change=max(previous.evolved_density_change, density_change),
               eos_residual=max(previous.eos_residual, eos),
               edac_pressure_change=max(previous.edac_pressure_change, pressure_change))
    return result
end

equation_of_state_residual(system, dv, du, v, u, semi, t) = nothing
function equation_of_state_residual(system::WeaklyCompressibleSPHSystem,
                                    dv, du, v_ode, u_ode, semi, t)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    residual = maximum(TrixiParticles.eachparticle(system)) do particle
        density = TrixiParticles.current_density(v, system, particle)
        pressure = TrixiParticles.current_pressure(v, system, particle)
        abs(pressure - system.state_equation(density))
    end
    return residual / (fluid_density * gravity * initial_fluid_size[2])
end

correction_plate_deflection(system, dv, du, v, u, semi, t) = nothing
function correction_plate_deflection(system::TotalLagrangianSPHSystem,
                                     dv, du, v_ode, u_ode, semi, t)
    # Track the middle of the plate, independently of the system declaration order.
    initial = system.initial_coordinates
    center = SVector(plate_size[1] / 2, -plate_size[2] / 2)
    particle = argmin([norm(SVector{2}(view(initial, :, i)) - center)
                       for i in TrixiParticles.each_integrated_particle(system)])
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    return TrixiParticles.current_coords(u, system, particle)[2] - initial[2, particle]
end

interface_force_balance(system, dv, du, v, u, semi, t) = nothing
function interface_force_balance(fluid::TrixiParticles.AbstractFluidSystem,
                                 dv, du, v_ode, u_ode, semi, t)
    structure = only(filter(system -> system isa TotalLagrangianSPHSystem, semi.systems))
    v_fluid = TrixiParticles.wrap_v(v_ode, fluid, semi)
    u_fluid = TrixiParticles.wrap_u(u_ode, fluid, semi)
    v_structure = TrixiParticles.wrap_v(v_ode, structure, semi)
    u_structure = TrixiParticles.wrap_u(u_ode, structure, semi)
    dv_fluid = zeros(eltype(v_ode), TrixiParticles.v_nvariables(fluid), nparticles(fluid))
    # Include the reaction on clamped particles, which are absent from the integrated state.
    dv_structure = zeros(eltype(v_ode), TrixiParticles.v_nvariables(structure),
                         nparticles(structure))
    TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure, u_structure,
                             fluid, structure, semi)
    TrixiParticles.interact!(dv_structure, v_structure, u_structure, v_fluid, u_fluid,
                             structure, fluid, semi;
                             eachparticle=TrixiParticles.eachparticle(structure))

    fluid_force = vec(sum(fluid.mass' .* view(dv_fluid, 1:2, :); dims=2))
    structure_forces = structure.mass' .* view(dv_structure, 1:2, :)
    structure_force = vec(sum(structure_forces; dims=2))
    clamped = (structure.n_integrated_particles + 1):nparticles(structure)
    clamp_force = vec(sum(view(structure_forces, :, clamped); dims=2))
    force_scale = norm(fluid_force) + norm(structure_force)
    residual = norm(fluid_force + structure_force) / max(force_scale, eps())

    # These are interface forces only: gravity, wall forces, and elastic forces are excluded.
    return [fluid_force..., structure_force..., clamp_force..., residual, force_scale]
end

function correction_series(data, name)
    key = only(filter(key -> startswith(key, name * "_"), collect(keys(data))))
    return data[key]
end

function correction_summary(data; method, correction, reverse_order, resolution,
                            spacing, analytical_deflection, averaging_interval)
    errors = correction_series(data, "hydrostatic_errors")["values"]
    forces = correction_series(data, "interface_force_balance")["values"]
    deflection = correction_series(data, "plate_deflection")
    selected = findall(t -> first(averaging_interval) <= t <= last(averaging_interval),
                       deflection["time"])
    late_density_error = isempty(selected) ? missing :
                         sum(errors[i][1] for i in selected) / length(selected)
    late_pressure_error = isempty(selected) ? missing :
                          sum(errors[i][2] for i in selected) / length(selected)
    late_deflection_span = isempty(selected) ? missing :
                           (maximum(deflection["values"][selected]) -
                            minimum(deflection["values"][selected])) /
                           abs(analytical_deflection)
    midpoint = sum(averaging_interval) / 2
    first_half = filter(i -> deflection["time"][i] <= midpoint, selected)
    second_half = filter(i -> deflection["time"][i] > midpoint, selected)
    late_deflection_drift = isempty(first_half) || isempty(second_half) ? missing :
                            abs(sum(deflection["values"][first_half]) / length(first_half) -
                                sum(deflection["values"][second_half]) /
                                length(second_half)) /
                            abs(analytical_deflection)
    deflection_error = isempty(selected) ? missing :
                       abs(sum(deflection["values"][selected]) / length(selected) -
                           analytical_deflection) / abs(analytical_deflection)
    eos_residual = method == :wcsph ?
                   maximum(correction_series(data, "eos_residual")["values"]) : missing

    return (; method, correction, reverse_order, resolution, spacing,
            final_density_error=last(errors)[1], final_pressure_error=last(errors)[2],
            max_density_error=maximum(error -> error[1], errors),
            max_pressure_error=maximum(error -> error[2], errors),
            eos_residual, interface_residual=maximum(force -> force[7], forces),
            min_interface_force_scale=minimum(force -> force[8], forces),
            final_deflection=last(deflection["values"]), deflection_error,
            late_density_error, late_pressure_error, late_deflection_span,
            late_deflection_drift)
end

function correction_order_comparison(forward, reversed; method, resolution,
                                     analytical_deflection)
    a = correction_series(forward, "hydrostatic_errors")
    b = correction_series(reversed, "hydrostatic_errors")
    a["time"] == b["time"] ||
        error("system-order comparison requires matching sample times")
    density_difference = maximum(abs(x[3] - y[3])
                                 for (x, y) in zip(a["values"], b["values"])) /
                         fluid_density
    pressure_difference = maximum(abs(x[4] - y[4])
                                  for (x, y) in zip(a["values"], b["values"])) /
                          (fluid_density * gravity * initial_fluid_size[2])
    deflection_a = correction_series(forward, "plate_deflection")["values"]
    deflection_b = correction_series(reversed, "plate_deflection")["values"]
    deflection_difference = maximum(abs.(deflection_a - deflection_b)) /
                            abs(analytical_deflection)
    return (; method, resolution, density_difference, pressure_difference,
            deflection_difference)
end
