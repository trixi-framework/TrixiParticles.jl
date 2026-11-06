module HydrostaticCorrectionMethods

const MATRIX_CASES = (:none, :kernel, :gradient, :blended, :mixed,
                      :shepard, :shepard_kernel, :shepard_gradient, :shepard_blended,
                      :combined)
const DEFAULT_CASES = (MATRIX_CASES..., :continuity_reinit)
const LEGACY_CASES = (:none, :shepard, :mixed, :combined, :continuity_reinit)

function definition(case)
    name = Symbol(case)
    name in DEFAULT_CASES ||
        throw(ArgumentError("unknown hydrostatic correction case: $case"))
    shepard = name in (:shepard, :shepard_kernel, :shepard_gradient, :shepard_blended,
                       :combined, :continuity_reinit)
    gradient = name in (:kernel, :shepard_kernel) ? :kernel :
               name in (:gradient, :shepard_gradient) ? :gradient :
               name in (:blended, :shepard_blended) ? :blended :
               name in (:mixed, :combined, :continuity_reinit) ? :mixed : :none
    gradient_name = gradient == :none ? "None" :
                    gradient == :kernel ? "Kernel" :
                    gradient == :gradient ? "Gradient" :
                    gradient == :blended ? "Blend λ=0.5" : "Mixed"
    return (; shepard, gradient, gradient_name,
            continuity_control=name == :continuity_reinit,
            density_definition=shepard ?
                               "Shepard: summation filter each RHS; continuity reset at prescribed times" :
                               "No Shepard density correction")
end

function axis_label(case)
    spec = definition(case)
    spec.continuity_control && return "Continuity\nShepard+Mixed"
    gradient = spec.gradient == :blended ? "Blend 0.5" : spec.gradient_name
    return gradient * (spec.shepard ? "\nShepard on" : "\nShepard off")
end

end # module
