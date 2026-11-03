using TrixiParticles
using LinearAlgebra: BLAS
using JSON

include("refinement_util.jl")
BLAS.set_num_threads(1)
input = JSON.parsefile(only(ARGS))
job, protocol = input["job"], input["protocol"]
HydrostaticRefinement.source_signature() == input["source_signature"] ||
    error("simulation source changed after this refinement job was scheduled")

trixi_include(@__MODULE__, joinpath(@__DIR__, "validation_corrections.jl");
              resolutions=(job["resolution"],), methods=(Symbol(job["method"]),),
              wcsph_density_mode=Symbol(HydrostaticRefinement.base_density_mode(job)),
              edac_density_mode=Symbol(HydrostaticRefinement.base_density_mode(job)),
              correction_cases=(Symbol(job["correction"]),),
              tspan=Tuple(protocol["tspan"]),
              averaging_interval=Tuple(protocol["averaging_interval"]),
              sample_dt=protocol["sample_dt"],
              reinitialization_dt=protocol["reinitialization_dt"],
              check_reverse_order=protocol["check_reverse_order"], save_solution=false,
              write_file_interval=4,
              output_root=job["directory"])

HydrostaticRefinement.validate_output(job["directory"], job, protocol)
HydrostaticRefinement.source_signature() == input["source_signature"] ||
    error("simulation source changed while this refinement job was running")
HydrostaticRefinement.write_json(joinpath(job["directory"], "complete.json"),
                                 (; protocol, source_signature=input["source_signature"],
                                  provenance="fresh simulation",
                                  postprocess_write_file_interval=4))
