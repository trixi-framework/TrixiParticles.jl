#
# DEM Contact Models using Multiple Dispatch
#
# This module demonstrates how to support different contact models
# by dispatching on the type of the contact model and storing model-
# specific values in the model type.
#
# Literature References:
#   [Cundall and Strack, 1979] – Linear spring–dashpot model.
#   [Di Renzo and Di Maio, 2004] – Hertzian contact model.
#   [Bicanic, 2004] – Overview of DEM implementations.
#

# Define an abstract type for contact models.
abstract type ContactModel end

# HertzContactModel: Non-linear contact using Hertzian theory.
struct HertzContactModel <: ContactModel
    elastic_modulus::Float64  # Material elastic modulus
    poissons_ratio::Float64   # Material Poisson's ratio
end

# LinearContactModel: Simple linear spring-dashpot contact.
struct LinearContactModel <: ContactModel
    normal_stiffness::Float64 # Constant stiffness value for linear contact
end
