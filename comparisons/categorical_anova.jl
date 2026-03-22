using DataFrames
using CSV
using MixedModels
using Printf

println("\n=== Julia (MixedModels.jl) ===")
const data = DataFrame(CSV.File("tests/data/pastes.csv"))

# MixedModels automatically detects string columns as categorical
model = fit(MixedModel, @formula(strength ~ cask + (1 | batch)), data, REML=true)
println(model)

# MixedModels automatically provides Wald tests for all components? Not natively as a unified DataFrame block without external packages.
