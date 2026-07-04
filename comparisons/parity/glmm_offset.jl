# Poisson GLMM with offset parity exporter (MixedModels.jl).
#   julia comparisons/parity/glmm_offset.jl

using CSV
using DataFrames
using JSON
using MixedModels

file_path = joinpath("tests", "data", "grouseticks.csv")
isfile(file_path) || error("missing dataset: $file_path")

df = CSV.read(file_path, DataFrame)
df.BROOD = string.(df.BROOD)

form = @formula(TICKS ~ 1 + YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD))
m = fit(MixedModel, form, df, Poisson(), LogLink())

coef_names = coefnames(m)
coef_vals = coef(m)
coefficients = Dict(zip(coef_names, coef_vals))

payload = Dict(
    "case" => "glmm_offset",
    "implementation" => "julia",
    "coefficients" => coefficients,
)
println(JSON.json(payload))
