# Weighted binomial GLMM parity exporter (MixedModels.jl).
#   julia comparisons/parity/glmm_weighted.jl

using CSV
using DataFrames
using JSON
using MixedModels

file_path = joinpath("tests", "data", "cbpp_binary_weighted.csv")
isfile(file_path) || error("missing dataset: $file_path")

df = CSV.read(file_path, DataFrame)
df.herd = string.(df.herd)

form = @formula(y ~ 1 + period2 + period3 + period4 + (1 | herd))
m = fit(MixedModel, form, df, Binomial(), LogitLink(); wts = df.prior_w)

coef_names = coefnames(m)
coef_vals = coef(m)
coefficients = Dict(zip(coef_names, coef_vals))

payload = Dict(
    "case" => "glmm_weighted",
    "implementation" => "julia",
    "coefficients" => coefficients,
)
println(JSON.json(payload))
