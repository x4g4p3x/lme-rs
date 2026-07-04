# LMM with offset parity exporter (MixedModels.jl).
#   julia comparisons/parity/sleepstudy_offset.jl

using CSV
using DataFrames
using JSON
using MixedModels

file_path = joinpath("tests", "data", "sleepstudy.csv")
isfile(file_path) || error("missing dataset: $file_path")

df = CSV.read(file_path, DataFrame)
df.Subject = string.(df.Subject)

form = @formula(Reaction ~ 1 + Days + offset(OffsetDays10) + (Days | Subject))
m = fit(MixedModel, form, df, REML = true)

coef_names = coefnames(m)
coef_vals = coef(m)
coefficients = Dict(zip(coef_names, coef_vals))

payload = Dict(
    "case" => "sleepstudy_offset",
    "implementation" => "julia",
    "coefficients" => coefficients,
)
println(JSON.json(payload))
