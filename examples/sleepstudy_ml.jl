using CSV
using DataFrames
using MixedModels

df = CSV.read("tests/data/sleepstudy.csv", DataFrame)
form = @formula(Reaction ~ 1 + Days + (1 + Days | Subject))

# REML=false implies Maximum Likelihood
m1 = fit(MixedModel, form, df, REML=false)
println("=== Model Summary ===")
println(m1)

newdata = DataFrame(Days = [0.0, 1.0, 5.0, 10.0])
preds = predict(m1, newdata)
println("\n=== Predictions ===")
println(preds)
