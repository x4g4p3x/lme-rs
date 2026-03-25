# Please install the required packages before running:
# import Pkg; Pkg.add(["CSV", "DataFrames", "MixedModels"])
#
# MixedModels.jl fits this GLMM with Laplace-style marginal likelihood (no nAGQ knob).
# For AGQ parity, use the R script (lme4 `nAGQ = 7`) or Rust/Python `n_agq = 7` examples.

using CSV
using DataFrames
using MixedModels

function main()
    file_path = joinpath("tests", "data", "cbpp_binary.csv")
    if !isfile(file_path)
        println("Could not find the dataset at: $file_path")
        println("Please run this example from the root of the lme-rs repository.")
        exit(1)
    end

    println("Loading data from $file_path...")
    
    # 1. Load the dataset
    df = CSV.read(file_path, DataFrame)
    # Ensure herd is treated as categorical
    df.herd = string.(df.herd)

    # 2. Fit the Binomial GLMM model
    println("\nFitting Binomial GLMM: y ~ period2 + period3 + period4 + (1 | herd)")
    
    form = @formula(y ~ 1 + period2 + period3 + period4 + (1 | herd))
    
    # Fit the Generalized Linear Mixed Model (Binomial with Logit link)
    m1 = fit(MixedModel, form, df, Binomial(), LogitLink())
    
    # 3. Print the summary
    println("\n=== Model Summary ===")
    println(m1)

    # 4. Generate Predictions (Response Scale)
    println("\n=== Predictions (Probabilities) ===")
    println("Generating predictions for herd 1 across periods...")
    
    newdata = DataFrame(
        herd = ["1", "1", "1", "1"],
        period2 = [0, 1, 0, 0],
        period3 = [0, 0, 1, 0],
        period4 = [0, 0, 0, 1]
    )
    
    # \mu = invlogit(X \beta)
    beta = coef(m1)
    X = hcat(ones(4), newdata.period2, newdata.period3, newdata.period4)
    eta = X * beta
    
    # Manual invlogit
    invlogit(x) = 1.0 / (1.0 + exp(-x))
    preds = invlogit.(eta)
    
    println("Predictions (Population-level):")
    println(preds)
end

main()
