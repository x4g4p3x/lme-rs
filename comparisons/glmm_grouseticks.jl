# Please install the required packages before running:
# import Pkg; Pkg.add(["CSV", "DataFrames", "MixedModels"])

using CSV
using DataFrames
using MixedModels

function main()
    file_path = joinpath("tests", "data", "grouseticks.csv")
    if !isfile(file_path)
        println("Could not find the dataset at: $file_path")
        println("Please run this example from the root of the lme-rs repository.")
        exit(1)
    end

    println("Loading data from $file_path...")
    
    # 1. Load the dataset
    df = CSV.read(file_path, DataFrame)
    println("Successfully loaded $(nrow(df)) rows.")

    # 2. Fit the Poisson GLMM model
    println("\nFitting Poisson GLMM: TICKS ~ YEAR + HEIGHT + (1 | BROOD)")
    println("Evaluating Maximum Likelihood via Laplace Approximation...")
    
    # MixedModels.jl uses the same formula syntax and Laplace approximation
    # as lme4, written by the same original author (Douglas Bates).
    form = @formula(TICKS ~ YEAR + HEIGHT + (1 | BROOD))
    
    # Fit the Generalized Linear Mixed Model (Poisson with Log link)
    m1 = fit(MixedModel, form, df, Poisson(), LogLink())
    
    # 3. Print the summary
    println("\n=== Model Summary ===")
    println(m1)

    # 4. Generate Predictions (Response Scale)
    println("\n=== Predictions (Response Scale) ===")
    println("Generating expected tick counts for 3 new broods...")
    
    # Create new data
    newdata = DataFrame(
        YEAR = [96, 96, 97],
        HEIGHT = [400, 500, 450],
        BROOD = ["new1", "new2", "new3"]
    )
    
    # Predict (Population-level, response scale)
    # MixedModels.jl doesn't have a direct equivalent to `re.form = NA` out of the box in the `predict` function,
    # it typically predicts exactly what is in the data, or requires the model matrix. 
    # To get population-level predictions we can extract the fixed effects and compute:
    
    # \mu = \exp(X \beta)
    beta = coef(m1)
    
    # Manual X matrix matching formula: Intercept, YEAR, HEIGHT
    X = hcat(ones(3), newdata.YEAR, newdata.HEIGHT)
    
    eta = X * beta
    preds = exp.(eta) # Inverse Log Link
    
    println("Expected Tick Counts:")
    println(preds)
end

main()
