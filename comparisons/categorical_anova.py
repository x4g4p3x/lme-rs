import pandas as pd
import statsmodels.formula.api as smf

print("\n=== Python (statsmodels) ===")
data = pd.read_csv("tests/data/pastes.csv")

# statsmodels automatically dummifies categorical variables in strings
model = smf.mixedlm("strength ~ C(cask)", data, groups=data["batch"])
result = model.fit(reml=True)
print(result.summary())

# Multi-Dof Wald ANOVA isn't generically available as a single function call in statsmodels for mixedlm without custom contrast matrices,
# but we can test the joint hypothesis of the 'cask' parameters being zero.
print("\n=== Wald Joint Test for cask ===")
wald_test = result.wald_test("C(cask)[T.b] = 0, C(cask)[T.c] = 0")
print(wald_test)
