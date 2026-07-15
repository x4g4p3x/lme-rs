"""Type stubs for the ``lme_python`` PyO3 extension."""

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

import polars as pl

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

DataFrameInput = Union[pl.DataFrame, "pd.DataFrame", "pa.Table"]

class PyConfintResult:
    lower: list[float]
    upper: list[float]
    names: list[str]
    level: float
    def as_tuples(self) -> list[tuple[float, float]]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[float, float]: ...

class PySimulateResult:
    simulations: list[list[float]]
    def __len__(self) -> int: ...

class PySimulateBatches:
    def __iter__(self) -> PySimulateBatches: ...
    def __next__(self) -> PySimulateResult: ...

class PyFixedEffectsAnova:
    anova_type: str
    method: str
    terms: list[str]
    num_df: list[float]
    den_df: list[float]
    f_value: list[float]
    p_value: list[float]

class PyContrastTest:
    method: str
    num_df: float
    den_df: float
    f_value: float
    p_value: float

class PyLikelihoodRatioAnova:
    n_params_0: int
    n_params_1: int
    deviance_0: float
    deviance_1: float
    chi_sq: float
    df: int
    p_value: float
    formula_0: str
    formula_1: str

class PyBootReplicate:
    index: int
    coefficients: list[float]
    theta: Optional[list[float]]
    sigma2: Optional[float]
    converged: bool

class PyBootLmerResult:
    method: str
    nsim: int
    fixed_names: list[str]
    t0: list[float]
    t0_sigma2: Optional[float]
    replicates: list[PyBootReplicate]
    prop_converged: float
    def confint(self, level: float = 0.95) -> PyBootConfintResult: ...

class PyBootConfintResult:
    names: list[str]
    estimate: list[float]
    lower: list[float]
    upper: list[float]
    level: float
    def __str__(self) -> str: ...

class PyFamily:
    Binomial: int
    Poisson: int
    Gamma: int
    Gaussian: int

class PyLmeFit:
    formula: Optional[str]
    family_name: Optional[str]
    link_name: Optional[str]
    family: Optional[str]
    coefficients: list[float]
    fixed_names: Optional[list[str]]
    b: list[float]
    u: Optional[list[float]]
    beta_se: Optional[list[float]]
    beta_t: Optional[list[float]]
    std_errors: Optional[list[float]]
    fixed_term_assign: Optional[list[str]]
    categorical_levels: Optional[dict[str, list[str]]]
    v_beta_unscaled: Optional[list[list[float]]]
    sigma2: Optional[float]
    theta: Optional[list[float]]
    aic: Optional[float]
    bic: Optional[float]
    log_likelihood: Optional[float]
    deviance: Optional[float]
    converged: Optional[bool]
    num_obs: int
    iterations: Optional[int]
    reml_criterion: Optional[float]
    residuals: list[float]
    fitted: list[float]
    ranef: Optional[list[tuple[str, str, str, float]]]
    var_corr: Optional[list[tuple[str, str, str, float, float]]]
    robust_se: Optional[list[float]]
    robust_t: Optional[list[float]]
    robust_p_values: Optional[list[float]]
    satterthwaite_dfs: Optional[list[float]]
    satterthwaite_p_values: Optional[list[float]]
    kenward_roger_dfs: Optional[list[float]]
    kenward_roger_p_values: Optional[list[float]]
    def summary(self) -> str: ...
    def predict(self, newdata: DataFrameInput) -> list[float]: ...
    def predict_conditional(
        self, newdata: DataFrameInput, allow_new_levels: bool = ...
    ) -> list[float]: ...
    def predict_response(self, newdata: DataFrameInput) -> list[float]: ...
    def predict_conditional_response(
        self, newdata: DataFrameInput, allow_new_levels: bool = ...
    ) -> list[float]: ...
    def confint(self, level: float = 0.95) -> PyConfintResult: ...
    def anova(
        self, ddf_method: str = "satterthwaite", anova_type: str = "III"
    ) -> PyFixedEffectsAnova: ...
    def linear_hypothesis(
        self, term: str, ddf_method: str = "satterthwaite"
    ) -> PyContrastTest: ...
    def linear_hypothesis_terms(
        self, terms: list[str], ddf_method: str = "satterthwaite"
    ) -> PyContrastTest: ...
    def test_contrast(
        self, l_matrix: list[list[float]], ddf_method: str = "satterthwaite"
    ) -> PyContrastTest: ...
    def test_contrast_vs(
        self,
        l_matrix: list[list[float]],
        beta_h: list[float],
        ddf_method: str = "satterthwaite",
    ) -> PyContrastTest: ...
    def simulate(
        self,
        nsim: int,
        n_jobs: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> PySimulateResult: ...
    def simulate_batches(
        self,
        nsim: int,
        batch_size: int,
        n_jobs: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> PySimulateBatches: ...
    def boot(
        self,
        formula: str,
        data: DataFrameInput,
        nsim: int = 200,
        method: str = "parametric",
        reml: bool = True,
        seed: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> PyBootLmerResult: ...
    def boot_glmer(
        self,
        formula: str,
        data: DataFrameInput,
        nsim: int = 200,
        method: str = "parametric",
        seed: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> PyBootLmerResult: ...
    def with_robust_se(
        self, data: DataFrameInput, cluster_col: Optional[str] = None
    ) -> None: ...
    def with_satterthwaite(self, data: DataFrameInput) -> None: ...
    def with_kenward_roger(self, data: DataFrameInput) -> None: ...

class PyLmerPrepared:
    blocked_kernel: bool
    blocked_kernel_detail: str

class PyCvFoldMetric:
    fold: int
    n_train_groups: int
    n_test_groups: int
    n_train_obs: int
    n_test_obs: int
    rmse: float
    mae: float
    converged: bool

class PyCvGroupedResult:
    oof_predictions: list[float]
    test_fold: list[int]
    rmse: float
    mae: float
    folds: list[PyCvFoldMetric]
    all_converged: bool
    n_splits: int
    group_col: str

def lm(formula: str, data: DataFrameInput) -> PyLmeFit: ...
def lm_matrix(y: list[float], x: list[list[float]]) -> PyLmeFit: ...
def lmer(formula: str, data: DataFrameInput, reml: bool = True) -> PyLmeFit: ...
def prepare_lmer(formula: str, data: DataFrameInput) -> PyLmerPrepared: ...
def fit_prepared(prepared: PyLmerPrepared, reml: bool = True) -> PyLmeFit: ...
def refit_lmer(formula: str, data: DataFrameInput, reml: bool = True) -> PyLmeFit: ...
def cv_grouped(
    formula: str,
    data: DataFrameInput,
    group: str,
    n_splits: int = 5,
    reml: bool = True,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> PyCvGroupedResult: ...
def boot_lmer(
    formula: str,
    data: DataFrameInput,
    fit: PyLmeFit,
    nsim: int = 200,
    method: str = "parametric",
    reml: bool = True,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> PyBootLmerResult: ...
def boot_glmer(
    formula: str,
    data: DataFrameInput,
    fit: PyLmeFit,
    nsim: int = 200,
    method: str = "parametric",
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> PyBootLmerResult: ...
def lmer_weighted(
    formula: str,
    data: DataFrameInput,
    reml: bool = True,
    weights: Optional[list[float]] = None,
) -> PyLmeFit: ...
def glmer(
    formula: str,
    data: DataFrameInput,
    family_name: str,
    n_agq: int = 1,
    link_name: Optional[str] = None,
) -> PyLmeFit: ...
def glmer_weighted(
    formula: str,
    data: DataFrameInput,
    family_name: str,
    n_agq: int = 1,
    weights: Optional[list[float]] = None,
    link_name: Optional[str] = None,
) -> PyLmeFit: ...
def nlmer(
    formula: str,
    data: DataFrameInput,
    start: Optional[dict[str, float]] = None,
    reml: bool = False,
    n_agq: int = 1,
    lower: Optional[dict[str, float]] = None,
    upper: Optional[dict[str, float]] = None,
) -> PyLmeFit: ...
def nlmer_with_mean(
    formula: str,
    data: DataFrameInput,
    mean_fn: Callable[[float, Sequence[float]], tuple[float, Sequence[float]]],
    param_names: list[str],
    start: Optional[dict[str, float]] = None,
    reml: bool = False,
    n_agq: int = 1,
    lower: Optional[dict[str, float]] = None,
    upper: Optional[dict[str, float]] = None,
) -> PyLmeFit: ...
def anova(fit_a: PyLmeFit, fit_b: PyLmeFit) -> PyLikelihoodRatioAnova: ...
def contrast_matrix(p: int, rows: list[list[tuple[int, float]]]) -> list[list[float]]: ...
def contrast_matrix_from_names(
    fixed_names: list[str], rows: list[list[tuple[str, float]]]
) -> list[list[float]]: ...
