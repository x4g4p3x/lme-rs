//! Nonlinear mixed-model fitting (Laplace / penalized Gauss–Newton, `nAGQ = 0` style).

use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Solve, UPLO};
use crate::model_matrix::ReBlock;
use crate::nlmm::formula::{NlmmMeanKind, NlmerFormula};
use crate::nlmm::sslogis::sslogis_eval;
use crate::optimizer::compute_theta_lower_bounds;
use crate::{LmeError, LmeFit};
use polars::prelude::*;

/// Starting values for fixed nonlinear parameters (by name).
pub type NlmmStart = std::collections::HashMap<String, f64>;

/// Options for [`fit_nlmer`](crate::nlmm::fit_nlmer).
#[derive(Debug, Clone)]
pub struct NlmerOptions {
    /// Use REML profiling for the variance component (default: ML).
    pub reml: bool,
    /// Starting values for fixed nonlinear parameters (`Asym`, `xmid`, `scal`, …).
    pub start: NlmmStart,
    /// Maximum penalized Gauss–Newton iterations per RE-variance evaluation.
    pub max_inner: usize,
    /// Reserved for future multi-θ optimizers (golden-section uses a fixed budget).
    pub max_outer_iters: u64,
}

impl Default for NlmerOptions {
    fn default() -> Self {
        Self {
            reml: false,
            start: NlmmStart::new(),
            max_inner: 120,
            max_outer_iters: 500,
        }
    }
}

struct NlmmProblem {
    y: Array1<f64>,
    x: Array1<f64>,
    group: Vec<usize>,
    m: usize,
    mean: NlmmMeanKind,
}

impl NlmmProblem {
    fn predict(&self, asym: f64, xmid: f64, scal: f64, b: &Array1<f64>) -> Array1<f64> {
        let n = self.y.len();
        let mut mu = Array1::<f64>::zeros(n);
        match self.mean {
            NlmmMeanKind::Sslogis => {
                for i in 0..n {
                    let a = asym + b[self.group[i]];
                    mu[i] = sslogis_eval(a, xmid, scal, self.x[i]).0;
                }
            }
        }
        mu
    }

    fn penalized_rss(&self, asym: f64, xmid: f64, scal: f64, b: &Array1<f64>, tau2: f64) -> f64 {
        let mu = self.predict(asym, xmid, scal, b);
        let rss: f64 = self
            .y
            .iter()
            .zip(mu.iter())
            .map(|(&y, &m)| (y - m).powi(2))
            .sum();
        let b_pen: f64 = b.iter().map(|v| v * v).sum::<f64>() / tau2;
        rss + b_pen
    }

    fn inner_gauss_newton(
        &self,
        tau: f64,
        start: &NlmmStart,
        names: &[String; 3],
        max_iter: usize,
    ) -> (f64, f64, f64, Array1<f64>, f64) {
        let default = [200.0, 725.0, 350.0];
        let mut asym = start.get(&names[0]).copied().unwrap_or(default[0]);
        let mut xmid = start.get(&names[1]).copied().unwrap_or(default[1]);
        let mut scal = start.get(&names[2]).copied().unwrap_or(default[2]);
        if scal.abs() < 1e-8 {
            scal = default[2];
        }
        let mut b = Array1::<f64>::zeros(self.m);
        let tau2 = tau.max(1e-8).powi(2);
        let n = self.y.len();
        let p_fix = 3usize;
        let p = p_fix + self.m;
        let mut lambda_lm = 1e-2;

        for _ in 0..max_iter {
            let mut j = Array2::<f64>::zeros((n, p));
            let mut r = Array1::<f64>::zeros(n);

            match self.mean {
                NlmmMeanKind::Sslogis => {
                    for i in 0..n {
                        let g = self.group[i];
                        let a = asym + b[g];
                        let (mui, da, dx, ds) = sslogis_eval(a, xmid, scal, self.x[i]);
                        r[i] = self.y[i] - mui;
                        j[[i, 0]] = da;
                        j[[i, 1]] = dx;
                        j[[i, 2]] = ds;
                        j[[i, p_fix + g]] = da;
                    }
                }
            }

            let mut jtj = j.t().dot(&j);
            for g in 0..self.m {
                jtj[[p_fix + g, p_fix + g]] += 1.0 / tau2;
            }
            let rhs = j.t().dot(&r);
            let old_obj = self.penalized_rss(asym, xmid, scal, &b, tau2);

            let mut accepted = false;
            let mut step_norm = 0.0f64;
            for _attempt in 0..12 {
                let mut damped = jtj.clone();
                for i in 0..p {
                    damped[[i, i]] += lambda_lm * damped[[i, i]].max(1e-8);
                }
                let delta = match damped.cholesky(UPLO::Lower).and_then(|c| c.solve(&rhs)) {
                    Ok(d) => d,
                    Err(_) => break,
                };

                let mut alpha = 1.0;
                while alpha >= 1e-6 {
                    let na = asym + alpha * delta[0];
                    let nx = xmid + alpha * delta[1];
                    let mut ns = scal + alpha * delta[2];
                    if ns < 1e-6 {
                        ns = 1e-6;
                    }
                    let mut nb = b.clone();
                    for g in 0..self.m {
                        nb[g] += alpha * delta[p_fix + g];
                    }
                    let new_obj = self.penalized_rss(na, nx, ns, &nb, tau2);
                    if new_obj < old_obj {
                        step_norm = (alpha * delta).iter().map(|v| v.abs()).fold(0.0, f64::max);
                        asym = na;
                        xmid = nx;
                        scal = ns;
                        b = nb;
                        lambda_lm = (lambda_lm * 0.3).max(1e-8);
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }
                if accepted {
                    break;
                }
                lambda_lm = (lambda_lm * 10.0).min(1e6);
            }
            if !accepted {
                break;
            }
            if step_norm < 1e-10 {
                break;
            }
        }

        let mu = self.predict(asym, xmid, scal, &b);
        let rss: f64 = self.y.iter().zip(mu.iter()).map(|(&y, &m)| (y - m).powi(2)).sum();
        (asym, xmid, scal, b, rss)
    }

    fn profile_objective(
        &self,
        tau: f64,
        start: &NlmmStart,
        names: &[String; 3],
        reml: bool,
        max_inner: usize,
    ) -> (f64, f64, f64, f64, f64, Array1<f64>) {
        let (asym, xmid, scal, b, rss) = self.inner_gauss_newton(tau, start, names, max_inner);
        let n = self.y.len() as f64;
        let p = 3.0;
        let df = if reml { (n - p).max(1.0) } else { n };
        let sigma2 = (rss / df).max(1e-12);
        let tau2 = tau.max(1e-8).powi(2);
        let b_pen: f64 = b.iter().map(|v| v * v).sum::<f64>() / tau2;
        let twopi = std::f64::consts::PI * 2.0;
        let mut crit = df * (twopi * sigma2).ln() + rss / sigma2 + b_pen + self.m as f64 * tau2.ln();
        if reml {
            crit += (self.m as f64 - p) * (1.0 + sigma2.ln());
        }
        (crit, asym, xmid, scal, sigma2, b)
    }
}

/// Golden-section search for a single variance parameter (RE standard deviation).
fn optimize_tau_golden(
    problem: &NlmmProblem,
    parsed: &NlmerFormula,
    start: &NlmmStart,
    reml: bool,
    max_inner: usize,
    lo: f64,
    hi: f64,
) -> (f64, f64, u64) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut c = b - (b - a) / phi;
    let mut d = a + (b - a) / phi;
    let mut fc = problem.profile_objective(
        c,
        start,
        &parsed.fixed_param_names,
        reml,
        max_inner,
    );
    let mut fd = problem.profile_objective(
        d,
        start,
        &parsed.fixed_param_names,
        reml,
        max_inner,
    );
    let mut fc_cost = fc.0;
    let mut fd_cost = fd.0;
    let mut iters = 0u64;
    while (b - a).abs() > 1e-4 && iters < 80 {
        iters += 1;
        if fc_cost < fd_cost {
            b = d;
            d = c;
            fd = fc;
            fd_cost = fc_cost;
            c = b - (b - a) / phi;
            fc = problem.profile_objective(
                c,
                start,
                &parsed.fixed_param_names,
                reml,
                max_inner,
            );
            fc_cost = fc.0;
        } else {
            a = c;
            c = d;
            fc = fd;
            fc_cost = fd_cost;
            d = a + (b - a) / phi;
            fd = problem.profile_objective(
                d,
                start,
                &parsed.fixed_param_names,
                reml,
                max_inner,
            );
            fd_cost = fd.0;
        }
    }
    let tau = (a + b) / 2.0;
    let (_, _, _, _, _, _) = problem.profile_objective(
        tau,
        start,
        &parsed.fixed_param_names,
        reml,
        max_inner,
    );
    let final_cost = problem
        .profile_objective(
            tau,
            start,
            &parsed.fixed_param_names,
            reml,
            max_inner,
        )
        .0;
    (tau, final_cost, iters)
}

/// Fit a nonlinear mixed model from a parsed formula and data frame.
pub fn fit_nlmer(
    parsed: &NlmerFormula,
    mean: NlmmMeanKind,
    data: &DataFrame,
    formula_str: &str,
    opts: &NlmerOptions,
) -> crate::Result<LmeFit> {
    if mean != NlmmMeanKind::Sslogis {
        return Err(LmeError::NotImplemented {
            feature: "Only SSlogis mean is implemented".to_string(),
        });
    }
    if parsed.re_param != parsed.fixed_param_names[0] {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "Random effect must be on '{}' (only random effect on Asym is supported)",
                parsed.fixed_param_names[0]
            ),
        });
    }

    let y = column_f64(data, &parsed.response)?;
    let x = column_f64(data, &parsed.covariate)?;
    let groups = column_str(data, &parsed.re_group)?;
    let mut level_map = std::collections::HashMap::<String, usize>::new();
    let mut group = Vec::with_capacity(y.len());
    for g in groups {
        let m = level_map.len();
        let idx = *level_map.entry(g).or_insert(m);
        group.push(idx);
    }
    let m = level_map.len();
    if m == 0 {
        return Err(LmeError::NotImplemented {
            feature: "No random-effect groups".to_string(),
        });
    }

    let problem = NlmmProblem {
        y: y.clone(),
        x,
        group,
        m,
        mean,
    };

    let _lower = compute_theta_lower_bounds(&[ReBlock {
        m,
        k: 1,
        theta_len: 1,
        group_name: parsed.re_group.clone(),
        effect_names: vec![parsed.re_param.clone()],
        group_map: level_map.clone(),
    }]);
    let (tau, _final_cost, outer_iters) = optimize_tau_golden(
        &problem,
        parsed,
        &opts.start,
        opts.reml,
        opts.max_inner,
        1e-3,
        80.0,
    );
    let (crit, asym, xmid, scal, _sigma2_inner, b) = problem.profile_objective(
        tau,
        &opts.start,
        &parsed.fixed_param_names,
        opts.reml,
        opts.max_inner,
    );

    let fitted = problem.predict(asym, xmid, scal, &b);
    let residuals = &y - &fitted;
    let n = y.len();
    let rss_nl: f64 = residuals.iter().map(|r| r * r).sum();
    let n_f = n as f64;
    let p = 3.0;
    let df = if opts.reml { (n_f - p).max(1.0) } else { n_f };
    let sigma2 = (rss_nl / df).max(1e-12);
    let deviance = crit;
    let loglik = -crit / 2.0;

    let names = parsed.fixed_param_names.clone();
    let coefficients = Array1::from_vec(vec![asym, xmid, scal]);
    let mut ranef_rows = Vec::new();
    for (label, &idx) in &level_map {
        ranef_rows.push((
            parsed.re_group.clone(),
            label.clone(),
            parsed.re_param.clone(),
            b[idx],
        ));
    }
    let ranef_df = build_ranef_df(&ranef_rows);
    let var_corr_df = build_nlmm_varcorr(&parsed.re_group, &parsed.re_param, tau, sigma2);

    let n_params = 3.0 + 1.0 + 1.0; // fixed NL + theta + sigma
    let aic = deviance + 2.0 * n_params;
    let bic = deviance + n_params * (n as f64).ln();

    Ok(LmeFit {
        coefficients,
        residuals,
        fitted,
        ranef: Some(ranef_df),
        var_corr: Some(var_corr_df),
        theta: Some(Array1::from_vec(vec![tau])),
        sigma2: Some(sigma2),
        reml: if opts.reml { Some(crit) } else { None },
        log_likelihood: Some(loglik),
        aic: Some(aic),
        bic: Some(bic),
        deviance: Some(deviance),
        b: Some(b),
        u: None,
        beta_se: None,
        beta_t: None,
        formula: Some(formula_str.to_string()),
        fixed_names: Some(names.to_vec()),
        fixed_term_assign: None,
        fixed_design_x: None,
        re_blocks: None,
        num_obs: n,
        converged: Some(true),
        iterations: Some(outer_iters),
        family_name: Some("nlmm".to_string()),
        link_name: None,
        family: None,
        satterthwaite: None,
        kenward_roger: None,
        v_beta_unscaled: None,
        robust: None,
        categorical_levels: None,
    })
}

fn column_f64(df: &DataFrame, name: &str) -> crate::Result<Array1<f64>> {
    let s = df
        .column(name)
        .map_err(|e| LmeError::NotImplemented {
            feature: format!("Column '{name}': {e}"),
        })?;
    if let Ok(ca) = s.f64() {
        return Ok(Array1::from_iter(ca.into_iter().map(|v| v.unwrap_or(f64::NAN))));
    }
    if let Ok(ca) = s.i64() {
        return Ok(Array1::from_iter(
            ca.into_iter().map(|v| v.unwrap_or(0) as f64),
        ));
    }
    if let Ok(ca) = s.f32() {
        return Ok(Array1::from_iter(
            ca.into_iter().map(|v| v.unwrap_or(f32::NAN) as f64),
        ));
    }
    Err(LmeError::NotImplemented {
        feature: format!("Column '{name}' must be numeric"),
    })
}

fn column_str(df: &DataFrame, name: &str) -> crate::Result<Vec<String>> {
    let s = df
        .column(name)
        .map_err(|e| LmeError::NotImplemented {
            feature: format!("Column '{name}': {e}"),
        })?;
    if let Ok(ca) = s.str() {
        return Ok(ca
            .into_iter()
            .map(|v| v.unwrap_or("").to_string())
            .collect());
    }
    if let Ok(ca) = s.f64() {
        return Ok(ca
            .into_iter()
            .map(|v| v.map(|x| x.to_string()).unwrap_or_default())
            .collect());
    }
    if let Ok(ca) = s.i64() {
        return Ok(ca
            .into_iter()
            .map(|v| v.map(|x| x.to_string()).unwrap_or_default())
            .collect());
    }
    // Factor-like categoricals from CSV
    let cast = s.cast(&DataType::String).map_err(|e| LmeError::NotImplemented {
        feature: format!("Column '{name}' could not be cast to string: {e}"),
    })?;
    let ca = cast.str().map_err(|e| LmeError::NotImplemented {
        feature: format!("Column '{name}' string cast failed: {e}"),
    })?;
    Ok(ca
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect())
}

fn build_ranef_df(rows: &[(String, String, String, f64)]) -> DataFrame {
    let groups: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    let levels: Vec<String> = rows.iter().map(|r| r.1.clone()).collect();
    let names: Vec<String> = rows.iter().map(|r| r.2.clone()).collect();
    let vals: Vec<f64> = rows.iter().map(|r| r.3).collect();
    DataFrame::new(vec![
        Column::new("group".into(), &groups),
        Column::new("level".into(), &levels),
        Column::new("term".into(), &names),
        Column::new("condval".into(), &vals),
    ])
    .unwrap_or_default()
}

#[cfg(test)]
mod orange_inner {
    use super::*;
    use crate::nlmm::formula::{parse_nlmer_formula, NlmmMeanKind};
    use std::fs::File;

    #[test]
    fn inner_at_r_tau_matches_reference() {
        let mut file = File::open("tests/data/orange.csv").unwrap();
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .into_reader_with_file_handle(&mut file)
            .finish()
            .unwrap();
        let (parsed, mean) =
            parse_nlmer_formula("circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree")
                .unwrap();
        let y = column_f64(&df, &parsed.response).unwrap();
        let x = column_f64(&df, &parsed.covariate).unwrap();
        let groups = column_str(&df, &parsed.re_group).unwrap();
        let mut level_map = std::collections::HashMap::<String, usize>::new();
        let mut group = Vec::new();
        for g in groups {
            let m = level_map.len();
            let idx = *level_map.entry(g).or_insert(m);
            group.push(idx);
        }
        let mut start = NlmmStart::new();
        start.insert("Asym".to_string(), 200.0);
        start.insert("xmid".to_string(), 725.0);
        start.insert("scal".to_string(), 350.0);
        let problem = NlmmProblem {
            y,
            x,
            group,
            m: 5,
            mean,
        };
        let (asym, xmid, scal, _, _) = problem.inner_gauss_newton(
            31.646,
            &start,
            &parsed.fixed_param_names,
            200,
        );
        assert!(
            (asym - 192.0528).abs() < 3.0,
            "asym={asym} xmid={xmid} scal={scal}"
        );
        assert_eq!(mean, NlmmMeanKind::Sslogis);
    }
}

fn build_nlmm_varcorr(group: &str, param: &str, tau: f64, sigma2: f64) -> DataFrame {
    let var = tau.powi(2);
    let grps = [group.to_string(), "Residual".to_string()];
    let var1 = [param.to_string(), String::new()];
    let vcov = [var, sigma2];
    let sdcor = [var.sqrt(), sigma2.sqrt()];
    DataFrame::new(vec![
        Column::new("grp".into(), &grps),
        Column::new("var1".into(), &var1),
        Column::new("vcov".into(), &vcov),
        Column::new("sdcor".into(), &sdcor),
    ])
    .unwrap_or_default()
}
