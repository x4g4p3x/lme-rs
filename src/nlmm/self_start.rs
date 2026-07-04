//! Data-driven starting values (`selfStart` / `getInitial` heuristics from R `stats`).

use crate::nlmm::fit::NlmmStart;
use crate::nlmm::formula::NlmmMeanKind;
use ndarray::Array1;

/// Compute starting values from `(x, y)` when the user omits `start`.
///
/// Mirrors R `stats::selfStart` for built-in means.
pub(crate) fn self_start(
    kind: NlmmMeanKind,
    x: &Array1<f64>,
    y: &Array1<f64>,
    param_names: &[String],
) -> NlmmStart {
    let xy = sorted_xy_data(x, y);
    let values: Vec<f64> = match kind {
        NlmmMeanKind::Sslogis => self_start_sslogis(&xy).to_vec(),
        NlmmMeanKind::Ssasymp | NlmmMeanKind::Ssfol => self_start_ssasymp(&xy).to_vec(),
        NlmmMeanKind::Ssmicmen => self_start_ssmicmen(&xy),
        NlmmMeanKind::Ssgompertz => self_start_ssgompertz(&xy),
    };
    let mut start = NlmmStart::new();
    for (name, value) in param_names.iter().zip(values.iter()) {
        start.insert(name.clone(), *value);
    }
    start
}

/// Sort by covariate and average duplicate `x` (R `sortedXyData`).
fn sorted_xy_data(x: &Array1<f64>, y: &Array1<f64>) -> Vec<(f64, f64)> {
    let n = x.len().min(y.len());
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .filter(|&i| x[i].is_finite() && y[i].is_finite())
        .map(|i| (x[i], y[i]))
        .collect();
    if pairs.is_empty() {
        return pairs;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = Vec::with_capacity(pairs.len());
    let mut i = 0;
    while i < pairs.len() {
        let xval = pairs[i].0;
        let mut sum = pairs[i].1;
        let mut count = 1usize;
        i += 1;
        while i < pairs.len() && (pairs[i].0 - xval).abs() <= 1e-12 * xval.abs().max(1.0) {
            sum += pairs[i].1;
            count += 1;
            i += 1;
        }
        out.push((xval, sum / count as f64));
    }
    out
}

/// Linear interpolation with flat extrapolation (R `approx(..., rule = 2)`).
fn approx(points: &[(f64, f64)], xout: f64) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    if points.len() == 1 {
        return Some(points[0].1);
    }
    if xout <= points[0].0 {
        return Some(points[0].1);
    }
    let last = points.len() - 1;
    if xout >= points[last].0 {
        return Some(points[last].1);
    }
    for w in points.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if xout >= x0 && xout <= x1 {
            if (x1 - x0).abs() < 1e-15 {
                return Some(y0);
            }
            let t = (xout - x0) / (x1 - x0);
            return Some(y0 + t * (y1 - y0));
        }
    }
    Some(points[last].1)
}

/// Given sorted `(x, y)` covariate-response pairs, find `x` at target `y`.
fn approx_x_from_y(xy: &[(f64, f64)], y_target: f64) -> Option<f64> {
    let mut pts: Vec<(f64, f64)> = xy.iter().map(|&(x, y)| (y, x)).collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    approx(&pts, y_target)
}

fn self_start_sslogis(xy: &[(f64, f64)]) -> [f64; 3] {
    if xy.len() < 3 {
        return [200.0, 725.0, 350.0];
    }
    let asym = xy.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    if !asym.is_finite() || asym <= 0.0 {
        return [200.0, 725.0, 350.0];
    }
    let xmid = approx_x_from_y(xy, asym / 2.0).unwrap_or(xy[xy.len() / 2].0);
    let x_at_quarter = approx_x_from_y(xy, asym / 4.0).unwrap_or(xmid);
    let mut scal = x_at_quarter - xmid;
    if !scal.is_finite() || scal.abs() < 1e-8 {
        scal = 1.0;
    }
    [asym, xmid, scal]
}

fn self_start_ssasymp(xy: &[(f64, f64)]) -> [f64; 3] {
    if xy.is_empty() {
        return [90.0, 20.0, (0.4_f64).ln()];
    }
    let asym = xy.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let r0 = xy.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let half = asym - (asym - r0) / 2.0;
    let xmid = approx_x_from_y(xy, half).unwrap_or(xy[0].0);
    let mut lrc = (2.0_f64).ln() / xmid;
    if !lrc.is_finite() || xmid.abs() < 1e-8 {
        lrc = 0.0;
    }
    [asym, r0, lrc]
}

fn self_start_ssmicmen(xy: &[(f64, f64)]) -> Vec<f64> {
    if xy.is_empty() {
        return vec![10.0, 1.0];
    }
    let vmax = xy.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let half = vmax / 2.0;
    let k = approx_x_from_y(xy, half).unwrap_or(xy[0].0).max(1e-6);
    vec![vmax.max(1.0), k]
}

fn self_start_ssgompertz(xy: &[(f64, f64)]) -> Vec<f64> {
    if xy.len() < 4 {
        return vec![50.0, 1.0, 0.3];
    }
    let asym = xy
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0);
    let x0 = xy.first().map(|(x, _)| *x).unwrap_or(0.0);
    let y0 = xy.first().map(|(_, y)| *y).unwrap_or(1.0).max(1e-6);
    let b3 = if x0 == 0.0 {
        0.3
    } else {
        (y0 / asym).ln().abs().max(0.1)
    };
    let b2 = 1.0;
    vec![asym, b2, b3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn ssasymp_start_on_synthetic_fixture() {
        let data = std::fs::read_to_string("tests/data/ssasymp_synthetic.csv").unwrap();
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (i, line) in data.lines().enumerate() {
            if i == 0 {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            ys.push(parts[0].parse::<f64>().unwrap());
            xs.push(parts[1].parse::<f64>().unwrap());
        }
        let start = self_start(
            NlmmMeanKind::Ssfol,
            &Array1::from_vec(xs),
            &Array1::from_vec(ys),
            &["Asym".into(), "R0".into(), "lrc".into()],
        );
        assert!(start["Asym"] > 70.0 && start["Asym"] < 95.0, "{:?}", start);
        assert!(start["R0"] > 10.0 && start["R0"] < 25.0, "{:?}", start);
        assert!(start["lrc"].is_finite(), "{:?}", start);
    }

    #[test]
    fn sslogis_start_on_orange_like_curve() {
        let x = arr1(&[100.0, 200.0, 400.0, 600.0, 800.0]);
        let y = arr1(&[30.0, 80.0, 150.0, 180.0, 195.0]);
        let start = self_start(
            NlmmMeanKind::Sslogis,
            &x,
            &y,
            &["Asym".into(), "xmid".into(), "scal".into()],
        );
        assert!(start["Asym"] > 180.0);
        assert!(start["xmid"] > 100.0);
        assert!(start["scal"].abs() > 0.0);
    }

    #[test]
    fn ssasymp_start_on_increasing_curve() {
        let x = arr1(&[0.0, 1.0, 2.0, 3.0]);
        let y = arr1(&[15.0, 40.0, 70.0, 90.0]);
        let start = self_start(
            NlmmMeanKind::Ssasymp,
            &x,
            &y,
            &["Asym".into(), "R0".into(), "lrc".into()],
        );
        assert!(start["Asym"] >= 90.0);
        assert!(start["R0"] <= 15.0);
    }
}
