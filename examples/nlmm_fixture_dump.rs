//! Print nlmer fits for golden JSON fixture authoring.

use lme_rs::nlmer;
use lme_rs::nlmm::NlmmStart;
use polars::prelude::*;
use std::fs::File;

fn fit_csv(path: &str, formula: &str) -> lme_rs::LmeFit {
    let mut file = File::open(path).unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();
    nlmer(formula, &df, NlmmStart::new(), false).unwrap()
}

fn main() {
    let mic = fit_csv(
        "tests/data/ssmicmen_synthetic.csv",
        "y ~ SSmicmen(x, Vmax, K) ~ Vmax|id",
    );
    println!("MICMEN coef: {:?}", mic.coefficients.as_slice().unwrap());
    println!(
        "MICMEN theta: {:?}",
        mic.theta.as_ref().unwrap().as_slice().unwrap()
    );
    println!("MICMEN sigma2: {:?}", mic.sigma2);

    let gom = fit_csv(
        "tests/data/ssgompertz_synthetic.csv",
        "y ~ SSgompertz(x, Asym, b2, b3) ~ Asym|id",
    );
    println!("GOMP coef: {:?}", gom.coefficients.as_slice().unwrap());
    println!(
        "GOMP theta: {:?}",
        gom.theta.as_ref().unwrap().as_slice().unwrap()
    );
    println!("GOMP sigma2: {:?}", gom.sigma2);
}
