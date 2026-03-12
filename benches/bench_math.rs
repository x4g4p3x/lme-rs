use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use lme_rs::anova::DdfMethod;
use lme_rs::math::LmmData;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::Deserialize;
use sprs::{CsMat, TriMat};
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;

#[derive(Debug, Deserialize)]
struct TestData {
    pub inputs: Inputs,
    pub outputs: Outputs,
}

#[derive(Debug, Deserialize)]
struct Inputs {
    #[serde(rename = "X")]
    pub x: Vec<Vec<f64>>,
    #[serde(rename = "Zt")]
    pub zt: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Outputs {
    pub theta: Vec<f64>,
}

fn load_test_data() -> (LmmData, Vec<f64>) {
    let file = File::open("tests/data/random_slopes.json")
        .expect("Failed to open random_slopes.json for benchmarking.");
    let reader = BufReader::new(file);
    let data: TestData = serde_json::from_reader(reader).expect("Failed to parse JSON");

    let x_arr = Array2::from_shape_vec(
        (data.inputs.x.len(), data.inputs.x[0].len()),
        data.inputs.x.into_iter().flatten().collect(),
    )
    .unwrap();

    let mut zt_tri = TriMat::new((data.inputs.zt.len(), data.inputs.zt[0].len()));
    for (i, row) in data.inputs.zt.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val != 0.0 {
                zt_tri.add_triplet(i, j, val);
            }
        }
    }
    let zt_arr: CsMat<f64> = zt_tri.to_csr();

    let y_arr = Array1::from_vec(data.inputs.y);

    let re_blocks = vec![lme_rs::model_matrix::ReBlock {
        m: 18,
        k: 2,
        theta_len: 3,
        group_name: "Subject".to_string(),
        effect_names: vec!["(Intercept)".to_string(), "Days".to_string()],
        group_map: std::collections::HashMap::new(),
    }];
    (
        LmmData::new(x_arr, zt_arr, y_arr, re_blocks),
        data.outputs.theta,
    )
}

fn load_csv(path: &str) -> DataFrame {
    let mut file = File::open(path).unwrap_or_else(|_| panic!("Could not open {}", path));
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap()
}

fn build_prediction_df(base: &DataFrame, repeats: usize) -> DataFrame {
    let days = base
        .column("Days")
        .expect("Days column missing")
        .cast(&DataType::Float64)
        .expect("Days should cast to f64");
    let days = days
        .f64()
        .expect("Days should be f64")
        .into_no_null_iter()
        .collect::<Vec<_>>();

    let subjects = base
        .column("Subject")
        .expect("Subject column missing")
        .cast(&DataType::String)
        .expect("Subject should cast to string");
    let subjects = subjects
        .str()
        .expect("Subject should be string")
        .into_no_null_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();

    let total = days.len() * repeats;
    let mut out_days = Vec::with_capacity(total);
    let mut out_subjects = Vec::with_capacity(total);
    for _ in 0..repeats {
        out_days.extend(days.iter().copied());
        out_subjects.extend(subjects.iter().cloned());
    }

    df!("Days" => out_days, "Subject" => out_subjects).unwrap()
}

fn repeat_dataframe(base: &DataFrame, repeats: usize) -> DataFrame {
    let mut out = base.clone();
    for _ in 1..repeats {
        out.vstack_mut(base).unwrap();
    }
    out
}

fn make_weights(n: usize) -> Array1<f64> {
    let weights = (0..n)
        .map(|i| 0.5 + ((i % 11) as f64) / 10.0)
        .collect::<Vec<_>>();
    Array1::from_vec(weights)
}

fn generate_large_synthetic_df(n_obs: usize, n_groups: usize) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut y = Vec::with_capacity(n_obs);
    let mut x1 = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);

    for _ in 0..n_obs {
        let g = rng.random_range(0..n_groups);
        let current_x = normal.sample(&mut rng);
        let current_y = 1.0 + 2.0 * current_x + normal.sample(&mut rng);

        y.push(current_y);
        x1.push(current_x);
        group.push(format!("G{}", g));
    }

    df!(
        "y" => &y,
        "x" => &x1,
        "group" => &group
    )
    .unwrap()
}

fn generate_large_crossed_df(n_obs: usize, n_plates: usize, n_samples: usize) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(1337);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let plate_effects: Vec<f64> = (0..n_plates).map(|_| normal.sample(&mut rng)).collect();
    let sample_effects: Vec<f64> = (0..n_samples).map(|_| normal.sample(&mut rng)).collect();

    let mut y = Vec::with_capacity(n_obs);
    let mut x = Vec::with_capacity(n_obs);
    let mut plate = Vec::with_capacity(n_obs);
    let mut sample = Vec::with_capacity(n_obs);

    for _ in 0..n_obs {
        let plate_idx = rng.random_range(0..n_plates);
        let sample_idx = rng.random_range(0..n_samples);
        let x_i = normal.sample(&mut rng);
        let noise = 0.25 * normal.sample(&mut rng);
        let y_i = 1.5 + 0.75 * x_i + plate_effects[plate_idx] + sample_effects[sample_idx] + noise;

        y.push(y_i);
        x.push(x_i);
        plate.push(format!("P{}", plate_idx));
        sample.push(format!("S{}", sample_idx));
    }

    df!(
        "y" => y,
        "x" => x,
        "plate" => plate,
        "sample" => sample
    )
    .unwrap()
}

fn generate_large_nested_df(
    n_batches: usize,
    casks_per_batch: usize,
    reps_per_cask: usize,
) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(2026);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let batch_effects: Vec<f64> = (0..n_batches).map(|_| normal.sample(&mut rng)).collect();
    let cask_effects: Vec<Vec<f64>> = (0..n_batches)
        .map(|_| {
            (0..casks_per_batch)
                .map(|_| normal.sample(&mut rng))
                .collect()
        })
        .collect();

    let total = n_batches * casks_per_batch * reps_per_cask;
    let mut y = Vec::with_capacity(total);
    let mut x = Vec::with_capacity(total);
    let mut batch = Vec::with_capacity(total);
    let mut cask = Vec::with_capacity(total);

    for batch_idx in 0..n_batches {
        for cask_idx in 0..casks_per_batch {
            for _ in 0..reps_per_cask {
                let x_i = normal.sample(&mut rng);
                let noise = 0.2 * normal.sample(&mut rng);
                let y_i = 2.0
                    + 1.25 * x_i
                    + batch_effects[batch_idx]
                    + cask_effects[batch_idx][cask_idx]
                    + noise;

                y.push(y_i);
                x.push(x_i);
                batch.push(format!("B{}", batch_idx));
                cask.push(format!("C{}", cask_idx));
            }
        }
    }

    df!(
        "y" => y,
        "x" => x,
        "batch" => batch,
        "cask" => cask
    )
    .unwrap()
}

fn bench_formula_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("formula_parsing");
    for (name, formula) in [
        ("random_slopes", "Reaction ~ Days + (Days | Subject)"),
        ("nested_intercepts", "strength ~ 1 + (1 | batch/cask)"),
        (
            "crossed_intercepts",
            "diameter ~ 1 + (1 | plate) + (1 | sample)",
        ),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| black_box(lme_rs::formula::parse(black_box(formula))).unwrap())
        });
    }
    group.finish();
}

fn bench_model_matrix_building(c: &mut Criterion) {
    let sleepstudy = load_csv("tests/data/sleepstudy.csv");
    let pastes = load_csv("tests/data/pastes.csv");
    let penicillin = load_csv("tests/data/penicillin.csv");

    let sleepstudy_ast = lme_rs::formula::parse("Reaction ~ Days + (Days | Subject)").unwrap();
    let pastes_ast = lme_rs::formula::parse("strength ~ 1 + (1 | batch/cask)").unwrap();
    let penicillin_ast =
        lme_rs::formula::parse("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();

    let mut group = c.benchmark_group("model_matrix_build");
    group.bench_function("sleepstudy_random_slopes", |b| {
        b.iter(|| {
            black_box(lme_rs::model_matrix::build_design_matrices(
                black_box(&sleepstudy_ast),
                black_box(&sleepstudy),
            ))
            .unwrap()
        })
    });
    group.bench_function("pastes_nested", |b| {
        b.iter(|| {
            black_box(lme_rs::model_matrix::build_design_matrices(
                black_box(&pastes_ast),
                black_box(&pastes),
            ))
            .unwrap()
        })
    });
    group.bench_function("penicillin_crossed", |b| {
        b.iter(|| {
            black_box(lme_rs::model_matrix::build_design_matrices(
                black_box(&penicillin_ast),
                black_box(&penicillin),
            ))
            .unwrap()
        })
    });
    group.finish();
}

fn bench_deviance_evaluation(c: &mut Criterion) {
    let (model, _theta) = load_test_data();

    c.bench_function("log_reml_deviance_random_slopes", |b| {
        b.iter(|| {
            black_box(model.log_reml_deviance(black_box(&[0.8078, 0.0, 1.0]), true));
        })
    });

    let n = 50_000;
    let groups = 5_000;

    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut x_large = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x_large[[i, 0]] = 1.0;
        x_large[[i, 1]] = normal.sample(&mut rng);
    }
    let y_large = Array1::from_shape_fn(n, |_| normal.sample(&mut rng));

    let mut zt_large_tri = TriMat::new((groups * 2, n));
    for i in 0..n {
        let group_idx = i % groups;
        zt_large_tri.add_triplet(group_idx * 2, i, 1.0);
        zt_large_tri.add_triplet(group_idx * 2 + 1, i, x_large[[i, 1]]);
    }
    let zt_large: CsMat<f64> = zt_large_tri.to_csr();

    let re_blocks_large = vec![lme_rs::model_matrix::ReBlock {
        m: groups,
        k: 2,
        theta_len: 3,
        group_name: "group".to_string(),
        effect_names: vec!["(Intercept)".to_string(), "x".to_string()],
        group_map: std::collections::HashMap::new(),
    }];

    let lmm_large = LmmData::new(x_large, zt_large, y_large, re_blocks_large);

    let theta_large = vec![1.0; lmm_large.re_blocks.iter().map(|b| b.theta_len).sum()];
    let mut large_group = c.benchmark_group("isolated_eval");
    large_group.sample_size(10);
    large_group.bench_function("eval_50k_lmm", |b| {
        b.iter(|| {
            black_box(lmm_large.log_reml_deviance(&theta_large, true));
        })
    });
    large_group.finish();
}

fn bench_lmer_end_to_end(c: &mut Criterion) {
    let df = load_csv("tests/data/sleepstudy.csv");

    c.bench_function("lmer_random_slopes (sleepstudy)", |b| {
        b.iter(|| {
            black_box(
                lme_rs::lmer(
                    black_box("Reaction ~ Days + (Days | Subject)"),
                    black_box(&df),
                    black_box(true),
                )
                .unwrap(),
            );
        })
    });
}

fn bench_weighted_models(c: &mut Criterion) {
    let sleepstudy = load_csv("tests/data/sleepstudy.csv");
    let sleepstudy_weights = make_weights(sleepstudy.height());
    let weighted_large = generate_large_synthetic_df(100_000, 500);
    let weighted_large_weights = make_weights(weighted_large.height());

    let mut group = c.benchmark_group("weighted_models");
    group.sample_size(10);

    group.bench_function("lmer_weighted_sleepstudy", |b| {
        b.iter(|| {
            black_box(
                lme_rs::lmer_weighted(
                    black_box("Reaction ~ Days + (Days | Subject)"),
                    black_box(&sleepstudy),
                    black_box(true),
                    black_box(Some(sleepstudy_weights.clone())),
                )
                .unwrap(),
            );
        })
    });

    group.bench_function("lmer_weighted_100k_obs", |b| {
        b.iter(|| {
            black_box(
                lme_rs::lmer_weighted(
                    black_box("y ~ x + (1 | group)"),
                    black_box(&weighted_large),
                    black_box(false),
                    black_box(Some(weighted_large_weights.clone())),
                )
                .unwrap(),
            );
        })
    });

    group.finish();
}

fn bench_glmer_end_to_end(c: &mut Criterion) {
    let df = load_csv("tests/data/grouseticks.csv");

    c.bench_function("glmer_poisson (grouseticks)", |b| {
        b.iter(|| {
            black_box(
                lme_rs::glmer(
                    black_box("TICKS ~ YEAR96 + YEAR97 + (1 | BROOD)"),
                    black_box(&df),
                    black_box(lme_rs::family::Family::Poisson),
                )
                .unwrap(),
            );
        })
    });

    let df2 = load_csv("tests/data/cbpp_binary.csv");
    c.bench_function("glmer_binomial (cbpp)", |b| {
        b.iter(|| {
            black_box(
                lme_rs::glmer(
                    black_box("y ~ period2 + period3 + period4 + (1 | herd)"),
                    black_box(&df2),
                    black_box(lme_rs::family::Family::Binomial),
                )
                .unwrap(),
            );
        })
    });
}

fn bench_prediction(c: &mut Criterion) {
    let df = load_csv("tests/data/sleepstudy.csv");
    let fit = lme_rs::lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    let prediction_df = build_prediction_df(&df, 100);

    let mut group = c.benchmark_group("prediction");
    group.bench_function("predict_population_18k_rows", |b| {
        b.iter(|| black_box(fit.predict(black_box(&prediction_df))).unwrap())
    });
    group.bench_function("predict_conditional_18k_rows", |b| {
        b.iter(|| {
            black_box(fit.predict_conditional(black_box(&prediction_df), black_box(true))).unwrap()
        })
    });
    group.finish();
}

fn bench_glmm_post_fit(c: &mut Criterion) {
    let poisson_df = load_csv("tests/data/grouseticks.csv");
    let poisson_fit = lme_rs::glmer(
        "TICKS ~ YEAR96 + YEAR97 + (1 | BROOD)",
        &poisson_df,
        lme_rs::family::Family::Poisson,
    )
    .unwrap();
    let poisson_prediction_df = repeat_dataframe(&poisson_df, 20);

    let binomial_df = load_csv("tests/data/cbpp_binary.csv");
    let binomial_fit = lme_rs::glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &binomial_df,
        lme_rs::family::Family::Binomial,
    )
    .unwrap();
    let binomial_prediction_df = repeat_dataframe(&binomial_df, 20);

    let mut group = c.benchmark_group("glmm_post_fit");
    group.sample_size(10);

    group.bench_function("poisson_predict_link_8k_rows", |b| {
        b.iter(|| black_box(poisson_fit.predict(black_box(&poisson_prediction_df))).unwrap())
    });

    group.bench_function("poisson_predict_response_8k_rows", |b| {
        b.iter(|| {
            black_box(poisson_fit.predict_response(black_box(&poisson_prediction_df))).unwrap()
        })
    });

    group.bench_function("poisson_predict_conditional_link_8k_rows", |b| {
        b.iter(|| {
            black_box(
                poisson_fit
                    .predict_conditional(black_box(&poisson_prediction_df), black_box(false)),
            )
            .unwrap()
        })
    });

    group.bench_function("poisson_predict_conditional_response_8k_rows", |b| {
        b.iter(|| {
            black_box(
                poisson_fit.predict_conditional_response(
                    black_box(&poisson_prediction_df),
                    black_box(false),
                ),
            )
            .unwrap()
        })
    });

    group.bench_function("binomial_predict_link_16k_rows", |b| {
        b.iter(|| black_box(binomial_fit.predict(black_box(&binomial_prediction_df))).unwrap())
    });

    group.bench_function("binomial_predict_response_16k_rows", |b| {
        b.iter(|| {
            black_box(binomial_fit.predict_response(black_box(&binomial_prediction_df))).unwrap()
        })
    });

    group.bench_function("binomial_predict_conditional_link_16k_rows", |b| {
        b.iter(|| {
            black_box(
                binomial_fit
                    .predict_conditional(black_box(&binomial_prediction_df), black_box(false)),
            )
            .unwrap()
        })
    });

    group.bench_function("binomial_predict_conditional_response_16k_rows", |b| {
        b.iter(|| {
            black_box(
                binomial_fit.predict_conditional_response(
                    black_box(&binomial_prediction_df),
                    black_box(false),
                ),
            )
            .unwrap()
        })
    });

    group.bench_function("poisson_confint_wald", |b| {
        b.iter(|| black_box(poisson_fit.confint(black_box(0.95))).unwrap())
    });

    group.bench_function("glmm_confint_wald", |b| {
        b.iter(|| black_box(binomial_fit.confint(black_box(0.95))).unwrap())
    });

    group.finish();
}

fn bench_lmer_large_synthetic(c: &mut Criterion) {
    let df_100k = generate_large_synthetic_df(100_000, 500);

    let mut group = c.benchmark_group("large_scale");
    group.sample_size(10);

    group.bench_function("lmer_100k_obs", |b| {
        b.iter(|| {
            black_box(
                lme_rs::lmer(
                    black_box("y ~ x + (1 | group)"),
                    black_box(&df_100k),
                    black_box(false),
                )
                .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_large_structure_fits(c: &mut Criterion) {
    let crossed_df = generate_large_crossed_df(20_000, 250, 100);
    let nested_df = generate_large_nested_df(200, 10, 5);

    let mut group = c.benchmark_group("large_structures");
    group.sample_size(10);

    group.bench_function("lmer_crossed_20k_obs", |b| {
        b.iter(|| {
            black_box(
                lme_rs::lmer(
                    black_box("y ~ x + (1 | plate) + (1 | sample)"),
                    black_box(&crossed_df),
                    black_box(false),
                )
                .unwrap(),
            );
        })
    });

    group.bench_function("lmer_nested_10k_obs", |b| {
        b.iter(|| {
            black_box(
                lme_rs::lmer(
                    black_box("y ~ x + (1 | batch/cask)"),
                    black_box(&nested_df),
                    black_box(false),
                )
                .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_size_sweeps(c: &mut Criterion) {
    let random_cases = [
        ("random_intercept_10k_obs_100_groups", 10_000usize, 100usize),
        ("random_intercept_50k_obs_500_groups", 50_000usize, 500usize),
        (
            "random_intercept_100k_obs_1000_groups",
            100_000usize,
            1_000usize,
        ),
    ];
    let random_dfs = random_cases
        .into_iter()
        .map(|(name, n_obs, n_groups)| (name, generate_large_synthetic_df(n_obs, n_groups)))
        .collect::<Vec<_>>();

    let crossed_cases = [
        ("crossed_5k_obs_100x50", 5_000usize, 100usize, 50usize),
        ("crossed_20k_obs_250x100", 20_000usize, 250usize, 100usize),
        ("crossed_50k_obs_500x250", 50_000usize, 500usize, 250usize),
    ];
    let crossed_dfs = crossed_cases
        .into_iter()
        .map(|(name, n_obs, n_plates, n_samples)| {
            (name, generate_large_crossed_df(n_obs, n_plates, n_samples))
        })
        .collect::<Vec<_>>();

    let nested_cases = [
        ("nested_2k_obs_50x5x8", 50usize, 5usize, 8usize),
        ("nested_10k_obs_200x10x5", 200usize, 10usize, 5usize),
        ("nested_30k_obs_300x20x5", 300usize, 20usize, 5usize),
    ];
    let nested_dfs = nested_cases
        .into_iter()
        .map(|(name, n_batches, casks_per_batch, reps_per_cask)| {
            (
                name,
                generate_large_nested_df(n_batches, casks_per_batch, reps_per_cask),
            )
        })
        .collect::<Vec<_>>();

    let mut random_group = c.benchmark_group("sweep_random_intercept_fit");
    random_group.sample_size(10);
    for (name, df) in &random_dfs {
        random_group.bench_function(*name, |b| {
            b.iter(|| {
                black_box(
                    lme_rs::lmer(
                        black_box("y ~ x + (1 | group)"),
                        black_box(df),
                        black_box(false),
                    )
                    .unwrap(),
                )
            })
        });
    }
    random_group.finish();

    let mut crossed_group = c.benchmark_group("sweep_crossed_fit");
    crossed_group.sample_size(5);
    for (name, df) in &crossed_dfs {
        crossed_group.bench_function(*name, |b| {
            b.iter(|| {
                black_box(
                    lme_rs::lmer(
                        black_box("y ~ x + (1 | plate) + (1 | sample)"),
                        black_box(df),
                        black_box(false),
                    )
                    .unwrap(),
                )
            })
        });
    }
    crossed_group.finish();

    let mut nested_group = c.benchmark_group("sweep_nested_fit");
    nested_group.sample_size(5);
    for (name, df) in &nested_dfs {
        nested_group.bench_function(*name, |b| {
            b.iter(|| {
                black_box(
                    lme_rs::lmer(
                        black_box("y ~ x + (1 | batch/cask)"),
                        black_box(df),
                        black_box(false),
                    )
                    .unwrap(),
                )
            })
        });
    }
    nested_group.finish();
}

fn bench_prediction_structure_sweeps(c: &mut Criterion) {
    let crossed_small = generate_large_crossed_df(5_000, 100, 50);
    let crossed_medium = generate_large_crossed_df(20_000, 250, 100);
    let nested_small = generate_large_nested_df(50, 5, 8);
    let nested_medium = generate_large_nested_df(200, 10, 5);

    let crossed_small_fit =
        lme_rs::lmer("y ~ x + (1 | plate) + (1 | sample)", &crossed_small, false).unwrap();
    let crossed_medium_fit =
        lme_rs::lmer("y ~ x + (1 | plate) + (1 | sample)", &crossed_medium, false).unwrap();
    let nested_small_fit = lme_rs::lmer("y ~ x + (1 | batch/cask)", &nested_small, false).unwrap();
    let nested_medium_fit =
        lme_rs::lmer("y ~ x + (1 | batch/cask)", &nested_medium, false).unwrap();

    let crossed_small_pred = repeat_dataframe(&crossed_small, 5);
    let crossed_medium_pred = repeat_dataframe(&crossed_medium, 2);
    let nested_small_pred = repeat_dataframe(&nested_small, 5);
    let nested_medium_pred = repeat_dataframe(&nested_medium, 2);

    let mut group = c.benchmark_group("prediction_structure_sweeps");
    group.sample_size(10);

    group.bench_function("crossed_small_predict_population", |b| {
        b.iter(|| black_box(crossed_small_fit.predict(black_box(&crossed_small_pred))).unwrap())
    });
    group.bench_function("crossed_small_predict_conditional", |b| {
        b.iter(|| {
            black_box(
                crossed_small_fit
                    .predict_conditional(black_box(&crossed_small_pred), black_box(false)),
            )
            .unwrap()
        })
    });
    group.bench_function("crossed_medium_predict_population", |b| {
        b.iter(|| black_box(crossed_medium_fit.predict(black_box(&crossed_medium_pred))).unwrap())
    });
    group.bench_function("crossed_medium_predict_conditional", |b| {
        b.iter(|| {
            black_box(
                crossed_medium_fit
                    .predict_conditional(black_box(&crossed_medium_pred), black_box(false)),
            )
            .unwrap()
        })
    });

    group.bench_function("nested_small_predict_population", |b| {
        b.iter(|| black_box(nested_small_fit.predict(black_box(&nested_small_pred))).unwrap())
    });
    group.bench_function("nested_small_predict_conditional", |b| {
        b.iter(|| {
            black_box(
                nested_small_fit
                    .predict_conditional(black_box(&nested_small_pred), black_box(false)),
            )
            .unwrap()
        })
    });
    group.bench_function("nested_medium_predict_population", |b| {
        b.iter(|| black_box(nested_medium_fit.predict(black_box(&nested_medium_pred))).unwrap())
    });
    group.bench_function("nested_medium_predict_conditional", |b| {
        b.iter(|| {
            black_box(
                nested_medium_fit
                    .predict_conditional(black_box(&nested_medium_pred), black_box(false)),
            )
            .unwrap()
        })
    });

    group.finish();
}

fn bench_inference(c: &mut Criterion) {
    let df = load_csv("tests/data/sleepstudy.csv");
    let base_fit = lme_rs::lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let mut fit_for_type3 = base_fit.clone();
    fit_for_type3.with_satterthwaite(&df).unwrap();
    let mut fit_for_type3_kr = base_fit.clone();
    fit_for_type3_kr.with_kenward_roger(&df).unwrap();

    let lrt_fit_0 = lme_rs::lmer("Reaction ~ 1 + (1 | Subject)", &df, false).unwrap();
    let lrt_fit_1 = lme_rs::lmer("Reaction ~ Days + (Days | Subject)", &df, false).unwrap();

    let mut group = c.benchmark_group("inference");
    group.sample_size(10);

    group.bench_function("with_robust_se_hc0", |b| {
        b.iter_batched(
            || base_fit.clone(),
            |mut fit| {
                fit.with_robust_se(black_box(&df), black_box(None)).unwrap();
                black_box(fit)
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("with_robust_se_clustered", |b| {
        b.iter_batched(
            || base_fit.clone(),
            |mut fit| {
                fit.with_robust_se(black_box(&df), black_box(Some("Subject")))
                    .unwrap();
                black_box(fit)
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("with_satterthwaite", |b| {
        b.iter_batched(
            || base_fit.clone(),
            |mut fit| {
                fit.with_satterthwaite(black_box(&df)).unwrap();
                black_box(fit)
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("with_kenward_roger", |b| {
        b.iter_batched(
            || base_fit.clone(),
            |mut fit| {
                fit.with_kenward_roger(black_box(&df)).unwrap();
                black_box(fit)
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("type3_anova_satterthwaite", |b| {
        b.iter(|| black_box(fit_for_type3.anova(black_box(DdfMethod::Satterthwaite))).unwrap())
    });

    group.bench_function("type3_anova_kenward_roger", |b| {
        b.iter(|| black_box(fit_for_type3_kr.anova(black_box(DdfMethod::KenwardRoger))).unwrap())
    });

    group.bench_function("lrt_anova", |b| {
        b.iter(|| black_box(lme_rs::anova(black_box(&lrt_fit_0), black_box(&lrt_fit_1))).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_formula_parsing,
    bench_model_matrix_building,
    bench_deviance_evaluation,
    bench_lmer_end_to_end,
    bench_weighted_models,
    bench_glmer_end_to_end,
    bench_prediction,
    bench_glmm_post_fit,
    bench_lmer_large_synthetic,
    bench_large_structure_fits,
    bench_size_sweeps,
    bench_prediction_structure_sweeps,
    bench_inference
);
criterion_main!(benches);
