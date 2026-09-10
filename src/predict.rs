//! Prediction using training encodings and new-data offsets.
use crate::{family, LmeFit};

impl LmeFit {
    fn is_nlmm(&self) -> bool {
        self.family_name.as_deref() == Some("nlmm")
    }

    /// Predict from fixed effects and offsets, omitting group effects (`re.form=NA`).
    ///
    /// For LMMs and GLMMs, returns `X_new * beta + offset`. GLMM results are on the
    /// linear-predictor scale; use [`Self::predict_response`] for response means.
    ///
    /// For nonlinear mixed models (`nlmer`), returns the mean function evaluated at the fixed
    /// nonlinear parameters only (random effects set to zero).
    pub fn predict(
        &self,
        newdata: &polars::prelude::DataFrame,
    ) -> anyhow::Result<ndarray::Array1<f64>> {
        if self.is_nlmm() {
            return crate::nlmm::predict::predict_population(self, newdata);
        }

        let (x, offset) = self.model_spec().prediction_matrix(newdata)?;
        let mut y_pred = x.dot(&self.coefficients);
        if let Some(offset) = offset {
            y_pred += &offset;
        }
        Ok(y_pred)
    }

    /// Predict on the response scale (applies inverse link for GLMMs).
    ///
    /// For LMMs this is identical to `predict()`. For GLMMs it applies the inverse link
    /// function to transform the linear predictor to the response scale (e.g., probabilities
    /// for binomial, counts for Poisson).
    pub fn predict_response(
        &self,
        newdata: &polars::prelude::DataFrame,
    ) -> anyhow::Result<ndarray::Array1<f64>> {
        let eta = self.predict(newdata)?;
        self.apply_inverse_link(eta)
    }

    /// Predict conditional expectations on the response scale (applies inverse link for GLMMs).
    ///
    /// Combines fixed + random effects, then applies the inverse link.
    pub fn predict_conditional_response(
        &self,
        newdata: &polars::prelude::DataFrame,
        allow_new_levels: bool,
    ) -> anyhow::Result<ndarray::Array1<f64>> {
        let eta = self.predict_conditional(newdata, allow_new_levels)?;
        self.apply_inverse_link(eta)
    }

    /// Apply the inverse link function if this is a GLMM, otherwise return as-is.
    fn apply_inverse_link(
        &self,
        eta: ndarray::Array1<f64>,
    ) -> anyhow::Result<ndarray::Array1<f64>> {
        match &self.family {
            Some(fam_enum) => {
                let link = self
                    .link_name
                    .as_deref()
                    .map(family::Link::parse)
                    .transpose()
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .unwrap_or_else(|| family::Link::default_for(*fam_enum));
                let fam_impl = fam_enum
                    .build_with_link(link)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let link_fn = fam_impl.link();
                Ok(link_fn.link_inv(&eta))
            }
            None => Ok(eta), // LMM: identity, return as-is
        }
    }

    /// Predict using fixed effects, offsets, and stored group effects (`re.form=NULL`).
    ///
    /// For LMMs and GLMMs, returns `X_new * beta + offset + Z_new * b`. GLMM
    /// results are on the linear-predictor scale; use
    /// [`Self::predict_conditional_response`] for response means.
    ///
    /// For nonlinear mixed models (`nlmer`), adds stored random effects on the nonlinear parameter
    /// (e.g. `Asym + b_group`) before evaluating the mean function.
    ///
    /// If `allow_new_levels` is true, unseen groups receive zero random-effect
    /// contributions. Otherwise, an unseen group returns an error.
    pub fn predict_conditional(
        &self,
        newdata: &polars::prelude::DataFrame,
        allow_new_levels: bool,
    ) -> anyhow::Result<ndarray::Array1<f64>> {
        if self.is_nlmm() {
            return crate::nlmm::predict::predict_conditional(self, newdata, allow_new_levels);
        }

        let y_pop = self.predict(newdata)?;

        let b = self.b.as_ref().ok_or_else(|| {
            anyhow::anyhow!("No random effects available for conditional predictions")
        })?;
        let re_blocks = self
            .re_blocks
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No RE block metadata available"))?;

        let n_obs = newdata.height();
        let mut z_b = ndarray::Array1::<f64>::zeros(n_obs);

        let mut b_offset = 0;
        for block in re_blocks {
            let (group_indices, group_labels, _) = if block.group_name.contains(':') {
                crate::model_matrix::try_build_interaction_groups(
                    newdata,
                    &block.group_name.split(':').collect::<Vec<_>>(),
                    n_obs,
                )?
                .ok_or_else(|| crate::LmeError::MissingColumn {
                    column: block.group_name.clone(),
                })?
            } else {
                crate::model_matrix::build_grouping_from_string_column(
                    newdata,
                    &block.group_name,
                    n_obs,
                )?
            };

            // Collect slope covariate data for non-intercept effects
            let has_intercept = block
                .effect_names
                .first()
                .is_some_and(|n| n == "(Intercept)");
            let mut slope_data: Vec<Vec<f64>> = Vec::new();
            for effect_name in &block.effect_names {
                if effect_name == "(Intercept)" {
                    continue;
                }
                slope_data
                    .push(crate::model_matrix::numeric_column_f64(newdata, effect_name)?.to_vec());
            }

            for (i, &index) in group_indices.iter().enumerate() {
                let group_name = group_labels[index].as_str();

                // Look up group index from the stored mapping; unknown groups get 0 contribution
                let group_idx = match block.group_map.get(group_name) {
                    Some(&idx) => idx,
                    None => {
                        if !allow_new_levels {
                            return Err(crate::LmeError::NewLevel {
                                group: block.group_name.clone(),
                                level: group_name.into(),
                            }
                            .into());
                        }
                        // Unknown group → population-level (no RE contribution)
                        continue;
                    }
                };

                let base = b_offset + group_idx * block.k;
                let mut effect_idx = 0;

                if has_intercept {
                    z_b[i] += b[base + effect_idx]; // intercept contribution (1.0 * b_intercept)
                    effect_idx += 1;
                }

                for (s_idx, s_vec) in slope_data.iter().enumerate() {
                    z_b[i] += s_vec[i] * b[base + effect_idx + s_idx];
                }
            }

            b_offset += block.m * block.k;
        }

        Ok(y_pop + z_b)
    }
}
