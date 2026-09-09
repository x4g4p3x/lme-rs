//! Tabular normalization and projected IPC interchange.
use super::*;
fn dataframe_input_error() -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(
        "data must be a polars.DataFrame, pandas.DataFrame, or pyarrow.Table \
         (install pandas or pyarrow if needed)",
    )
}

/// Normalize Python tabular input to a Polars ``DataFrame`` for IPC serialization.
fn to_polars_dataframe<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(module) = data.get_type().getattr("__module__") {
        if let Ok(mod_name) = module.extract::<String>() {
            if mod_name.starts_with("polars") {
                return Ok(data.clone());
            }
        }
    }

    let polars = py.import("polars")?;

    if let Ok(pandas) = py.import("pandas") {
        let pandas_df = pandas.getattr("DataFrame")?;
        if data.is_instance(&pandas_df)? {
            return polars.call_method1("from_pandas", (data,));
        }
    }

    if let Ok(pyarrow) = py.import("pyarrow") {
        let table_cls = pyarrow.getattr("Table")?;
        if data.is_instance(&table_cls)? {
            return polars.call_method1("from_arrow", (data,));
        }
    }

    if data.getattr("write_ipc").is_ok() {
        return Ok(data.clone());
    }

    Err(dataframe_input_error())
}

pub(super) fn read_dataframe(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    formula: Option<&str>,
    extra: &[&str],
) -> PyResult<DataFrame> {
    let mut pl_df = to_polars_dataframe(py, data)?;
    if let Some(formula) = formula {
        if let Ok(ast) = lme_rs::formula::parse(formula) {
            let dot = ast
                .columns
                .values()
                .any(|c| matches!(c.basis, Some(lme_rs::formula::BasisSpec::Dot)));
            if !dot {
                let mut required = std::collections::HashSet::<String>::new();
                for (name, info) in &ast.columns {
                    required.insert(name.clone());
                    if info.has_role(lme_rs::formula::ColumnRole::GroupingVariable) {
                        required.extend(name.split(':').map(str::to_owned));
                    }
                    if let Some(expr) = &info.expr {
                        expr.for_each_column(&mut |c| {
                            required.insert(c.to_owned());
                        });
                    }
                    if let Some(basis) = &info.basis {
                        basis.for_each_column(&mut |c| {
                            required.insert(c.to_owned());
                        });
                    }
                }
                if let Some(offset) = &ast.offset {
                    offset.for_each_column(&mut |c| {
                        required.insert(c.to_owned());
                    });
                }
                required.extend(extra.iter().map(|s| s.to_string()));
                let available: Vec<String> = pl_df.getattr("columns")?.extract()?;
                let columns: Vec<String> = available
                    .into_iter()
                    .filter(|s| required.contains(s))
                    .collect();
                if !columns.is_empty() {
                    pl_df = pl_df.call_method1("select", (columns,))?;
                }
            }
        }
    }
    let io = py.import("io")?;
    let buffer = io.call_method0("BytesIO")?;
    pl_df.call_method1("write_ipc", (&buffer,))?;
    let value = buffer.call_method0("getvalue")?;
    let bytes = value.cast::<PyBytes>()?;
    let raw = bytes.as_bytes();
    py.detach(|| {
        IpcReader::new(Cursor::new(raw)).finish().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to parse dataframe from IPC: {e}"
            ))
        })
    })
}
