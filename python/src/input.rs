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
    required: Option<&std::collections::HashSet<String>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(module) = data.get_type().getattr("__module__") {
        if let Ok(mod_name) = module.extract::<String>() {
            if mod_name.starts_with("polars") {
                return project_columns(data, required, "columns", false);
            }
        }
    }

    let polars = py.import("polars")?;

    if let Ok(pandas) = py.import("pandas") {
        let pandas_df = pandas.getattr("DataFrame")?;
        if data.is_instance(&pandas_df)? {
            let projected = project_columns(data, required, "columns", true)?;
            return polars.call_method1("from_pandas", (projected,));
        }
    }

    if let Ok(pyarrow) = py.import("pyarrow") {
        let table_cls = pyarrow.getattr("Table")?;
        if data.is_instance(&table_cls)? {
            let projected = project_columns(data, required, "column_names", false)?;
            return polars.call_method1("from_arrow", (projected,));
        }
    }

    if data.getattr("write_ipc").is_ok() {
        return project_columns(data, required, "columns", false);
    }

    Err(dataframe_input_error())
}

/// Project before pandas/Arrow conversion: unrelated application metadata may
/// not have a tabular representation at all. Preserve the caller's row order.
fn project_columns<'py>(
    data: &Bound<'py, PyAny>,
    required: Option<&std::collections::HashSet<String>>,
    columns_attr: &str,
    pandas: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(required) = required else {
        return Ok(data.clone());
    };
    let available = data.getattr(columns_attr)?;
    let mut columns = Vec::<String>::new();
    for item in available.try_iter()? {
        // Explicit formulas name string columns. Ignore unrelated non-string
        // pandas column labels rather than making metadata break projection.
        if let Ok(name) = item?.extract::<String>() {
            if required.contains(&name) {
                columns.push(name);
            }
        }
    }
    if columns.is_empty() {
        return Ok(data.clone());
    }
    if pandas {
        data.get_item(columns)
    } else {
        data.call_method1("select", (columns,))
    }
}

pub(super) fn read_dataframe(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    formula: Option<&str>,
    extra: &[&str],
) -> PyResult<DataFrame> {
    let mut projected_columns = None;
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
                projected_columns = Some(required);
            }
        }
    }
    let mut pl_df = to_polars_dataframe(py, data, projected_columns.as_ref())?;
    // Python Polars uses dictionary arrays for pandas categories and Enum.
    // The Rust IPC reader does not enable dtype-categorical; sending dictionaries
    // would panic. Labels preserve the engine's existing categorical semantics.
    let polars = py.import("polars")?;
    let categorical = polars.getattr("Categorical")?;
    let enum_dtype = polars.getattr("Enum")?;
    let schema = pl_df.getattr("schema")?;
    let mut labels = Vec::<String>::new();
    for pair in schema.call_method0("items")?.try_iter()? {
        let pair = pair?;
        let dtype = pair.get_item(1)?.call_method0("base_type")?;
        if dtype.eq(&categorical)? || dtype.eq(&enum_dtype)? {
            labels.push(pair.get_item(0)?.extract()?);
        }
    }
    if !labels.is_empty() {
        let expression = polars
            .call_method1("col", (labels,))?
            .call_method1("cast", (polars.getattr("String")?,))?;
        pl_df = pl_df.call_method1("with_columns", (expression,))?;
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
