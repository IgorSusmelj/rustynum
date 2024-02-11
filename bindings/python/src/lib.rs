use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList; // Ensure PyList is imported
use rustynum_rs::{NumArray32, NumArray64};


#[pyclass]
struct PyNumArray32 {
    inner: NumArray32,
}

#[pyclass]
struct PyNumArray64 {
    inner: NumArray64,
}

#[pymethods]
impl PyNumArray32 {
    #[new]
    fn new(data: Vec<f32>) -> Self {
        PyNumArray32 {
            inner: NumArray32::new(data),
        }
    }

    fn __imul__(&mut self, scalar: f32) -> PyResult<()> {
        ///self.inner = self.inner.scale(scalar);
        Ok(())
    }

    fn tolist(&self, py: Python) -> PyObject {
        let list = PyList::new(py, self.inner.get_data());
        list.into()
    }
}

#[pymethods]
impl PyNumArray64 {
    #[new]
    fn new(data: Vec<f64>) -> Self {
        PyNumArray64 {
            inner: NumArray64::new(data),
        }
    }

    fn __imul__(&mut self, scalar: f64) -> PyResult<()> {
        ///self.inner = self.inner.scale(scalar);
        Ok(())
    }

    fn tolist(&self, py: Python) -> PyObject {
        let list = PyList::new(py, self.inner.get_data());
        list.into()
    }
}

#[pyfunction]
fn dot_f32(a: &PyNumArray32, b: &PyNumArray32) -> PyResult<f32> {
    Ok(a.inner.dot(&b.inner))
}

#[pyfunction]
fn dot_f64(a: &PyNumArray64, b: &PyNumArray64) -> PyResult<f64> {
    Ok(a.inner.dot(&b.inner))
}

#[pymodule]
fn _rustynum(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNumArray32>()?;
    m.add_class::<PyNumArray64>()?; // Ensure PyNumArray64 is also registered
    m.add_function(wrap_pyfunction!(dot_f32, m)?)?;
    m.add_function(wrap_pyfunction!(dot_f64, m)?)?;

    Ok(())
}
