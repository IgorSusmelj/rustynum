// bindings/python/src/array_f64.rs

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple}; // Keep only what's actually used by PyNumArrayF32 methods
use rustynum_rs::NumArrayF64;

#[pyclass(module = "_rustynum")]
#[derive(Clone)]
pub struct PyNumArrayF64 {
    pub(crate) inner: NumArrayF64,
}

#[pymethods]
impl PyNumArrayF64 {
    #[new]
    fn new(data: Vec<f64>, shape: Option<Vec<usize>>) -> Self {
        let inner = match shape {
            Some(s) => NumArrayF64::new_with_shape(data, s),
            None => NumArrayF64::new(data),
        };
        PyNumArrayF64 { inner }
    }

    fn __imul__(&mut self, scalar: f64) -> PyResult<()> {
        let original_shape = self.inner.shape().to_vec();
        let result_data: Vec<f64> = self.inner.get_data().iter().map(|&x| x * scalar).collect();
        self.inner = NumArrayF64::new_with_shape(result_data, original_shape);
        Ok(())
    }

    fn add_scalar(&self, scalar: f64) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = &self.inner + scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn add_array(&self, other: PyRef<PyNumArrayF64>) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = &self.inner + &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn sub_scalar(&self, scalar: f64) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = &self.inner - scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn sub_array(&self, other: PyRef<PyNumArrayF64>) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = &self.inner - &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn mul_scalar(&self, scalar: f64) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = &self.inner * scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn mul_array(&self, other: PyRef<PyNumArrayF64>) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = &self.inner * &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn div_scalar(&self, scalar: f64) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = &self.inner / scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn div_array(&self, other: PyRef<PyNumArrayF64>) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = &self.inner / &other.inner;
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn mean_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?;
                    self.inner.mean_axis(Some(&axis_vec))
                }
                None => self.inner.mean_axis(None),
            };
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn median_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?;
                    self.inner.median_axis(Some(&axis_vec))
                }
                None => self.inner.median_axis(None),
            };
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn norm(
        &self,
        p: u32,
        axis: Option<&PyList>,
        keepdims: Option<bool>,
    ) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?;
                    self.inner.norm(p, Some(&axis_vec), keepdims)
                }
                None => self.inner.norm(p, None, keepdims),
            };
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn tolist(&self, py: Python) -> PyObject {
        let list = PyList::new(py, self.inner.get_data());
        list.into()
    }

    fn slice(&self, axis: usize, start: usize, end: usize) -> PyResult<PyNumArrayF64> {
        Ok(PyNumArrayF64 {
            inner: self.inner.slice(axis, start, end),
        })
    }

    fn shape(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let shape_vec = self.inner.shape();
            Ok(PyTuple::new(py, shape_vec.iter()).to_object(py))
        })
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyNumArrayF64> {
        Ok(PyNumArrayF64 {
            inner: self.inner.reshape(&shape),
        })
    }

    fn flip_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let axis_vec: Vec<usize> = match axis {
                Some(list) => list.extract()?,
                None => vec![],
            };
            let result = if axis_vec.is_empty() {
                self.inner.clone()
            } else {
                self.inner.flip_axis(axis_vec)
            };
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn exp(&self) -> PyNumArrayF64 {
        PyNumArrayF64 {
            inner: self.inner.exp(),
        }
    }

    fn log(&self) -> PyNumArrayF64 {
        PyNumArrayF64 {
            inner: self.inner.log(),
        }
    }

    fn sigmoid(&self) -> PyNumArrayF64 {
        PyNumArrayF64 {
            inner: self.inner.sigmoid(),
        }
    }

    fn min_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?;
                    self.inner.min_axis(Some(&axis_vec))
                }
                None => self.inner.min_axis(None),
            };
            Ok(PyNumArrayF64 { inner: result })
        })
    }

    fn max_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?;
                    self.inner.max_axis(Some(&axis_vec))
                }
                None => self.inner.max_axis(None),
            };
            Ok(PyNumArrayF64 { inner: result })
        })
    }
}
