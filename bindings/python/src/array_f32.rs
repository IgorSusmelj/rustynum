// bindings/python/src/array_f32.rs

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple}; // Keep only what's actually used by PyNumArrayF32 methods
use rustynum_rs::NumArrayF32;

#[pyclass(module = "_rustynum")]
#[derive(Clone)]
pub struct PyNumArrayF32 {
    pub(crate) inner: NumArrayF32,
}

#[pymethods]
impl PyNumArrayF32 {
    #[new]
    fn new(data: Vec<f32>, shape: Option<Vec<usize>>) -> Self {
        let inner = match shape {
            Some(s) => NumArrayF32::new_with_shape(data, s),
            None => NumArrayF32::new(data),
        };
        PyNumArrayF32 { inner }
    }

    fn __imul__(&mut self, scalar: f32) -> PyResult<()> {
        let original_shape = self.inner.shape().to_vec();
        let result_data: Vec<f32> = self.inner.get_data().iter().map(|&x| x * scalar).collect();
        self.inner = NumArrayF32::new_with_shape(result_data, original_shape);
        Ok(())
    }

    // Implement addition with a scalar value
    fn add_scalar(&self, scalar: f32) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = &self.inner + scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    // Implement addition with another NumArray
    fn add_array(&self, other: PyRef<PyNumArrayF32>) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = &self.inner + &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn sub_scalar(&self, scalar: f32) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = &self.inner - scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn sub_array(&self, other: PyRef<PyNumArrayF32>) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = &self.inner - &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn mul_scalar(&self, scalar: f32) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = &self.inner * scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn mul_array(&self, other: PyRef<PyNumArrayF32>) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = &self.inner * &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn div_scalar(&self, scalar: f32) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = &self.inner / scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn div_array(&self, other: PyRef<PyNumArrayF32>) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = &self.inner / &other.inner;
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn mean_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
                    self.inner.mean_axis(Some(&axis_vec)) // Now correctly passing a slice wrapped in Some
                }
                None => self.inner.mean(),
            };
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn median_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?;
                    self.inner.median_axis(Some(&axis_vec))
                }
                None => self.inner.median_axis(None),
            };
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn norm(
        &self,
        p: u32,
        axis: Option<&PyList>,
        keepdims: Option<bool>,
    ) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?;
                    self.inner.norm(p, Some(&axis_vec), keepdims)
                }
                None => self.inner.norm(p, None, keepdims),
            };
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn tolist(&self, py: Python) -> PyObject {
        let list = PyList::new(py, self.inner.get_data());
        list.into()
    }

    fn slice(&self, axis: usize, start: usize, end: usize) -> PyResult<PyNumArrayF32> {
        Ok(PyNumArrayF32 {
            inner: self.inner.slice(axis, start, end),
        })
    }

    fn shape(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let shape_vec = self.inner.shape();
            Ok(PyTuple::new(py, shape_vec.iter()).to_object(py))
        })
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyNumArrayF32> {
        Ok(PyNumArrayF32 {
            inner: self.inner.reshape(&shape),
        })
    }

    fn flip_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
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
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn exp(&self) -> PyNumArrayF32 {
        PyNumArrayF32 {
            inner: self.inner.exp(),
        }
    }

    fn log(&self) -> PyNumArrayF32 {
        PyNumArrayF32 {
            inner: self.inner.log(),
        }
    }

    fn sigmoid(&self) -> PyNumArrayF32 {
        PyNumArrayF32 {
            inner: self.inner.sigmoid(),
        }
    }

    fn min_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?;
                    self.inner.min_axis(Some(&axis_vec))
                }
                None => self.inner.min_axis(None),
            };
            Ok(PyNumArrayF32 { inner: result })
        })
    }

    fn max_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = match axis {
                Some(axis_list) => {
                    let axis_vec: Vec<usize> = axis_list.extract()?;
                    self.inner.max_axis(Some(&axis_vec))
                }
                None => self.inner.max_axis(None),
            };
            Ok(PyNumArrayF32 { inner: result })
        })
    }
}
