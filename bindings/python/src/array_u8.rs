// bindings/python/src/array_u8.rs
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rustynum_rs::NumArrayU8;

#[pyclass(module = "_rustynum")]
#[derive(Clone)]
pub struct PyNumArrayU8 {
    pub(crate) inner: NumArrayU8,
}

#[pymethods]
impl PyNumArrayU8 {
    #[new]
    fn new(data: Vec<u8>, shape: Option<Vec<usize>>) -> Self {
        let inner = match shape {
            Some(s) => NumArrayU8::new_with_shape(data, s),
            None => NumArrayU8::new(data),
        };
        PyNumArrayU8 { inner }
    }

    fn add_scalar(&self, scalar: u8) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|py| {
            let result = &self.inner + scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn add_array(&self, other: PyRef<PyNumArrayU8>) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|py| {
            let result = &self.inner + &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn sub_scalar(&self, scalar: u8) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|py| {
            let result = &self.inner - scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn sub_array(&self, other: PyRef<PyNumArrayU8>) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|py| {
            let result = &self.inner - &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn mul_scalar(&self, scalar: u8) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|py| {
            let result = &self.inner * scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn mul_array(&self, other: PyRef<PyNumArrayU8>) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|py| {
            let result = &self.inner * &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn div_scalar(&self, scalar: u8) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|py| {
            let result = &self.inner / scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn div_array(&self, other: PyRef<PyNumArrayU8>) -> PyResult<PyNumArrayU8> {
        Python::with_gil(|py| {
            let result = &self.inner / &other.inner; // Leveraging Rust's Add implementation
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn tolist(&self, py: Python) -> PyObject {
        let list = PyList::new(py, self.inner.get_data());
        list.into()
    }

    fn slice(&self, axis: usize, start: usize, end: usize) -> PyResult<PyNumArrayU8> {
        Ok(PyNumArrayU8 {
            inner: self.inner.slice(axis, start, end),
        })
    }

    fn shape(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let shape_vec = self.inner.shape();
            Ok(PyTuple::new(py, shape_vec.iter()).to_object(py))
        })
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyNumArrayU8> {
        Ok(PyNumArrayU8 {
            inner: self.inner.reshape(&shape),
        })
    }

    fn flip_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayU8> {
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
            Ok(PyNumArrayU8 { inner: result })
        })
    }

    fn __imul__(&mut self, scalar: u8) -> PyResult<()> {
        self.inner = &self.inner * scalar;
        Ok(())
    }
}
