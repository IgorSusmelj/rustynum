use pyo3::prelude::*;
use pyo3::types::PyList; // Ensure PyList is imported
use pyo3::wrap_pyfunction;
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

    // Implement addition with a scalar value
    fn add_scalar(&self, scalar: f32) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner + scalar; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f32> to Python list
        })
    }

    // Implement addition with another NumArray
    fn add_array(&self, other: PyRef<PyNumArray32>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner + &other.inner; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f32> to Python list
        })
    }

    fn sub_scalar(&self, scalar: f32) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner - scalar; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn sub_array(&self, other: PyRef<PyNumArray32>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner - &other.inner; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn mul_scalar(&self, scalar: f32) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner * scalar; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn mul_array(&self, other: PyRef<PyNumArray32>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner * &other.inner; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn div_scalar(&self, scalar: f32) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner / scalar; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn div_array(&self, other: PyRef<PyNumArray32>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner / &other.inner; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn tolist(&self, py: Python) -> PyObject {
        let list = PyList::new(py, self.inner.get_data());
        list.into()
    }

    fn slice(&self, start: usize, end: usize) -> PyResult<PyNumArray32> {
        Ok(PyNumArray32 {
            inner: self.inner.slice(start, end),
        })
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

    fn add_scalar(&self, scalar: f64) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner + scalar; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn add_array(&self, other: PyRef<PyNumArray64>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner + &other.inner; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn sub_scalar(&self, scalar: f64) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner - scalar; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn sub_array(&self, other: PyRef<PyNumArray64>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner - &other.inner; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn mul_scalar(&self, scalar: f64) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner * scalar; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn mul_array(&self, other: PyRef<PyNumArray64>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner * &other.inner; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn div_scalar(&self, scalar: f64) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner / scalar; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn div_array(&self, other: PyRef<PyNumArray64>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = &self.inner / &other.inner; // Leveraging Rust's Add implementation
            Ok(result.get_data().to_object(py)) // Convert Vec<f64> to Python list
        })
    }

    fn tolist(&self, py: Python) -> PyObject {
        let list = PyList::new(py, self.inner.get_data());
        list.into()
    }

    fn slice(&self, start: usize, end: usize) -> PyResult<PyNumArray64> {
        Ok(PyNumArray64 {
            inner: self.inner.slice(start, end),
        })
    }
}

#[pyfunction]
fn dot_f32(a: &PyNumArray32, b: &PyNumArray32) -> PyResult<f32> {
    Ok(a.inner.dot(&b.inner))
}

#[pyfunction]
fn mean_f32(a: &PyNumArray32) -> PyResult<f32> {
    Ok(a.inner.mean())
}

#[pyfunction]
fn min_f32(a: &PyNumArray32) -> PyResult<f32> {
    Ok(a.inner.min())
}

#[pyfunction]
fn max_f32(a: &PyNumArray32) -> PyResult<f32> {
    Ok(a.inner.max())
}

#[pyfunction]
fn dot_f64(a: &PyNumArray64, b: &PyNumArray64) -> PyResult<f64> {
    Ok(a.inner.dot(&b.inner))
}

#[pyfunction]
fn mean_f64(a: &PyNumArray64) -> PyResult<f64> {
    Ok(a.inner.mean())
}

#[pyfunction]
fn min_f64(a: &PyNumArray64) -> PyResult<f64> {
    Ok(a.inner.min())
}

#[pyfunction]
fn max_f64(a: &PyNumArray64) -> PyResult<f64> {
    Ok(a.inner.max())
}

#[pymodule]
fn _rustynum(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNumArray32>()?;
    m.add_class::<PyNumArray64>()?; // Ensure PyNumArray64 is also registered
    m.add_function(wrap_pyfunction!(dot_f32, m)?)?;
    m.add_function(wrap_pyfunction!(mean_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_f32, m)?)?;
    m.add_function(wrap_pyfunction!(dot_f64, m)?)?;
    m.add_function(wrap_pyfunction!(mean_f64, m)?)?;
    m.add_function(wrap_pyfunction!(min_f64, m)?)?;
    m.add_function(wrap_pyfunction!(max_f64, m)?)?;

    Ok(())
}
