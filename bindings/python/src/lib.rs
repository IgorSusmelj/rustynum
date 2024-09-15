use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple}; // Ensure PyList is imported
use pyo3::wrap_pyfunction;
use rustynum_rs::{NumArray32, NumArray64};

#[pyclass]
#[derive(Clone)]
struct PyNumArray32 {
    inner: NumArray32,
}

#[pyclass]
#[derive(Clone)]
struct PyNumArray64 {
    inner: NumArray64,
}

#[pymethods]
impl PyNumArray32 {
    #[new]
    fn new(data: Vec<f32>, shape: Option<Vec<usize>>) -> Self {
        let inner = match shape {
            Some(s) => NumArray32::new_with_shape(data, s),
            None => NumArray32::new(data),
        };
        PyNumArray32 { inner }
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

    fn mean_axes(&self, axes: Option<&PyList>) -> PyResult<PyNumArray32> {
        Python::with_gil(|py| {
            let result = match axes {
                Some(axes_list) => {
                    let axes_vec: Vec<usize> = axes_list.extract()?; // Convert PyList to Vec<usize>
                    self.inner.mean_axes(Some(&axes_vec)) // Now correctly passing a slice wrapped in Some
                }
                None => self.inner.mean(),
            };
            Ok(PyNumArray32 { inner: result })
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

    fn shape(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let shape_vec = self.inner.shape();
            Ok(PyTuple::new(py, shape_vec.iter()).to_object(py))
        })
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyNumArray32> {
        Ok(PyNumArray32 {
            inner: self.inner.reshape(&shape),
        })
    }

    fn exp(&self) -> PyNumArray32 {
        PyNumArray32 {
            inner: self.inner.exp(),
        }
    }

    fn log(&self) -> PyNumArray32 {
        PyNumArray32 {
            inner: self.inner.log(),
        }
    }

    fn sigmoid(&self) -> PyNumArray32 {
        PyNumArray32 {
            inner: self.inner.sigmoid(),
        }
    }
}

#[pymethods]
impl PyNumArray64 {
    #[new]
    fn new(data: Vec<f64>, shape: Option<Vec<usize>>) -> Self {
        let inner = match shape {
            Some(s) => NumArray64::new_with_shape(data, s),
            None => NumArray64::new(data),
        };
        PyNumArray64 { inner }
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

    fn mean_axes(&self, axes: Option<&PyList>) -> PyResult<PyNumArray64> {
        Python::with_gil(|py| {
            let result = match axes {
                Some(axes_list) => {
                    let axes_vec: Vec<usize> = axes_list.extract()?; // Convert PyList to Vec<usize>
                    self.inner.mean_axes(Some(&axes_vec)) // Now correctly passing a slice wrapped in Some
                }
                None => self.inner.mean_axes(None),
            };
            Ok(PyNumArray64 { inner: result })
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

    fn shape(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let shape_vec = self.inner.shape();
            Ok(PyTuple::new(py, shape_vec.iter()).to_object(py))
        })
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyNumArray64> {
        Ok(PyNumArray64 {
            inner: self.inner.reshape(&shape),
        })
    }

    fn exp(&self) -> PyNumArray64 {
        PyNumArray64 {
            inner: self.inner.exp(),
        }
    }

    fn log(&self) -> PyNumArray64 {
        PyNumArray64 {
            inner: self.inner.log(),
        }
    }

    fn sigmoid(&self) -> PyNumArray64 {
        PyNumArray64 {
            inner: self.inner.sigmoid(),
        }
    }
}

#[pyfunction]
fn zeros_f32(shape: Vec<usize>) -> PyResult<PyNumArray32> {
    Python::with_gil(|py| {
        let result = NumArray32::zeros(shape);
        Ok(PyNumArray32 { inner: result })
    })
}

#[pyfunction]
fn ones_f32(shape: Vec<usize>) -> PyResult<PyNumArray32> {
    Python::with_gil(|py| {
        let result = NumArray32::ones(shape);
        Ok(PyNumArray32 { inner: result })
    })
}

#[pyfunction]
fn matmul_f32(a: &PyNumArray32, b: &PyNumArray32) -> PyResult<PyNumArray32> {
    Python::with_gil(|py| {
        // Ensure both arrays are matrices for matrix multiplication
        assert!(
            a.inner.shape().len() == 2 && b.inner.shape().len() == 2,
            "Both NumArray32 instances must be 2D for matrix multiplication."
        );
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArray32 { inner: result })
    })
}

#[pyfunction]
fn dot_f32(a: &PyNumArray32, b: &PyNumArray32) -> PyResult<PyNumArray32> {
    Python::with_gil(|py| {
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArray32 { inner: result })
    })
}

#[pyfunction]
fn arange_f32(start: f32, end: f32, step: f32) -> PyResult<PyNumArray32> {
    Python::with_gil(|py| {
        let result = NumArray32::arange(start, end, step);
        Ok(PyNumArray32 { inner: result })
    })
}

#[pyfunction]
fn linspace_f32(start: f32, end: f32, num: usize) -> PyResult<PyNumArray32> {
    Python::with_gil(|py| {
        let result = NumArray32::linspace(start, end, num);
        Ok(PyNumArray32 { inner: result })
    })
}

#[pyfunction]
fn mean_f32(a: &PyNumArray32, axes: Option<&PyList>) -> PyResult<PyNumArray32> {
    Python::with_gil(|py| {
        let result = match axes {
            Some(axes_list) => {
                let axes_vec: Vec<usize> = axes_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.mean_axes(Some(&axes_vec))
            }
            None => a.inner.mean_axes(None), // Handle the case where no axes are provided
        };
        Ok(PyNumArray32 { inner: result }) // Convert the result data to a Python object
    })
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
fn exp_f32(a: &PyNumArray32) -> PyNumArray32 {
    PyNumArray32 {
        inner: a.inner.exp(),
    }
}

#[pyfunction]
fn log_f32(a: &PyNumArray32) -> PyNumArray32 {
    PyNumArray32 {
        inner: a.inner.log(),
    }
}

#[pyfunction]
fn sigmoid_f32(a: &PyNumArray32) -> PyNumArray32 {
    PyNumArray32 {
        inner: a.inner.sigmoid(),
    }
}

#[pyfunction]
fn concatenate_f32(arrays: Vec<PyNumArray32>, axis: usize) -> PyResult<PyNumArray32> {
    let rust_arrays: Vec<NumArray32> = arrays.iter().map(|array| array.inner.clone()).collect();
    let result = NumArray32::concatenate(&rust_arrays, axis);
    Ok(PyNumArray32 { inner: result })
}

#[pyfunction]
fn zeros_f64(shape: Vec<usize>) -> PyResult<PyNumArray64> {
    Python::with_gil(|py| {
        let result = NumArray64::zeros(shape);
        Ok(PyNumArray64 { inner: result })
    })
}

#[pyfunction]
fn ones_f64(shape: Vec<usize>) -> PyResult<PyNumArray64> {
    Python::with_gil(|py| {
        let result = NumArray64::ones(shape);
        Ok(PyNumArray64 { inner: result })
    })
}

#[pyfunction]
fn matmul_f64(a: &PyNumArray64, b: &PyNumArray64) -> PyResult<PyNumArray64> {
    Python::with_gil(|py| {
        // Ensure both arrays are matrices for matrix multiplication
        assert!(
            a.inner.shape().len() == 2 && b.inner.shape().len() == 2,
            "Both NumArray64 instances must be 2D for matrix multiplication."
        );
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArray64 { inner: result })
    })
}

#[pyfunction]
fn dot_f64(a: &PyNumArray64, b: &PyNumArray64) -> PyResult<PyNumArray64> {
    Python::with_gil(|py| {
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArray64 { inner: result })
    })
}

#[pyfunction]
fn arange_f64(start: f64, end: f64, step: f64) -> PyResult<PyNumArray64> {
    Python::with_gil(|py| {
        let result = NumArray64::arange(start, end, step);
        Ok(PyNumArray64 { inner: result })
    })
}

#[pyfunction]
fn linspace_f64(start: f64, end: f64, num: usize) -> PyResult<PyNumArray64> {
    Python::with_gil(|py| {
        let result = NumArray64::linspace(start, end, num);
        Ok(PyNumArray64 { inner: result })
    })
}

#[pyfunction]
fn mean_f64(a: &PyNumArray64, axes: Option<&PyList>) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        let result = match axes {
            Some(axes_list) => {
                let axes_vec: Vec<usize> = axes_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.mean_axes(Some(&axes_vec))
            }
            None => a.inner.mean_axes(None), // Handle the case where no axes are provided
        };
        Ok(result.get_data().to_object(py)) // Convert the result data to a Python object
    })
}

#[pyfunction]
fn min_f64(a: &PyNumArray64) -> PyResult<f64> {
    Ok(a.inner.min())
}

#[pyfunction]
fn max_f64(a: &PyNumArray64) -> PyResult<f64> {
    Ok(a.inner.max())
}

#[pyfunction]
fn exp_f64(a: &PyNumArray64) -> PyNumArray64 {
    PyNumArray64 {
        inner: a.inner.exp(),
    }
}

#[pyfunction]
fn log_f64(a: &PyNumArray64) -> PyNumArray64 {
    PyNumArray64 {
        inner: a.inner.log(),
    }
}

#[pyfunction]
fn sigmoid_f64(a: &PyNumArray64) -> PyNumArray64 {
    PyNumArray64 {
        inner: a.inner.sigmoid(),
    }
}

#[pyfunction]
fn concatenate_f64(arrays: Vec<PyNumArray64>, axis: usize) -> PyResult<PyNumArray64> {
    let rust_arrays: Vec<NumArray64> = arrays.iter().map(|array| array.inner.clone()).collect();
    let result = NumArray64::concatenate(&rust_arrays, axis);
    Ok(PyNumArray64 { inner: result })
}

#[pymodule]
fn _rustynum(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNumArray32>()?;
    m.add_class::<PyNumArray64>()?; // Ensure PyNumArray64 is also registered
    m.add_function(wrap_pyfunction!(zeros_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ones_f32, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(dot_f32, m)?)?;
    m.add_function(wrap_pyfunction!(arange_f32, m)?)?;
    m.add_function(wrap_pyfunction!(linspace_f32, m)?)?;
    m.add_function(wrap_pyfunction!(mean_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_f32, m)?)?;
    m.add_function(wrap_pyfunction!(exp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(log_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_f32, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_f32, m)?)?;

    m.add_function(wrap_pyfunction!(zeros_f64, m)?)?;
    m.add_function(wrap_pyfunction!(ones_f64, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f64, m)?)?;
    m.add_function(wrap_pyfunction!(dot_f64, m)?)?;
    m.add_function(wrap_pyfunction!(arange_f64, m)?)?;
    m.add_function(wrap_pyfunction!(linspace_f64, m)?)?;
    m.add_function(wrap_pyfunction!(mean_f64, m)?)?;
    m.add_function(wrap_pyfunction!(min_f64, m)?)?;
    m.add_function(wrap_pyfunction!(max_f64, m)?)?;
    m.add_function(wrap_pyfunction!(exp_f64, m)?)?;
    m.add_function(wrap_pyfunction!(log_f64, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_f64, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_f64, m)?)?;

    Ok(())
}
