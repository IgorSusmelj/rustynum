use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple}; // Ensure PyList is imported
use pyo3::wrap_pyfunction;
use rustynum_rs::{NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8};

#[pyclass]
#[derive(Clone)]
struct PyNumArrayF32 {
    inner: NumArrayF32,
}

#[pyclass]
#[derive(Clone)]
struct PyNumArrayF64 {
    inner: NumArrayF64,
}

#[pyclass]
#[derive(Clone)]
struct PyNumArrayU8 {
    inner: NumArrayU8,
}

// #[pyclass]
// #[derive(Clone)]
// struct PyNumArrayI32 {
//     inner: NumArrayI32,
// }

// #[pyclass]
// #[derive(Clone)]
// struct PyNumArrayI64 {
//     inner: NumArrayI64,
// }

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
        ///self.inner = self.inner.scale(scalar);
        Ok(())
    }

    // Implement addition with a scalar value
    fn add_scalar(&self, scalar: f32) -> PyResult<PyNumArrayF32> {
        Python::with_gil(|py| {
            let result = &self.inner + scalar; // Leveraging Rust's Add implementation
            Ok(PyNumArrayF32 { inner: result })
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
        ///self.inner = self.inner.scale(scalar);
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
}

// #[pymethods]
// impl PyNumArrayI32 {
//     #[new]
//     fn new(data: Vec<i32>, shape: Option<Vec<usize>>) -> Self {
//         let inner = match shape {
//             Some(s) => NumArrayI32::new_with_shape(data, s),
//             None => NumArrayI32::new(data),
//         };
//         PyNumArrayI32 { inner }
//     }

//     fn add_scalar(&self, scalar: i32) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner + scalar; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i32> to Python list
//         })
//     }

//     fn add_array(&self, other: PyRef<PyNumArrayI32>) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner + &other.inner; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i32> to Python list
//         })
//     }

//     fn sub_scalar(&self, scalar: i32) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner - scalar; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i32> to Python list
//         })
//     }

//     fn sub_array(&self, other: PyRef<PyNumArrayI32>) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner - &other.inner; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i32> to Python list
//         })
//     }

//     fn mul_scalar(&self, scalar: i32) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner * scalar; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i32> to Python list
//         })
//     }

//     fn mul_array(&self, other: PyRef<PyNumArrayI32>) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner * &other.inner; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i32> to Python list
//         })
//     }

//     fn div_scalar(&self, scalar: i32) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner / scalar; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i32> to Python list
//         })
//     }

//     fn div_array(&self, other: PyRef<PyNumArrayI32>) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner / &other.inner; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i32> to Python list
//         })
//     }

//     fn tolist(&self, py: Python) -> PyObject {
//         let list = PyList::new(py, self.inner.get_data());
//         list.into()
//     }

//     fn slice(&self, start: usize, end: usize) -> PyResult<PyNumArrayI32> {
//         Ok(PyNumArrayI32 {
//             inner: self.inner.slice(start, end),
//         })
//     }

//     fn shape(&self) -> PyResult<PyObject> {
//         Python::with_gil(|py| {
//             let shape_vec = self.inner.shape();
//             Ok(PyTuple::new(py, shape_vec.iter()).to_object(py))
//         })
//     }

//     fn reshape(&self, shape: Vec<usize>) -> PyResult<PyNumArrayI32> {
//         Ok(PyNumArrayI32 {
//             inner: self.inner.reshape(&shape),
//         })
//     }

//     fn flip_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayI32> {
//         Python::with_gil(|py| {
//             let axis_vec: Vec<usize> = match axis {
//                 Some(list) => list.extract()?,
//                 None => vec![],
//             };
//             let result = if axis_vec.is_empty() {
//                 self.inner.clone()
//             } else {
//                 self.inner.flip_axis(axis_vec)
//             };
//             Ok(PyNumArrayI32 { inner: result })
//         })
//     }
// }

// #[pymethods]
// impl PyNumArrayI64 {
//     #[new]
//     fn new(data: Vec<i64>, shape: Option<Vec<usize>>) -> Self {
//         let inner = match shape {
//             Some(s) => NumArrayI64::new_with_shape(data, s),
//             None => NumArrayI64::new(data),
//         };
//         PyNumArrayI64 { inner }
//     }

//     fn add_scalar(&self, scalar: i64) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner + scalar; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i64> to Python list
//         })
//     }

//     fn add_array(&self, other: PyRef<PyNumArrayI64>) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner + &other.inner; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i64> to Python list
//         })
//     }

//     fn sub_scalar(&self, scalar: i64) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner - scalar; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i64> to Python list
//         })
//     }

//     fn sub_array(&self, other: PyRef<PyNumArrayI64>) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner - &other.inner; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i64> to Python list
//         })
//     }

//     fn mul_scalar(&self, scalar: i64) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner * scalar; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i64> to Python list
//         })
//     }

//     fn mul_array(&self, other: PyRef<PyNumArrayI64>) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner * &other.inner; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i64> to Python list
//         })
//     }

//     fn div_scalar(&self, scalar: i64) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner / scalar; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i64> to Python list
//         })
//     }

//     fn div_array(&self, other: PyRef<PyNumArrayI64>) -> PyResult<Py<PyAny>> {
//         Python::with_gil(|py| {
//             let result = &self.inner / &other.inner; // Leveraging Rust's Add implementation
//             Ok(result.get_data().to_object(py)) // Convert Vec<i64> to Python list
//         })
//     }

//     fn mean_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayI64> {
//         Python::with_gil(|py| {
//             let result = match axis {
//                 Some(axis_list) => {
//                     let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
//                     self.inner.mean_axis(Some(&axis_vec)) // Now correctly passing a slice wrapped in Some
//                 }
//                 None => self.inner.mean_axis(None),
//             };
//             Ok(PyNumArrayI64 { inner: result })
//         })
//     }

//     fn tolist(&self, py: Python) -> PyObject {
//         let list = PyList::new(py, self.inner.get_data());
//         list.into()
//     }

//     fn slice(&self, start: usize, end: usize) -> PyResult<PyNumArrayI64> {
//         Ok(PyNumArrayI64 {
//             inner: self.inner.slice(start, end),
//         })
//     }

//     fn shape(&self) -> PyResult<PyObject> {
//         Python::with_gil(|py| {
//             let shape_vec = self.inner.shape();
//             Ok(PyTuple::new(py, shape_vec.iter()).to_object(py))
//         })
//     }

//     fn reshape(&self, shape: Vec<usize>) -> PyResult<PyNumArrayI64> {
//         Ok(PyNumArrayI64 {
//             inner: self.inner.reshape(&shape),
//         })
//     }

//     fn flip_axis(&self, axis: Option<&PyList>) -> PyResult<PyNumArrayI64> {
//         Python::with_gil(|py| {
//             let axis_vec: Vec<usize> = match axis {
//                 Some(list) => list.extract()?,
//                 None => vec![],
//             };
//             let result = if axis_vec.is_empty() {
//                 self.inner.clone()
//             } else {
//                 self.inner.flip_axis(axis_vec)
//             };
//             Ok(PyNumArrayI64 { inner: result })
//         })
//     }

//     fn exp(&self) -> PyNumArrayI64 {
//         PyNumArrayI64 {
//             inner: self.inner.exp(),
//         }
//     }

//     fn log(&self) -> PyNumArrayI64 {
//         PyNumArrayI64 {
//             inner: self.inner.log(),
//         }
//     }

//     fn sigmoid(&self) -> PyNumArrayI64 {
//         PyNumArrayI64 {
//             inner: self.inner.sigmoid(),
//         }
//     }
// }

#[pyfunction]
fn zeros_f32(shape: Vec<usize>) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|py| {
        let result = NumArrayF32::zeros(shape);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
fn ones_f32(shape: Vec<usize>) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|py| {
        let result = NumArrayF32::ones(shape);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
fn matmul_f32(a: &PyNumArrayF32, b: &PyNumArrayF32) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|py| {
        // Ensure both arrays are matrices for matrix multiplication
        assert!(
            a.inner.shape().len() == 2 && b.inner.shape().len() == 2,
            "Both NumArrayF32 instances must be 2D for matrix multiplication."
        );
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
fn dot_f32(a: &PyNumArrayF32, b: &PyNumArrayF32) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|py| {
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
fn arange_f32(start: f32, end: f32, step: f32) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|py| {
        let result = NumArrayF32::arange(start, end, step);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
fn linspace_f32(start: f32, end: f32, num: usize) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|py| {
        let result = NumArrayF32::linspace(start, end, num);
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
fn mean_f32(a: &PyNumArrayF32, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.mean_axis(Some(&axis_vec))
            }
            None => a.inner.mean_axis(None), // Handle the case where no axis are provided
        };
        Ok(PyNumArrayF32 { inner: result }) // Convert the result data to a Python object
    })
}

#[pyfunction]
fn median_f32(a: &PyNumArrayF32, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.median_axis(Some(&axis_vec))
            }
            None => a.inner.median_axis(None), // Handle the case where no axis are provided
        };
        Ok(PyNumArrayF32 { inner: result }) // Convert the result data to a Python object
    })
}

#[pyfunction]
fn min_f32(a: &PyNumArrayF32) -> PyResult<f32> {
    Ok(a.inner.min())
}

#[pyfunction]
fn min_axis_f32(a: &PyNumArrayF32, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
    let result = match axis {
        Some(axis_list) => {
            let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
            a.inner.min_axis(Some(&axis_vec))
        }
        None => a.inner.min_axis(None),
    };
    Ok(PyNumArrayF32 { inner: result })
}

#[pyfunction]
fn max_f32(a: &PyNumArrayF32) -> PyResult<f32> {
    Ok(a.inner.max())
}

#[pyfunction]
fn max_axis_f32(a: &PyNumArrayF32, axis: Option<&PyList>) -> PyResult<PyNumArrayF32> {
    let result = match axis {
        Some(axis_list) => {
            let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
            a.inner.max_axis(Some(&axis_vec))
        }
        None => a.inner.max_axis(None),
    };
    Ok(PyNumArrayF32 { inner: result })
}

#[pyfunction]
fn exp_f32(a: &PyNumArrayF32) -> PyNumArrayF32 {
    PyNumArrayF32 {
        inner: a.inner.exp(),
    }
}

#[pyfunction]
fn log_f32(a: &PyNumArrayF32) -> PyNumArrayF32 {
    PyNumArrayF32 {
        inner: a.inner.log(),
    }
}

#[pyfunction]
fn sigmoid_f32(a: &PyNumArrayF32) -> PyNumArrayF32 {
    PyNumArrayF32 {
        inner: a.inner.sigmoid(),
    }
}

#[pyfunction]
fn concatenate_f32(arrays: Vec<PyNumArrayF32>, axis: usize) -> PyResult<PyNumArrayF32> {
    let rust_arrays: Vec<NumArrayF32> = arrays.iter().map(|array| array.inner.clone()).collect();
    let result = NumArrayF32::concatenate(&rust_arrays, axis);
    Ok(PyNumArrayF32 { inner: result })
}

#[pyfunction]
fn zeros_f64(shape: Vec<usize>) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|py| {
        let result = NumArrayF64::zeros(shape);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
fn ones_f64(shape: Vec<usize>) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|py| {
        let result = NumArrayF64::ones(shape);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
fn matmul_f64(a: &PyNumArrayF64, b: &PyNumArrayF64) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|py| {
        // Ensure both arrays are matrices for matrix multiplication
        assert!(
            a.inner.shape().len() == 2 && b.inner.shape().len() == 2,
            "Both NumArrayF64 instances must be 2D for matrix multiplication."
        );
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
fn dot_f64(a: &PyNumArrayF64, b: &PyNumArrayF64) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|py| {
        let result = a.inner.dot(&b.inner);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
fn arange_f64(start: f64, end: f64, step: f64) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|py| {
        let result = NumArrayF64::arange(start, end, step);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
fn linspace_f64(start: f64, end: f64, num: usize) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|py| {
        let result = NumArrayF64::linspace(start, end, num);
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
fn mean_f64(a: &PyNumArrayF64, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.mean_axis(Some(&axis_vec))
            }
            None => a.inner.mean_axis(None), // Handle the case where no axis are provided
        };
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
fn median_f64(a: &PyNumArrayF64, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
                a.inner.median_axis(Some(&axis_vec))
            }
            None => a.inner.median_axis(None), // Handle the case where no axis are provided
        };
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pyfunction]
fn min_f64(a: &PyNumArrayF64) -> PyResult<f64> {
    Ok(a.inner.min())
}

#[pyfunction]
fn min_axis_f64(a: &PyNumArrayF64, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
    let result = match axis {
        Some(axis_list) => {
            let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
            a.inner.min_axis(Some(&axis_vec))
        }
        None => a.inner.min_axis(None),
    };
    Ok(PyNumArrayF64 { inner: result })
}

#[pyfunction]
fn max_f64(a: &PyNumArrayF64) -> PyResult<f64> {
    Ok(a.inner.max())
}

#[pyfunction]
fn max_axis_f64(a: &PyNumArrayF64, axis: Option<&PyList>) -> PyResult<PyNumArrayF64> {
    let result = match axis {
        Some(axis_list) => {
            let axis_vec: Vec<usize> = axis_list.extract()?; // Convert PyList to Vec<usize>
            a.inner.max_axis(Some(&axis_vec))
        }
        None => a.inner.max_axis(None),
    };
    Ok(PyNumArrayF64 { inner: result })
}

#[pyfunction]
fn exp_f64(a: &PyNumArrayF64) -> PyNumArrayF64 {
    PyNumArrayF64 {
        inner: a.inner.exp(),
    }
}

#[pyfunction]
fn log_f64(a: &PyNumArrayF64) -> PyNumArrayF64 {
    PyNumArrayF64 {
        inner: a.inner.log(),
    }
}

#[pyfunction]
fn sigmoid_f64(a: &PyNumArrayF64) -> PyNumArrayF64 {
    PyNumArrayF64 {
        inner: a.inner.sigmoid(),
    }
}

#[pyfunction]
fn concatenate_f64(arrays: Vec<PyNumArrayF64>, axis: usize) -> PyResult<PyNumArrayF64> {
    let rust_arrays: Vec<NumArrayF64> = arrays.iter().map(|array| array.inner.clone()).collect();
    let result = NumArrayF64::concatenate(&rust_arrays, axis);
    Ok(PyNumArrayF64 { inner: result })
}

#[pyfunction]
fn norm_f32(
    a: &PyNumArrayF32,
    p: u32,
    axis: Option<&PyList>,
    keepdims: Option<bool>,
) -> PyResult<PyNumArrayF32> {
    Python::with_gil(|py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?;
                a.inner.norm(p, Some(&axis_vec), keepdims)
            }
            None => a.inner.norm(p, None, keepdims),
        };
        Ok(PyNumArrayF32 { inner: result })
    })
}

#[pyfunction]
fn norm_f64(
    a: &PyNumArrayF64,
    p: u32,
    axis: Option<&PyList>,
    keepdims: Option<bool>,
) -> PyResult<PyNumArrayF64> {
    Python::with_gil(|py| {
        let result = match axis {
            Some(axis_list) => {
                let axis_vec: Vec<usize> = axis_list.extract()?;
                a.inner.norm(p, Some(&axis_vec), keepdims)
            }
            None => a.inner.norm(p, None, keepdims),
        };
        Ok(PyNumArrayF64 { inner: result })
    })
}

#[pymodule]
fn _rustynum(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNumArrayF32>()?;
    m.add_class::<PyNumArrayF64>()?;
    m.add_class::<PyNumArrayU8>()?;

    m.add_function(wrap_pyfunction!(zeros_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ones_f32, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(dot_f32, m)?)?;
    m.add_function(wrap_pyfunction!(arange_f32, m)?)?;
    m.add_function(wrap_pyfunction!(linspace_f32, m)?)?;
    m.add_function(wrap_pyfunction!(mean_f32, m)?)?;
    m.add_function(wrap_pyfunction!(median_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_axis_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_axis_f32, m)?)?;
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
    m.add_function(wrap_pyfunction!(median_f64, m)?)?;
    m.add_function(wrap_pyfunction!(min_f64, m)?)?;
    m.add_function(wrap_pyfunction!(max_f64, m)?)?;
    m.add_function(wrap_pyfunction!(exp_f64, m)?)?;
    m.add_function(wrap_pyfunction!(log_f64, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_f64, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_f64, m)?)?;
    m.add_function(wrap_pyfunction!(norm_f32, m)?)?;
    m.add_function(wrap_pyfunction!(norm_f64, m)?)?;

    Ok(())
}
