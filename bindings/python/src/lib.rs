// bindings/python/src/lib.rs

// use pyo3::exceptions::PyValueError;
// use pyo3::prelude::*;
// use pyo3::types::{PyList, PyTuple}; // Ensure PyList is imported
// use pyo3::wrap_pyfunction;
// use rustynum_rs::{NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod array_f32;
mod array_f64;
mod array_u8;
mod functions;

use array_f32::PyNumArrayF32;
use array_f64::PyNumArrayF64;
use array_u8::PyNumArrayU8;

use functions::*;

// #[pyclass]
// #[derive(Clone)]
// struct PyNumArrayF32 {
//     inner: NumArrayF32,
// }

// #[pyclass]
// #[derive(Clone)]
// struct PyNumArrayF64 {
//     inner: NumArrayF64,
// }

// #[pyclass]
// #[derive(Clone)]
// struct PyNumArrayU8 {
//     inner: NumArrayU8,
// }

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
