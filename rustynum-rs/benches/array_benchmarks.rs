use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use rustynum_rs::num_array::linalg::matrix_multiply;
use rustynum_rs::NumArray32; // Adjust this to the path to your library

// Define the operations you want to benchmark
enum Operation {
    AddVectors,
    Mean,
    DotProduct,
    MatrixVectorMultiplication,
    MatrixMatrixMultiplication,
}

// Helper functions to create data
fn create_vector(size: usize) -> Vec<f32> {
    (0..size).map(|x| x as f32).collect()
}

fn create_matrix(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|x| x as f32).collect()
}

// Generic benchmark function for vector operations
fn benchmark_vector_operation(c: &mut Criterion, op: Operation, sizes: &[usize]) {
    let mut group = c.benchmark_group(match op {
        Operation::AddVectors => "Vector Addition",
        Operation::Mean => "Vector Mean",
        Operation::DotProduct => "Vector Dot Product",
        _ => "Unknown Operation",
    });

    for &size in sizes {
        // Setup data
        let a_rustynum = NumArray32::new(create_vector(size));
        let b_rustynum = if matches!(op, Operation::AddVectors | Operation::DotProduct) {
            Some(NumArray32::new(create_vector(size)))
        } else {
            None
        };

        let a_ndarray = Array1::from_vec(create_vector(size));
        let b_ndarray = if matches!(op, Operation::AddVectors | Operation::DotProduct) {
            Some(Array1::from_vec(create_vector(size)))
        } else {
            None
        };

        let a_nalgebra = DVector::from_vec(create_vector(size));
        let b_nalgebra = if matches!(op, Operation::AddVectors | Operation::DotProduct) {
            Some(DVector::from_vec(create_vector(size)))
        } else {
            None
        };

        // Set throughput for better reporting
        group.throughput(Throughput::Elements(size as u64));

        match op {
            Operation::AddVectors => {
                group.bench_with_input(
                    BenchmarkId::new("rustynum_rs", size),
                    &size,
                    |bencher, &_| {
                        bencher.iter(|| {
                            black_box(&a_rustynum) + black_box(b_rustynum.as_ref().unwrap())
                        })
                    },
                );

                group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bencher, &_| {
                    bencher.iter(|| black_box(&a_ndarray) + black_box(b_ndarray.as_ref().unwrap()))
                });

                group.bench_with_input(BenchmarkId::new("nalgebra", size), &size, |bencher, &_| {
                    bencher
                        .iter(|| black_box(&a_nalgebra) + black_box(b_nalgebra.as_ref().unwrap()))
                });
            }
            Operation::Mean => {
                group.bench_with_input(
                    BenchmarkId::new("rustynum_rs", size),
                    &size,
                    |bencher, &_| bencher.iter(|| black_box(a_rustynum.mean())),
                );

                group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bencher, &_| {
                    bencher.iter(|| black_box(a_ndarray.mean().unwrap()))
                });

                group.bench_with_input(BenchmarkId::new("nalgebra", size), &size, |bencher, &_| {
                    bencher.iter(|| {
                        black_box(a_nalgebra.iter().sum::<f32>() / a_nalgebra.len() as f32)
                    })
                });
            }
            Operation::DotProduct => {
                group.bench_with_input(
                    BenchmarkId::new("rustynum_rs", size),
                    &size,
                    |bencher, &_| {
                        bencher.iter(|| black_box(a_rustynum.dot(&b_rustynum.as_ref().unwrap())))
                    },
                );

                group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bencher, &_| {
                    bencher.iter(|| black_box(a_ndarray.dot(b_ndarray.as_ref().unwrap())))
                });

                group.bench_with_input(BenchmarkId::new("nalgebra", size), &size, |bencher, &_| {
                    bencher.iter(|| black_box(a_nalgebra.dot(&b_nalgebra.as_ref().unwrap())))
                });
            }
            _ => {}
        }
    }

    group.finish();
}

// Generic benchmark function for matrix operations
fn benchmark_matrix_operation(c: &mut Criterion, op: Operation, sizes: &[usize]) {
    let mut group = c.benchmark_group(match op {
        Operation::MatrixVectorMultiplication => "Matrix-Vector Multiplication",
        Operation::MatrixMatrixMultiplication => "Matrix-Matrix Multiplication",
        _ => "Unknown Operation",
    });

    for &size in sizes {
        // Assuming square matrices for simplicity; adjust as needed
        let rows = size;
        let cols = size;

        // Setup data
        let matrix1 = create_matrix(rows, cols);
        let matrix2 = if matches!(op, Operation::MatrixMatrixMultiplication) {
            Some(create_matrix(cols, rows))
        } else {
            None
        };
        let vector = create_vector(cols);

        let num_array_matrix1 = NumArray32::new_with_shape(matrix1.clone(), vec![rows, cols]);
        let num_array_matrix2 = if let Some(ref m2) = matrix2 {
            Some(NumArray32::new_with_shape(m2.clone(), vec![cols, rows]))
        } else {
            None
        };
        let num_array_vector = if matches!(op, Operation::MatrixVectorMultiplication) {
            Some(NumArray32::new_with_shape(vector.clone(), vec![cols]))
        } else {
            None
        };

        let ndarray_matrix1 = Array2::from_shape_vec((rows, cols), matrix1.clone()).unwrap();
        let ndarray_matrix2 = if let Some(ref m2) = matrix2 {
            Some(Array2::from_shape_vec((cols, rows), m2.clone()).unwrap())
        } else {
            None
        };
        let ndarray_vector = if matches!(op, Operation::MatrixVectorMultiplication) {
            Some(Array1::from_vec(vector.clone()))
        } else {
            None
        };

        let nalgebra_matrix1 = DMatrix::from_vec(rows, cols, matrix1.clone());
        let nalgebra_matrix2 = if let Some(ref m2) = matrix2 {
            Some(DMatrix::from_vec(cols, rows, m2.clone()))
        } else {
            None
        };
        let nalgebra_vector = if matches!(op, Operation::MatrixVectorMultiplication) {
            Some(DVector::from_vec(vector.clone()))
        } else {
            None
        };

        // Set throughput
        group.throughput(match op {
            Operation::MatrixVectorMultiplication => Throughput::Elements(size as u64),
            Operation::MatrixMatrixMultiplication => Throughput::Elements((size * size) as u64),
            _ => Throughput::Elements(0),
        });

        match op {
            Operation::MatrixVectorMultiplication => {
                group.bench_with_input(
                    BenchmarkId::new("rustynum_rs", size),
                    &size,
                    |bencher, &_| {
                        bencher.iter(|| {
                            matrix_multiply(&num_array_matrix1, &num_array_vector.as_ref().unwrap())
                        })
                    },
                );

                group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bencher, &_| {
                    bencher.iter(|| ndarray_matrix1.dot(ndarray_vector.as_ref().unwrap()))
                });

                group.bench_with_input(BenchmarkId::new("nalgebra", size), &size, |bencher, &_| {
                    bencher.iter(|| {
                        nalgebra_matrix1.clone() * nalgebra_vector.as_ref().unwrap().clone()
                    })
                });
            }
            Operation::MatrixMatrixMultiplication => {
                group.bench_with_input(
                    BenchmarkId::new("rustynum_rs", size),
                    &size,
                    |bencher, &_| {
                        bencher.iter(|| {
                            matrix_multiply(
                                &num_array_matrix1,
                                &num_array_matrix2.as_ref().unwrap(),
                            )
                        })
                    },
                );

                group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bencher, &_| {
                    bencher.iter(|| ndarray_matrix1.dot(ndarray_matrix2.as_ref().unwrap()))
                });

                group.bench_with_input(BenchmarkId::new("nalgebra", size), &size, |bencher, &_| {
                    bencher.iter(|| {
                        nalgebra_matrix1.clone() * nalgebra_matrix2.as_ref().unwrap().clone()
                    })
                });
            }
            _ => {}
        }
    }

    group.finish();
}

fn main_benchmark(c: &mut Criterion) {
    let vector_sizes = [1_000, 10_000, 100_000];
    let matrix_sizes = [100, 500, 1_000];

    // Vector operations
    benchmark_vector_operation(c, Operation::AddVectors, &vector_sizes);
    benchmark_vector_operation(c, Operation::Mean, &vector_sizes);
    benchmark_vector_operation(c, Operation::DotProduct, &vector_sizes);

    // Matrix operations
    benchmark_matrix_operation(c, Operation::MatrixVectorMultiplication, &matrix_sizes);
    benchmark_matrix_operation(c, Operation::MatrixMatrixMultiplication, &matrix_sizes);
}

criterion_group!(benches, main_benchmark);
criterion_main!(benches);
