use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion,
};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use rustynum_rs::num_array::linalg::matrix_multiply;
use rustynum_rs::NumArray32; // Adjust this to the path to your library

fn create_matrix(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|x| x as f32).collect()
}

fn create_vector(size: usize) -> Vec<f32> {
    (0..size).map(|x| x as f32).collect()
}

fn add_vectors_benchmark(c: &mut Criterion) {
    let size = 10_000; // Example size
    let mut group = c.benchmark_group("Vector Addition");

    // Setup for rustynum_rs
    let a_rustynum = NumArray32::new(create_vector(size));
    let b_rustynum = NumArray32::new(create_vector(size));

    // Setup for ndarray
    let a_ndarray = Array1::from_vec(create_vector(size));
    let b_ndarray = Array1::from_vec(create_vector(size));

    // Setup for nalgebra
    let a_nalgebra = DVector::from_vec(create_vector(size));
    let b_nalgebra = DVector::from_vec(create_vector(size));

    // Benchmark rustynum_rs
    group.bench_with_input(
        BenchmarkId::new("rustynum_rs", size),
        &size,
        |bencher, _| {
            bencher.iter(|| black_box(&a_rustynum) + black_box(&b_rustynum));
        },
    );

    // Benchmark ndarray
    group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bencher, _| {
        bencher.iter(|| black_box(&a_ndarray) + black_box(&b_ndarray));
    });

    // Benchmark nalgebra
    group.bench_with_input(BenchmarkId::new("nalgebra", size), &size, |bencher, _| {
        bencher.iter(|| black_box(&a_nalgebra) + black_box(&b_nalgebra));
    });

    group.finish(); // Important to call finish when done with a group
}

fn mean_benchmark(c: &mut Criterion) {
    let size = 10_000; // Example size
    let mut group = c.benchmark_group("Vector Mean");

    // Setup for rustynum_rs
    let a_rustynum = NumArray32::new(create_vector(size));

    // Setup for ndarray
    let a_ndarray = Array1::from_vec(create_vector(size));

    // Setup for nalgebra
    let a_nalgebra = DVector::from_vec(create_vector(size));

    // Benchmark rustynum_rs
    group.bench_with_input(
        BenchmarkId::new("rustynum_rs", size),
        &size,
        |bencher, _| {
            bencher.iter(|| {
                let mean = a_rustynum.mean();
                mean
                // Assertions for mean
            });
        },
    );

    // Benchmark ndarray
    group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bencher, _| {
        bencher.iter(|| {
            let mean = a_ndarray.mean().unwrap();
            mean
            // Assertions for mean
        });
    });

    // Benchmark nalgebra
    group.bench_with_input(BenchmarkId::new("nalgebra", size), &size, |bencher, _| {
        bencher.iter(|| {
            let mean = a_nalgebra.iter().sum::<f32>() / a_nalgebra.len() as f32;
            mean
            // Assertions for mean
        });
    });

    group.finish(); // Important to call finish when done with a group
}

fn dot_product_benchmark(c: &mut Criterion) {
    let size = 10_000; // Example size
    let mut group = c.benchmark_group("Vector Addition");

    // Setup for rustynum_rs
    let a_rustynum = NumArray32::new(create_vector(size));
    let b_rustynum = NumArray32::new(create_vector(size));

    // Setup for ndarray
    let a_ndarray = Array1::from_vec(create_vector(size));
    let b_ndarray = Array1::from_vec(create_vector(size));

    // Setup for nalgebra
    let a_nalgebra = DVector::from_vec(create_vector(size));
    let b_nalgebra = DVector::from_vec(create_vector(size));

    // Benchmark rustynum_rs
    group.bench_with_input(
        BenchmarkId::new("rustynum_rs", size),
        &size,
        |bencher, _| {
            bencher.iter(|| {
                let dot = a_rustynum.dot(&b_rustynum);
                dot
                // Assertions for dot product
            });
        },
    );

    // Benchmark ndarray
    group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bencher, _| {
        bencher.iter(|| {
            let dot = a_ndarray.dot(&b_ndarray);
            dot
            // Assertions for dot product
        });
    });

    // Benchmark nalgebra
    group.bench_with_input(BenchmarkId::new("nalgebra", size), &size, |bencher, _| {
        bencher.iter(|| {
            let dot = a_nalgebra.dot(&b_nalgebra);
            dot
            // Assertions for dot product
        });
    });

    group.finish(); // Important to call finish when done with a group
}

fn matrix_vector_multiplication_benchmark(c: &mut Criterion) {
    let rows = 1000;
    let cols = 1000;
    let matrix = create_matrix(rows, cols);
    let vector = create_vector(cols);

    let mut group = c.benchmark_group("Matrix-Vector Multiplication");

    let num_array_matrix = NumArray32::new_with_shape(matrix.clone(), vec![rows, cols]);
    let num_array_vector = NumArray32::new_with_shape(vector.clone(), vec![cols]);
    let ndarray_matrix = Array2::from_shape_vec((rows, cols), matrix.clone()).unwrap();
    let ndarray_vector = Array1::from_vec(vector);
    let nalgebra_matrix = DMatrix::from_element(rows, cols, 0.0);
    let nalgebra_vector = DVector::from_element(cols, 0.0);

    group.bench_with_input(
        BenchmarkId::new("rustynum_rs", rows),
        &rows,
        |bencher, _| {
            bencher.iter(|| matrix_multiply(&num_array_matrix, &num_array_vector));
        },
    );

    group.bench_with_input(BenchmarkId::new("ndarray", rows), &rows, |bencher, _| {
        bencher.iter(|| ndarray_matrix.dot(&ndarray_vector));
    });

    group.bench_with_input(BenchmarkId::new("nalgebra", rows), &rows, |bencher, _| {
        bencher.iter(|| nalgebra_matrix.clone() * nalgebra_vector.clone());
    });

    group.finish();
}

fn matrix_matrix_multiplication_benchmark(c: &mut Criterion) {
    let rows = 1000;
    let cols = 1000;
    let matrix1 = create_matrix(rows, cols);
    let matrix2 = create_matrix(cols, rows);

    let mut group = c.benchmark_group("Matrix-Matrix Multiplication");

    let num_array_matrix1 = NumArray32::new_with_shape(matrix1.clone(), vec![rows, cols]);
    let num_array_matrix2 = NumArray32::new_with_shape(matrix2.clone(), vec![cols, rows]);
    let ndarray_matrix1 = Array2::from_shape_vec((rows, cols), matrix1).unwrap();
    let ndarray_matrix2 = Array2::from_shape_vec((cols, rows), matrix2).unwrap();
    let nalgebra_matrix1 = DMatrix::from_element(rows, cols, 0.0);
    let nalgebra_matrix2 = DMatrix::from_element(cols, rows, 0.0);

    group.bench_with_input(
        BenchmarkId::new("rustynum_rs", rows),
        &rows,
        |bencher, _| {
            bencher.iter(|| matrix_multiply(&num_array_matrix1, &num_array_matrix2));
        },
    );

    group.bench_with_input(BenchmarkId::new("ndarray", rows), &rows, |bencher, _| {
        bencher.iter(|| ndarray_matrix1.dot(&ndarray_matrix2));
    });

    group.bench_with_input(BenchmarkId::new("nalgebra", rows), &rows, |bencher, _| {
        bencher.iter(|| nalgebra_matrix1.clone() * nalgebra_matrix2.clone());
    });

    group.finish();
}

criterion_group!(
    benches,
    add_vectors_benchmark,
    mean_benchmark,
    matrix_vector_multiplication_benchmark,
    matrix_matrix_multiplication_benchmark
);
criterion_main!(benches);
