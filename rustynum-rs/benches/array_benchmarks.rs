use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustynum_rs::NumArray32; // Adjust this to the path to your library
use ndarray::Array1;
use nalgebra::DVector;

fn create_large_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..size).map(|x| (size as f32 - x as f32)).collect();
    (a, b)
}

fn add_vectors_benchmark(c: &mut Criterion) {
    let size = 10_000; // Example size, adjust based on needs
    let (a_vec, b_vec) = create_large_vectors(size);

    // Setup for rustynum_rs
    let a_rustynum = NumArray32::new(a_vec.clone());
    let b_rustynum = NumArray32::new(b_vec.clone());

    // Setup for ndarray
    let a_ndarray = Array1::from_vec(a_vec.clone());
    let b_ndarray = Array1::from_vec(b_vec.clone());

    // Setup for nalgebra
    let a_nalgebra = DVector::from_vec(a_vec.clone());
    let b_nalgebra = DVector::from_vec(b_vec);

    // Benchmark for rustynum_rs
    c.bench_function("rustynum_rs add large vectors", |bencher| {
        bencher.iter(|| {
            let result = black_box(&a_rustynum) + black_box(&b_rustynum);
            result
            // Perform any necessary assertions or result checks here
        });
    });

    // Benchmark for ndarray
    c.bench_function("ndarray add large vectors", |bencher| {
        bencher.iter(|| {
            let result = black_box(&a_ndarray) + black_box(&b_ndarray);
            result
            // Perform any necessary assertions or result checks here
        });
    });

    // Benchmark for nalgebra
    c.bench_function("nalgebra add large vectors", |bencher| {
        bencher.iter(|| {
            let result = black_box(&a_nalgebra) + black_box(&b_nalgebra);
            result
            // Perform any necessary assertions or result checks here
        });
    });


    // Benchmark for Mean
    c.bench_function("rustynum_rs mean", |bencher| {
      bencher.iter(|| {
          let mean = a_rustynum.mean();
          mean
          // Assertions for mean
      });
  });

  c.bench_function("ndarray mean", |bencher| {
      bencher.iter(|| {
          let mean = a_ndarray.mean().unwrap();
          mean
          // Assertions for mean
      });
  });

  // nalgebra does not have a direct mean method, so we calculate it manually
  c.bench_function("nalgebra mean", |bencher| {
      bencher.iter(|| {
          let mean = a_nalgebra.iter().sum::<f32>() / a_nalgebra.len() as f32;
          mean
          // Assertions for mean
      });
  });


  // Benchmark for Dot Product
  c.bench_function("rustynum_rs dot", |bencher| {
    bencher.iter(|| {
        let dot = a_rustynum.dot(&b_rustynum);
        dot
        // Assertions for dot product
    });
  });

  c.bench_function("ndarray dot", |bencher| {
    bencher.iter(|| {
        let dot = a_ndarray.dot(&b_ndarray);
        dot
        // Assertions for dot product
    });
  });

  c.bench_function("nalgebra dot", |bencher| {
    bencher.iter(|| {
        let dot = a_nalgebra.dot(&b_nalgebra);
        dot
        // Assertions for dot product
    });
  });

    // Repeat the setup for mean and dot product benchmarks
}

criterion_group!(benches, add_vectors_benchmark);
criterion_main!(benches);
