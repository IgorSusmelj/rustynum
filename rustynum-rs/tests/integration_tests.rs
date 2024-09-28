use rustynum_rs::{NumArrayF32, NumArrayF64};

#[test]
fn test_num_array_creation_and_dot_product() {
    // Create two `NumArray` instances with test data.
    let array1 = NumArrayF32::from(&[1.0, 2.0, 3.0, 4.0][..]);
    let array2 = NumArrayF32::from(vec![4.0, 3.0, 2.0, 1.0]);

    // Perform a dot product operation between the two arrays.
    let result = array1.dot(&array2);

    // Check that the dot product result is as expected.
    assert_eq!(
        result.get_data(),
        &[20.0],
        "The dot product of the two arrays should be 20.0"
    );

    // Test with arrays of different sizes to ensure it handles non-multiples of SIMD width
    let array3 = NumArrayF32::from(&[1.0, 2.0, 3.0][..]);
    let array4 = NumArrayF32::from(vec![4.0, 5.0, 6.0]);

    // Perform a dot product operation between the two smaller arrays.
    let result_small = array3.dot(&array4);

    // Check that the dot product result is as expected for the smaller arrays.
    assert_eq!(
        result_small.get_data(),
        &[32.0],
        "The dot product of the two smaller arrays should be 32.0"
    );
}

#[test]
fn test_complex_operations() {
    let size = 1000; // Choose a size for the vectors
    let constant = 2.5f64;

    // Generate two large NumArray instances
    let data1: Vec<f64> = (0..size).map(|x| x as f64).collect();

    let step = size as f64 / (size - 1) as f64; // Correct step calculation
    let data2: Vec<f64> = (0..size).map(|x| size as f64 - x as f64 * step).collect();

    let array1 = NumArrayF64::from(data1.clone());
    let array2 = NumArrayF64::from(data2.clone());

    // Perform addition of two NumArrays
    let added = &array1 + &array2;

    // Subtract the mean from the added array
    let mean = added.mean().item();
    let subtracted_mean = &added - mean;

    // Divide by a constant value
    let divided = &subtracted_mean / constant;

    // Multiply with another NumArray (using the original array1 for simplicity)
    let multiplied = &divided * &array1;

    // Perform a dot product with the initial input (array2)
    let dot_product_result = multiplied.dot(&array2).item();
    // Print arrray2

    // Expected result calculation using ndarray or manual calculation
    // Placeholder for expected result
    let expected_result = 0.000012200523883620917; // Calculate the expected result

    let tolerance = 1e-1; // Define a suitable tolerance for your scenario
    let actual_error = (dot_product_result - expected_result).abs();
    assert!(
        actual_error <= tolerance,
        "The complex operation result does not match the expected value. \
        Expected: {}, Actual: {}, Error: {}",
        expected_result,
        dot_product_result,
        actual_error
    );
}
