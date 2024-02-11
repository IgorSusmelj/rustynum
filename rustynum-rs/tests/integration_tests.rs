use rustynum_rs::NumArray32;

#[test]
fn test_num_array_creation_and_dot_product() {
    // Create two `NumArray` instances with test data.
    let array1 = NumArray32::from(&[1.0, 2.0, 3.0, 4.0][..]);
    let array2 = NumArray32::from(vec![4.0, 3.0, 2.0, 1.0]);

    // Perform a dot product operation between the two arrays.
    let result = array1.dot(&array2);

    // Check that the dot product result is as expected.
    assert_eq!(
        result, 20.0,
        "The dot product of the two arrays should be 20.0"
    );

    // Test with arrays of different sizes to ensure it handles non-multiples of SIMD width
    let array3 = NumArray32::from(&[1.0, 2.0, 3.0][..]);
    let array4 = NumArray32::from(vec![4.0, 5.0, 6.0]);

    // Perform a dot product operation between the two smaller arrays.
    let result_small = array3.dot(&array4);

    // Check that the dot product result is as expected for the smaller arrays.
    assert_eq!(
        result_small, 32.0,
        "The dot product of the two smaller arrays should be 32.0"
    );
}
