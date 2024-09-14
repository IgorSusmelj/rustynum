/// Executes a closure in parallel over chunks of a range `[start, end)`.
///
/// # Arguments
/// * `start` - The starting index of the range.
/// * `end` - The ending index of the range (exclusive).
/// * `f` - The closure to execute, which takes `chunk_start` and `chunk_end` as arguments.
pub fn parallel_for_chunks<F>(start: usize, end: usize, f: F)
where
    F: Fn(usize, usize) + Sync + Send + Copy,
{
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4); // Default to 4 threads if unavailable
    let chunk_size = (end - start + num_threads - 1) / num_threads;

    std::thread::scope(|s| {
        for chunk_start in (start..end).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(end);
            s.spawn(move || {
                f(chunk_start, chunk_end);
            });
        }
    });
}
