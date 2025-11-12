/// Binary search to find the largest step such that `time_out[s * n + p] <= t_end`.
///
/// # Complexity
/// - Per call: **O(log filled_len)**
/// - For all particles and all target times in `ts`: **O(n_particles × |ts| × log filled_len)**
#[inline]
pub fn find_preceding_step(
    time_out: &[f64],
    n_particles: usize,
    p: usize,
    filled_len: usize, // NOTE: per-particle valid length (>=1, <= steps_cap)
    t_end: f64,
) -> (f64, isize) {
    let mut lo = 0usize;
    let mut hi = filled_len; // exclusive
    while lo < hi {
        let mid = (lo + hi) / 2;
        let val = time_out[mid * n_particles + p];
        if val <= t_end {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let step = lo.saturating_sub(1);
    let idx = (step * n_particles + p) as isize;
    (time_out[idx as usize], step as isize)
}

pub fn find_last_times_and_indices(
    time_out: &[f64],
    ts: &[f64],
    n_particles: usize,
    steps_cap: usize,
    filled_lens: &[usize],
) -> (Vec<Vec<f64>>, Vec<Vec<isize>>) {
    assert_eq!(time_out.len(), n_particles * steps_cap);
    assert_eq!(filled_lens.len(), n_particles);

    let mut all_times = Vec::with_capacity(n_particles);
    let mut all_indices = Vec::with_capacity(n_particles);

    for p in 0..n_particles {
        let filled_len = filled_lens[p].min(steps_cap).max(1);
        let first = time_out[p];
        let last  = time_out[(filled_len - 1) * n_particles + p];

        let mut times_row = Vec::with_capacity(ts.len());
        let mut idx_row = Vec::with_capacity(ts.len());

        for &t_end in ts {
            // debug_assert!(t_end >= first && t_end <= last, "desired time {t_end} out of range [{first}, {last}] for particle {p}");
            // or for now:
            let t = if t_end < first {
                eprintln!("desired time {t_end} before first ({first}) for particle {p}; clamping");
                first
            } else if t_end > last {
                eprintln!("desired time {t_end} beyond last ({last}) for particle {p}; clamping");
                last
            } else {
                t_end
            };

            let (val, idx) = find_preceding_step(time_out, n_particles, p, filled_len, t);
            times_row.push(val);
            idx_row.push(idx);
        }

        all_times.push(times_row);
        all_indices.push(idx_row);
    }

    (all_times, all_indices)
}
