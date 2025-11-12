use drift_rs::index_helpers::find_last_times_and_indices;
use drift_rs::index_helpers::find_preceding_step;

/// Interleave per-particle timelines into the flattened `[steps_cap * n]` layout:
/// global_index = step * n + p
fn interleave(time_lines: &[&[f64]]) -> Vec<f64> {
    let n_particles = time_lines.len();
    assert!(n_particles > 0);
    let steps_cap = time_lines[0].len();
    for tl in time_lines {
        assert_eq!(tl.len(), steps_cap, "all timelines must have equal length");
    }
    let mut out = vec![0.0; steps_cap * n_particles];
    for step in 0..steps_cap {
        for p in 0..n_particles {
            out[step * n_particles + p] = time_lines[p][step];
        }
    }
    out
}

#[test]
fn single_particle_exact_and_between() {
    let n = 1usize;
    let steps_cap = 5usize;

    // Per-particle output times
    let time_out = vec![0.0, 0.10, 0.21, 0.33, 0.60];

    // Desired evenly spaced targets (all within [0, t_end])
    let desired_ts = vec![0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60];
    let filled_lens = vec![steps_cap; n];

    let (times, indices) = find_last_times_and_indices(&time_out, &desired_ts, n, steps_cap, &filled_lens);

    assert_eq!(times.len(), n);
    assert_eq!(indices.len(), n);
    assert_eq!(times[0].len(), desired_ts.len());
    assert_eq!(indices[0].len(), desired_ts.len());

    let expected_times  = vec![0.0, 0.0, 0.10, 0.10, 0.10, 0.21, 0.21, 0.33, 0.33, 0.33, 0.33, 0.33, 0.60];
    let expected_indices = vec![0isize, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4];

    assert_eq!(times[0], expected_times);
    assert_eq!(indices[0], expected_indices);
}

#[test]
fn two_particles() {
    let n = 2usize;
    let steps_cap = 4usize;

    let p0 = [0.0, 0.2, 0.5, 1.0];
    let p1 = [0.0, 0.1, 0.4, 0.9];
    let time_out = interleave(&[&p0, &p1]);

    let desired_ts = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let filled_lens = vec![steps_cap; n];

    let (times, indices) = find_last_times_and_indices(&time_out, &desired_ts, n, steps_cap, &filled_lens);

    let expected_p0_times = vec![0.0, 0.0, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0];
    let expected_p0_indices = vec![0, 0, 2, 2, 2, 4, 4, 4, 4, 4, 6];

    let expected_p1_times = vec![0.0, 0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.9, 0.9];
    let expected_p1_indices = vec![1, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7];

    assert_eq!(times.len(), n);
    assert_eq!(indices.len(), n);

    assert_eq!(times[0], expected_p0_times);
    assert_eq!(indices[0], expected_p0_indices);

    assert_eq!(times[1], expected_p1_times);
    assert_eq!(indices[1], expected_p1_indices);
}

#[test]
fn empty_desired_ts_returns_empty_rows() {
    let n = 2usize;
    let steps_cap = 3usize;
    let p0 = [0.0, 0.2, 0.5];
    let p1 = [0.0, 0.1, 0.4];
    let time_out = interleave(&[&p0, &p1]);
    let desired_ts: Vec<f64> = vec![];
    let filled_lens = vec![steps_cap; n];

    let (times, indices) = find_last_times_and_indices(&time_out, &desired_ts, n, steps_cap, &filled_lens);

    assert_eq!(times.len(), n);
    assert_eq!(indices.len(), n);
    assert!(times.iter().all(|row| row.is_empty()));
    assert!(indices.iter().all(|row| row.is_empty()));
}

#[test]
fn duplicate_desired_times_produce_duplicate_outputs() {
    let n = 1usize;
    let steps_cap = 4usize;
    let time_out = vec![0.0, 0.25, 0.5, 1.0];
    let desired_ts = vec![0.5, 0.5, 0.5];
    let filled_lens = vec![steps_cap; n];

    let (times, indices) = find_last_times_and_indices(&time_out, &desired_ts, n, steps_cap, &filled_lens);

    assert_eq!(times[0], vec![0.5, 0.5, 0.5]);
    assert_eq!(indices[0], vec![2, 2, 2]);
}

#[test]
fn larger_n_indexing() {
    let n = 4usize;
    let steps_cap = 3usize;
    // p0..p3
    let p0 = [0.0, 0.1, 0.2];
    let p1 = [0.0, 0.2, 0.4];
    let p2 = [0.0, 0.3, 0.6];
    let p3 = [0.0, 0.4, 0.8];
    let time_out = interleave(&[&p0, &p1, &p2, &p3]);

    let desired_ts = (0..=8).map(|k| k as f64 * 0.1).collect::<Vec<_>>();
    let filled_lens = vec![steps_cap; n];

    let (times, indices) = find_last_times_and_indices(&time_out, &desired_ts, n, steps_cap, &filled_lens);

    // For p3 and ts=0.4, preceding time is 0.4 at step=1 -> idx = 1*4 + 3 = 7
    let pos_04 = desired_ts.iter().position(|&t| (t - 0.4).abs() < 1e-15).unwrap();
    assert_eq!(times[3][pos_04], 0.4);
    assert_eq!(indices[3][pos_04], 7);

    // For p2 and ts=0.5, preceding time is 0.3 at step=1 -> idx = 1*4 + 2 = 6
    let pos_05 = desired_ts.iter().position(|&t| (t - 0.5).abs() < 1e-15).unwrap();
    assert_eq!(times[2][pos_05], 0.3);
    assert_eq!(indices[2][pos_05], 6);
}

#[test]
fn clamp_before_first_and_after_last() {
    // p0 ends at 0.8, p1 ends at 0.9. desired grid includes values before first and beyond last
    let n = 2usize;
    let steps_cap = 4usize;
    let p0 = [0.1, 0.2, 0.5, 0.8];
    let p1 = [0.1, 0.3, 0.6, 0.9];
    let time_out = interleave(&[&p0, &p1]);

    let desired_ts = vec![0.05, 0.1, 0.35, 0.8, 0.85, 0.9, 0.95];
    let filled_lens = vec![steps_cap; n];

    let (times, indices) = find_last_times_and_indices(&time_out, &desired_ts, n, steps_cap, &filled_lens);

    // p0: clamp 0.05 -> 0.1, clamp 0.85/0.9/0.95 -> 0.8
    assert_eq!(times[0], vec![0.1, 0.1, 0.2, 0.8, 0.8, 0.8, 0.8]);
    assert_eq!(indices[0], vec![0, 0, 2, 6, 6, 6, 6]);

    // p1: clamp 0.05 -> 0.1, clamp 0.95 -> 0.9
    assert_eq!(times[1], vec![0.1, 0.1, 0.3, 0.6, 0.6, 0.9, 0.9]);
    assert_eq!(indices[1], vec![1, 1, 3, 5, 5, 7, 7]);
}

#[test]
#[should_panic]
fn panics_on_len_mismatch() {
    let n = 2usize; 
    let steps_cap = 3usize;
    let time_out = vec![0.0; 5]; // should be 6
    let filled_lens = vec![steps_cap; n];
    let _ = find_last_times_and_indices(&time_out, &[0.0], n, steps_cap, &filled_lens);
}

#[test]
fn bs_exact_match_first_middle_last() {
    let n = 1usize;
    let steps_cap = 5usize;
    let time_out = vec![0.0, 0.2, 0.5, 0.7, 1.0];

    // exact first
    let (v, idx) = find_preceding_step(&time_out, n, 0, steps_cap, 0.0);
    assert_eq!(v, 0.0);
    assert_eq!(idx, 0);

    // exact middle
    let (v, idx) = find_preceding_step(&time_out, n, 0, steps_cap, 0.5);
    assert_eq!(v, 0.5);
    assert_eq!(idx, 2);

    // exact last
    let (v, idx) = find_preceding_step(&time_out, n, 0, steps_cap, 1.0);
    assert_eq!(v, 1.0);
    assert_eq!(idx, 4);
}

#[test]
fn bs_between_values_picks_preceding() {
    let n = 1usize;
    let steps_cap = 4usize;
    let time_out = vec![0.0, 0.3, 0.6, 1.0];

    // 0.45 -> preceding 0.3 (idx=1)
    let (v, idx) = find_preceding_step(&time_out, n, 0, steps_cap, 0.45);
    assert_eq!(v, 0.3);
    assert_eq!(idx, 1);

    // 0.999 -> preceding 0.6 (idx=2)
    let (v, idx) = find_preceding_step(&time_out, n, 0, steps_cap, 0.999);
    assert_eq!(v, 0.6);
    assert_eq!(idx, 2);
}

#[test]
fn bs_plateau_returns_last_equal() {
    let n = 1usize;
    let steps_cap = 5usize;
    let time_out = vec![0.0, 0.2, 0.2, 0.6, 1.0];

    // exact 0.2 should return the *last* equal (step 2)
    let (v, idx) = find_preceding_step(&time_out, n, 0, steps_cap, 0.2);
    assert_eq!(v, 0.2);
    assert_eq!(idx, 2);

    // slightly above 0.2 still returns step 2
    let (v, idx) = find_preceding_step(&time_out, n, 0, steps_cap, 0.2000000000001);
    assert_eq!(v, 0.2);
    assert_eq!(idx, 2);
}

#[test]
fn respects_filled_len_ignores_trailing_zeros() {
    let n = 2usize;
    let steps_cap = 5usize;

    let p0 = [0.0, 0.2, 0.5, 0.0, 0.0];
    let p1 = [0.0, 0.1, 0.4, 0.9, 0.0];
    let time_out = interleave(&[&p0, &p1]);

    let filled_lens = vec![3usize, 4usize];

    // Query across and beyond the filled region; beyond should clamp to the last filled value.
    let desired_ts = vec![0.0, 0.05, 0.2, 0.3, 0.5, 0.7, 0.8, 0.95];

    let (times, indices) =
        find_last_times_and_indices(&time_out, &desired_ts, n, steps_cap, &filled_lens);

    let expected_p0_times   = vec![0.0, 0.0, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5];
    let expected_p0_indices = vec![0isize, 0,    2,   2,   4,   4,   4,   4];

    let expected_p1_times   = vec![0.0, 0.0, 0.1, 0.1, 0.4, 0.4, 0.4, 0.9];
    let expected_p1_indices = vec![1isize, 1,    3,   3,   5,   5,   5,   7];

    assert_eq!(times.len(), n);
    assert_eq!(indices.len(), n);

    assert_eq!(times[0], expected_p0_times);
    assert_eq!(indices[0], expected_p0_indices);

    assert_eq!(times[1], expected_p1_times);
    assert_eq!(indices[1], expected_p1_indices);
}