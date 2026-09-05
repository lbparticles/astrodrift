mod dop853;
mod dopr54;

#[cfg(all(feature = "rust-cuda", feature = "cuda-oxide"))]
compile_error!("features `rust-cuda` and `cuda-oxide` are mutually exclusive");

#[cfg(not(any(feature = "rust-cuda", feature = "cuda-oxide")))]
compile_error!("enable exactly one CUDA backend feature: `rust-cuda` or `cuda-oxide`");

const STATE_DIM: usize = 6;

/// Writes one particle state into the `(time, particle, component)` output.
///
/// # Safety
///
/// `state_out` must point to at least `((step + 1) * n) * STATE_DIM` writable
/// elements, and `tid` must be less than `n`. Concurrent callers must use
/// distinct `(step, tid)` pairs.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub(crate) unsafe fn write_time_major_state(
    state_out: *mut f64,
    step: usize,
    n: usize,
    tid: usize,
    state: &[f64; STATE_DIM],
) {
    // FIXME: replace this raw strided write with a proof-carrying cuda-device view
    // once one can represent a runtime-sized time-major trajectory without padding.
    let out_base = ((step * n) + tid) * STATE_DIM;
    for component in 0..STATE_DIM {
        unsafe { *state_out.add(out_base + component) = state[component] };
    }
}

#[cfg(feature = "rust-cuda")]
#[inline(always)]
fn rust_cuda_thread_id(n: usize) -> Option<usize> {
    use cuda_std::thread;

    let tid = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;
    (tid < n).then_some(tid)
}

#[cfg(feature = "rust-cuda")]
#[cuda_std::kernel]
#[allow(clippy::too_many_arguments)]
/// Runs DOPR54 for one particle per CUDA thread.
///
/// # Safety
///
/// `state0`, `times`, and `state_out` must be valid, aligned device pointers to
/// at least `n * STATE_DIM`, `nt`, and `nt * n * STATE_DIM` elements. The output
/// must not overlap either input, those size products must not overflow, and no
/// concurrent launch may write the same output. `n` must be positive and
/// `2 <= nt <= 1024`.
pub unsafe fn dopr54_cpu_port(
    state0: *const f64,
    times: *const f64,
    state_out: *mut f64,
    n: usize,
    nt: usize,
    rtol: f64,
    atol: f64,
    dt_one_init: f64,
) {
    let Some(tid) = rust_cuda_thread_id(n) else {
        return;
    };

    // SAFETY: the Rust-CUDA launch boundary requires these raw buffers to
    // contain the extents described by n and nt.
    let state0 = unsafe { core::slice::from_raw_parts(state0, n * STATE_DIM) };
    let times = unsafe { core::slice::from_raw_parts(times, nt) };
    unsafe {
        dopr54::integrate_particle(
            tid,
            n,
            nt,
            state0,
            times,
            state_out,
            rtol,
            atol,
            dt_one_init,
        )
    };
}

#[cfg(feature = "rust-cuda")]
#[cuda_std::kernel]
#[allow(clippy::too_many_arguments)]
/// Runs DOP853 for one particle per CUDA thread.
///
/// # Safety
///
/// `state0`, `times`, and `state_out` must be valid, aligned device pointers to
/// at least `n * STATE_DIM`, `nt`, and `nt * n * STATE_DIM` elements. The output
/// must not overlap either input, those size products must not overflow, and no
/// concurrent launch may write the same output. `n` must be positive and
/// `2 <= nt <= 1024`.
pub unsafe fn dop853_cpu_port(
    state0: *const f64,
    times: *const f64,
    state_out: *mut f64,
    n: usize,
    nt: usize,
    rtol: f64,
    atol: f64,
    _dt_one_init: f64,
) {
    let Some(tid) = rust_cuda_thread_id(n) else {
        return;
    };

    // SAFETY: the Rust-CUDA launch boundary requires these raw buffers to
    // contain the extents described by n and nt.
    let state0 = unsafe { core::slice::from_raw_parts(state0, n * STATE_DIM) };
    let times = unsafe { core::slice::from_raw_parts(times, nt) };
    unsafe { dop853::integrate_particle(tid, n, nt, state0, times, state_out, rtol, atol) };
}

#[cfg(feature = "cuda-oxide")]
#[cuda_host::cuda_module]
pub mod oxide {
    use cuda_device::{DisjointSlice, Uniform, kernel, launch_contract, thread};

    #[kernel]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n >= 1,
            nt >= 2,
            nt <= 1024,
            state0.len() >= n * 6,
            times.len() >= nt,
            state_out.len() >= nt * n * 6
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn dopr54_cpu_port(
        state0: &[f64],
        times: &[f64],
        mut state_out: DisjointSlice<f64>,
        n: Uniform<usize>,
        nt: Uniform<usize>,
        rtol: f64,
        atol: f64,
        dt_one_init: f64,
    ) {
        let tid = thread::index_1d().get();
        let n = n.get();
        if tid >= n {
            return;
        }

        // SAFETY: the launch contract proves all buffer extents; each thread has a unique tid.
        unsafe {
            super::dopr54::integrate_particle(
                tid,
                n,
                nt.get(),
                state0,
                times,
                state_out.as_mut_ptr(),
                rtol,
                atol,
                dt_one_init,
            )
        }
    }

    #[kernel]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n >= 1,
            nt >= 2,
            nt <= 1024,
            state0.len() >= n * 6,
            times.len() >= nt,
            state_out.len() >= nt * n * 6
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn dop853_cpu_port(
        state0: &[f64],
        times: &[f64],
        mut state_out: DisjointSlice<f64>,
        n: Uniform<usize>,
        nt: Uniform<usize>,
        rtol: f64,
        atol: f64,
        _dt_one_init: f64,
    ) {
        let tid = thread::index_1d().get();
        let n = n.get();
        if tid >= n {
            return;
        }

        // SAFETY: the launch contract proves all buffer extents; each thread has a unique tid.
        unsafe {
            super::dop853::integrate_particle(
                tid,
                n,
                nt.get(),
                state0,
                times,
                state_out.as_mut_ptr(),
                rtol,
                atol,
            )
        }
    }
}
