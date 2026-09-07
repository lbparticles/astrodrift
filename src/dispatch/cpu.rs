use std::time::Instant;

use log::{debug, info};
use shared::{Config, INPUT_STATE_DIM, Method, Model, Real, Tolerance};

use crate::{
    dispatch::{DispatchError, dispatch_stages, sample_times},
    integrators::{dop853_cpu, dopr54_cpu},
    state::{InputFrame, InputState, OutputFrame, OutputState},
};

const INITIAL_STEP_SENTINEL: Real = -9999.99;
const LOG_TARGET: &str = "drift::dispatch::cpu";

// Progress is reported at most this many times per stage, and not at all for
// stages smaller than this, so log volume stays proportional to the work.
const PROGRESS_UPDATES_PER_STAGE: usize = 10;

fn integrate_particle(
    method: Method,
    initial_state: &[Real; INPUT_STATE_DIM],
    times: &[Real],
    tolerance: Tolerance,
) -> Result<Vec<Real>, DispatchError> {
    match method {
        Method::DOPR54 => dopr54_cpu::integrate_kepler(
            *initial_state,
            times,
            INITIAL_STEP_SENTINEL,
            0,
            tolerance.rtol,
            tolerance.atol,
        )
        .map_err(|code| DispatchError::CpuIntegration { method, code }),
        Method::DOP853 => Ok(dop853_cpu::integrate_kepler(
            *initial_state,
            times,
            INITIAL_STEP_SENTINEL,
            0,
            tolerance.rtol,
            tolerance.atol,
        )),
    }
}

fn integrate_stage(
    stage: usize,
    method: Method,
    input_state: &InputState,
    times: &[Real],
    tolerance: Tolerance,
) -> Result<OutputState, DispatchError> {
    let particle_count = input_state.num_particles;
    let mut output = OutputState::new_zeroed(times.len(), particle_count);
    let input = &input_state.data[..particle_count * INPUT_STATE_DIM];
    let (initial_states, remainder) = input.as_chunks::<INPUT_STATE_DIM>();
    debug_assert!(remainder.is_empty());

    let progress_interval = particle_count.div_ceil(PROGRESS_UPDATES_PER_STAGE);
    for (particle, initial_state) in initial_states.iter().enumerate() {
        let trajectory = integrate_particle(method, initial_state, times, tolerance)?;
        let (states, remainder) = trajectory.as_chunks::<INPUT_STATE_DIM>();
        debug_assert!(remainder.is_empty());
        for (time, state) in states.iter().enumerate() {
            let offset = (time * particle_count + particle) * INPUT_STATE_DIM;
            output.data[offset..offset + INPUT_STATE_DIM].copy_from_slice(state);
        }

        let completed = particle + 1;
        if particle_count >= PROGRESS_UPDATES_PER_STAGE
            && completed % progress_interval == 0
            && completed != particle_count
        {
            debug!(
                target: LOG_TARGET,
                "stage {stage}: integrated {completed}/{particle_count} particles"
            );
        }
    }

    Ok(output)
}

pub fn cpu_dispatch(
    config: Config,
    model: Model,
    input_frame: InputFrame,
) -> Result<OutputFrame, DispatchError> {
    let times = sample_times(config.settings.ts);
    info!(
        target: LOG_TARGET,
        "CPU integration starting: method={:?}, {} output times over [{}, {}], rtol={:.2e}, atol={:.2e}",
        config.method,
        times.len(),
        times.first().copied().unwrap_or_default(),
        times.last().copied().unwrap_or_default(),
        config.settings.tolerance.rtol.exp(),
        config.settings.tolerance.atol.exp(),
    );

    let started = Instant::now();
    let mut stage = 0usize;
    let mut total_particles = 0usize;
    let output_frame = dispatch_stages(model, input_frame, |_model_component, input_state| {
        stage += 1;
        let particle_count = input_state.num_particles;
        total_particles += particle_count;
        debug!(
            target: LOG_TARGET,
            "stage {stage}: integrating {particle_count} particles"
        );

        // FIXME: The reference CPU and GPU paths currently use their
        // kernel-local Kepler force instead of the supplied model.
        let stage_started = Instant::now();
        let result = integrate_stage(
            stage,
            config.method,
            input_state,
            &times,
            config.settings.tolerance,
        );
        if result.is_ok() {
            debug!(
                target: LOG_TARGET,
                "stage {stage}: {particle_count} particles finished in {:.3?}",
                stage_started.elapsed()
            );
        }
        result
    })?;

    info!(
        target: LOG_TARGET,
        "CPU integration finished: {stage} stage(s), {total_particles} particle trajectories in {:.3?}",
        started.elapsed()
    );
    Ok(output_frame)
}
