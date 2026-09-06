use shared::{Config, INPUT_STATE_DIM, Method, Model, Real, Tolerance};

use crate::{
    dispatch::{DispatchError, dispatch_stages, sample_times},
    integrators::{dop853_cpu, dopr54_cpu},
    state::{InputFrame, InputState, OutputFrame, OutputState},
};

const INITIAL_STEP_SENTINEL: Real = -9999.99;

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

    for (particle, initial_state) in initial_states.iter().enumerate() {
        let trajectory = integrate_particle(method, initial_state, times, tolerance)?;
        let (states, remainder) = trajectory.as_chunks::<INPUT_STATE_DIM>();
        debug_assert!(remainder.is_empty());
        for (time, state) in states.iter().enumerate() {
            let offset = (time * particle_count + particle) * INPUT_STATE_DIM;
            output.data[offset..offset + INPUT_STATE_DIM].copy_from_slice(state);
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
    dispatch_stages(model, input_frame, |_model_component, input_state| {
        // FIXME: The reference CPU and GPU paths currently use their
        // kernel-local Kepler force instead of the supplied model.
        integrate_stage(
            config.method,
            input_state,
            &times,
            config.settings.tolerance,
        )
    })
}
