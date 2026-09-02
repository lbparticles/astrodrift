use std::array;

use shared::{Config, Method, Model, Real};

use crate::integrators::dop853_cpu::{dop853_mw2014_batch, CpuAnnulusCtx, MwCpuContext};
use crate::state::{InputFrame, OutputFrame, OutputState};

use crate::dispatch::gpu::GPUDispatchError;

/// CPU batch dispatch for DOP853: integrates every particle of every model
/// component in MW2014 (bulge LUT from the background container's
/// supertable), rayon-parallel over particles. This is the CPU analogue of
/// the GPU's chunked launch: one call, N particles.
///
/// Requires the Bovy/MW2014 background (`bg_feature` with the bulge LUT);
/// the annulus perturber stack is GPU-only.
pub fn cpu_dispatch(
    config: Config,
    model: Model,
    input_frame: InputFrame,
    pot: Option<&super::PotSpec>,
) -> Result<OutputFrame, GPUDispatchError> {
    if config.method != Method::DOP853 {
        return Err(GPUDispatchError::Message(format!(
            "cpu_dispatch implements Method::DOP853 only (got {:#?})",
            config.method
        )));
    }
    let spec = pot.ok_or_else(|| {
        GPUDispatchError::Message(
            "CPU DOP853 requires a Bovy/MW2014 background (bg_feature with the bulge LUT)"
                .to_string(),
        )
    })?;
    let annulus = spec.annulus.as_ref().map(|a| CpuAnnulusCtx {
        coeffs: a.coeffs.clone(),
        n_gmc: a.n_gmc,
        division: a.division,
        final_time: a.final_time,
        amp: a.plummer_amp,
        b: a.plummer_b,
    });
    let ctx = MwCpuContext {
        lut: spec.supertable.clone(),
        r_min: spec.fparams[0],
        dr: spec.fparams[1],
        annulus,
    };

    // Output times: same linspace semantics as the GPU path (endpoint
    // included; dividing by `steps` would compress the grid by (steps-1)/steps
    // and desynchronise drift's time axis from galpy's).
    let ts = &config.settings.ts;
    let denom = if ts.steps > 1 { (ts.steps - 1) as Real } else { 1.0 };
    let times: Vec<Real> = (0..ts.steps)
        .map(|i| ts.start + (ts.end - ts.start) * i as Real / denom)
        .collect();

    let rtol = config.settings.tolerance.rtol;
    let atol = config.settings.tolerance.atol;

    let mut output_frame = OutputFrame(core::array::from_fn(|_| None));
    for (stage, (model_component_opt, input_state_opt)) in
        model.into_iter().zip(input_frame.into_iter()).enumerate()
    {
        if let (Some(_model_component), Some(input_state)) = (model_component_opt, input_state_opt)
        {
            let n = input_state.num_particles;
            if n == 0 || times.is_empty() {
                continue;
            }
            let states = &input_state.data[..n * 6];
            let data = dop853_mw2014_batch(states, &times, rtol, atol, &ctx);
            output_frame.0[stage] = Some(OutputState { data });
        }
    }
    Ok(output_frame)
}
