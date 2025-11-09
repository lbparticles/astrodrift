use libm::{pow, sqrt};

pub struct State6 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

#[inline(always)]
pub fn rk_norm(
    prev_state: State6,
    curr_state: State6,
    erro_state: State6,
    atol: f64,
    rtol: f64,
) -> f64 {
    let sc_x = atol + rtol * f64::max(prev_state.x.abs(), curr_state.x.abs());
    let sc_y = atol + rtol * f64::max(prev_state.y.abs(), curr_state.y.abs());
    let sc_z = atol + rtol * f64::max(prev_state.z.abs(), curr_state.z.abs());
    let sc_vx = atol + rtol * f64::max(prev_state.vx.abs(), curr_state.vx.abs());
    let sc_vy = atol + rtol * f64::max(prev_state.vy.abs(), curr_state.vy.abs());
    let sc_vz = atol + rtol * f64::max(prev_state.vz.abs(), curr_state.vz.abs());

    // might need to guard against div by 0
    let sum = pow(erro_state.x / sc_x, 2.0)
        + pow(erro_state.y / sc_y, 2.0)
        + pow(erro_state.z / sc_z, 2.0)
        + pow(erro_state.vx / sc_vx, 2.0)
        + pow(erro_state.vy / sc_vy, 2.0)
        + pow(erro_state.vz / sc_vz, 2.0);

    sqrt(sum / 6.0)
}

