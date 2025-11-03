use libm::expf;
use libm::powf;
use libm::sqrtf;

const PI: f32 = 3.14159265358979323846264338327950288;

const AMP_S: f32 = 67168.3897826272;
const AMP_MN: f32 = 0.7574802019;
const AMP_NFW: f32 = 7.03716754404;

#[inline(always)]
fn sphericalcutoff_force(
    x: f32,
    y: f32,
    z: f32,
    amp: f32,
    alpha: f32,
    r1: f32,
    c2: f32,
) -> (f32, f32, f32) {
    let r2 = powf(x, 2.) + powf(y, 2.) + powf(z, 2.);
    let r = sqrtf(r2);
    let ar = -amp * (powf(r1 / r, alpha) * (alpha * c2 + 2. * r2) * expf(-r2 / c2)) / (r * c2);
    let ax = ar * (x / r);
    let ay = ar * (y / r);
    let az = ar * (z / r);
    (ax, ay, az)
}
fn navarro_frenk_white_force(x: f32, y: f32, z: f32, amp: f32, a: f32) -> (f32, f32, f32) {
    let r2 = powf(x, 2.) + powf(y, 2.) + powf(z, 2.);
    let r = sqrtf(r2);
    let ar3 = powf((a + r), 3);
    let ar = -amp * (1. / (4. * PI)) * ((a + 3. * r) / (r2 * ar3));
    let ax = ar * (x / r);
    let ay = ar * (y / r);
    let az = ar * (z / r);
    (ax, ay, az)
}
fn miyamoto_nagai_force(x: f32, y: f32, z: f32, amp: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let R2 = powf(x, 2.) + powf(y, 2.);
    let R = sqrtf(R2);
    let z2 = powf(z, 2.);
    let b2 = powf(b, 2.);
    let sqrtz2b2 = sqrtf(z2 + b2);
    let pyth = powf(a + sqrtz2b2, 2.);
    let denom = powf(pyth + R2, 3. / 2.);
    let aR = -amp * (R / denom);
    let ax = aR * (x / R);
    let ay = aR * (y / R);
    let az = -z * (a + sqrtz2b2) / (sqrtz2b2 * denom);
    (ax, ay, az)
}
