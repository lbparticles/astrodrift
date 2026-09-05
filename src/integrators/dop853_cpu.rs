// Preserve the published DOP853 coefficients and galpy operation ordering in
// this CPU reference implementation, including expressions Clippy can shorten.
#![allow(clippy::assign_op_pattern, clippy::excessive_precision)]

use libc::{c_double, c_int};

#[repr(C)]
pub struct potentialArg {
    _private: [u8; 0],
}

pub type FuncPtr = Option<
    extern "C" fn(
        t: c_double,
        q: *mut c_double,
        a: *mut c_double,
        nargs: c_int,
        potential_args: *mut potentialArg,
    ),
>;

#[link(name = "m")]
unsafe extern "C" {
    fn exp(x: c_double) -> c_double;
    fn fabs(x: c_double) -> c_double;
    fn pow(x: c_double, y: c_double) -> c_double;
    fn sqrt(x: c_double) -> c_double;
}

const UROUND: c_double = 2.3e-16;

const C2: c_double = 0.526001519587677318785587544488e-1;
const C3: c_double = 0.789002279381515978178381316732e-1;
const C4: c_double = 0.118350341907227396726757197510;
const C5: c_double = 0.281649658092772603273242802490;
const C6: c_double = 0.333333333333333333333333333333;
const C7: c_double = 0.25;
const C8: c_double = 0.307692307692307692307692307692;
const C9: c_double = 0.651282051282051282051282051282;
const C10: c_double = 0.6;
const C11: c_double = 0.857142857142857142857142857142;
const C14: c_double = 0.1;
const C15: c_double = 0.2;
const C16: c_double = 0.777777777777777777777777777778;

const A21: c_double = 5.26001519587677318785587544488e-2;
const A31: c_double = 1.97250569845378994544595329183e-2;
const A32: c_double = 5.91751709536136983633785987549e-2;
const A41: c_double = 2.95875854768068491816892993775e-2;
const A43: c_double = 8.87627564304205475450678981324e-2;
const A51: c_double = 2.41365134159266685502369798665e-1;
const A53: c_double = -8.84549479328286085344864962717e-1;
const A54: c_double = 9.24834003261792003115737966543e-1;
const A61: c_double = 3.7037037037037037037037037037e-2;
const A64: c_double = 1.70828608729473871279604482173e-1;
const A65: c_double = 1.25467687566822425016691814123e-1;
const A71: c_double = 3.7109375e-2;
const A74: c_double = 1.70252211019544039314978060272e-1;
const A75: c_double = 6.02165389804559606850219397283e-2;
const A76: c_double = -1.7578125e-2;
const A81: c_double = 3.70920001185047927108779319836e-2;
const A84: c_double = 1.70383925712239993810214054705e-1;
const A85: c_double = 1.07262030446373284651809199168e-1;
const A86: c_double = -1.53194377486244017527936158236e-2;
const A87: c_double = 8.27378916381402288758473766002e-3;
const A91: c_double = 6.24110958716075717114429577812e-1;
const A94: c_double = -3.36089262944694129406857109825e0;
const A95: c_double = -8.68219346841726006818189891453e-1;
const A96: c_double = 2.75920996994467083049415600797e1;
const A97: c_double = 2.01540675504778934086186788979e1;
const A98: c_double = -4.34898841810699588477366255144e1;
const A101: c_double = 4.77662536438264365890433908527e-1;
const A104: c_double = -2.48811461997166764192642586468e0;
const A105: c_double = -5.90290826836842996371446475743e-1;
const A106: c_double = 2.12300514481811942347288949897e1;
const A107: c_double = 1.52792336328824235832596922938e1;
const A108: c_double = -3.32882109689848629194453265587e1;
const A109: c_double = -2.03312017085086261358222928593e-2;
const A111: c_double = -9.3714243008598732571704021658e-1;
const A114: c_double = 5.18637242884406370830023853209e0;
const A115: c_double = 1.09143734899672957818500254654e0;
const A116: c_double = -8.14978701074692612513997267357e0;
const A117: c_double = -1.85200656599969598641566180701e1;
const A118: c_double = 2.27394870993505042818970056734e1;
const A119: c_double = 2.49360555267965238987089396762e0;
const A1110: c_double = -3.0467644718982195003823669022e0;
const A121: c_double = 2.27331014751653820792359768449e0;
const A124: c_double = -1.05344954667372501984066689879e1;
const A125: c_double = -2.00087205822486249909675718444e0;
const A126: c_double = -1.79589318631187989172765950534e1;
const A127: c_double = 2.79488845294199600508499808837e1;
const A128: c_double = -2.85899827713502369474065508674e0;
const A129: c_double = -8.87285693353062954433549289258e0;
const A1210: c_double = 1.23605671757943030647266201528e1;
const A1211: c_double = 6.43392746015763530355970484046e-1;
const A141: c_double = 5.61675022830479523392909219681e-2;
const A147: c_double = 2.53500210216624811088794765333e-1;
const A148: c_double = -2.46239037470802489917441475441e-1;
const A149: c_double = -1.24191423263816360469010140626e-1;
const A1410: c_double = 1.5329179827876569731206322685e-1;
const A1411: c_double = 8.20105229563468988491666602057e-3;
const A1412: c_double = 7.56789766054569976138603589584e-3;
const A1413: c_double = -8.298e-3;
const A151: c_double = 3.18346481635021405060768473261e-2;
const A156: c_double = 2.83009096723667755288322961402e-2;
const A157: c_double = 5.35419883074385676223797384372e-2;
const A158: c_double = -5.49237485713909884646569340306e-2;
const A1511: c_double = -1.08347328697249322858509316994e-4;
const A1512: c_double = 3.82571090835658412954920192323e-4;
const A1513: c_double = -3.40465008687404560802977114492e-4;
const A1514: c_double = 1.41312443674632500278074618366e-1;
const A161: c_double = -4.28896301583791923408573538692e-1;
const A166: c_double = -4.69762141536116384314449447206e0;
const A167: c_double = 7.68342119606259904184240953878e0;
const A168: c_double = 4.06898981839711007970213554331e0;
const A169: c_double = 3.56727187455281109270669543021e-1;
const A1613: c_double = -1.39902416515901462129418009734e-3;
const A1614: c_double = 2.9475147891527723389556272149e0;
const A1615: c_double = -9.15095847217987001081870187138e0;

const B1: c_double = 5.42937341165687622380535766363e-2;
const B6: c_double = 4.45031289275240888144113950566;
const B7: c_double = 1.89151789931450038304281599044;
const B8: c_double = -5.8012039600105847814672114227;
const B9: c_double = 3.1116436695781989440891606237e-1;
const B10: c_double = -1.52160949662516078556178806805e-1;
const B11: c_double = 2.01365400804030348374776537501e-1;
const B12: c_double = 4.47106157277725905176885569043e-2;
const BHH1: c_double = 0.244094488188976377952755905512;
const BHH2: c_double = 0.733846688281611857341361741547;
const BHH3: c_double = 0.220588235294117647058823529412e-1;

const D41: c_double = -0.84289382761090128651353491142e1;
const D46: c_double = 0.56671495351937776962531783590;
const D47: c_double = -0.30689499459498916912797304727e1;
const D48: c_double = 0.23846676565120698287728149680e1;
const D49: c_double = 0.21170345824450282767155149946e1;
const D410: c_double = -0.87139158377797299206789907490;
const D411: c_double = 0.22404374302607882758541771650e1;
const D412: c_double = 0.63157877876946881815570249290;
const D413: c_double = -0.88990336451333310820698117400e-1;
const D414: c_double = 0.18148505520854727256656404962e2;
const D415: c_double = -0.91946323924783554000451984436e1;
const D416: c_double = -0.44360363875948939664310572000e1;
const D51: c_double = 0.10427508642579134603413151009e2;
const D56: c_double = 0.24228349177525818288430175319e3;
const D57: c_double = 0.16520045171727028198505394887e3;
const D58: c_double = -0.37454675472269020279518312152e3;
const D59: c_double = -0.22113666853125306036270938578e2;
const D510: c_double = 0.77334326684722638389603898808e1;
const D511: c_double = -0.30674084731089398182061213626e2;
const D512: c_double = -0.93321305264302278729567221706e1;
const D513: c_double = 0.15697238121770843886131091075e2;
const D514: c_double = -0.31139403219565177677282850411e2;
const D515: c_double = -0.93529243588444783865713862664e1;
const D516: c_double = 0.35816841486394083752465898540e2;
const D61: c_double = 0.19985053242002433820987653617e2;
const D66: c_double = -0.38703730874935176555105901742e3;
const D67: c_double = -0.18917813819516756882830838328e3;
const D68: c_double = 0.52780815920542364900561016686e3;
const D69: c_double = -0.11573902539959630126141871134e2;
const D610: c_double = 0.68812326946963000169666922661e1;
const D611: c_double = -0.10006050966910838403183860980e1;
const D612: c_double = 0.77771377980534432092869265740;
const D613: c_double = -0.27782057523535084065932004339e1;
const D614: c_double = -0.60196695231264120758267380846e2;
const D615: c_double = 0.84320405506677161018159903784e2;
const D616: c_double = 0.11992291136182789328035130030e2;
const D71: c_double = -0.25693933462703749003312586129e2;
const D76: c_double = -0.15418974869023643374053993627e3;
const D77: c_double = -0.23152937917604549567536039109e3;
const D78: c_double = 0.35763911791061412378285349910e3;
const D79: c_double = 0.93405324183624310003907691704e2;
const D710: c_double = -0.37458323136451633156875139351e2;
const D711: c_double = 0.10409964950896230045147246184e3;
const D712: c_double = 0.29840293426660503123344363579e2;
const D713: c_double = -0.43533456590011143754432175058e2;
const D714: c_double = 0.96324553959188282948394950600e2;
const D715: c_double = -0.39177261675615439165231486172e2;
const D716: c_double = -0.14972683625798562581422125276e3;

const ER1: c_double = 0.1312004499419488073250102996e-1;
const ER6: c_double = -0.1225156446376204440720569753e1;
const ER7: c_double = -0.4957589496572501915214079952;
const ER8: c_double = 0.1664377182454986536961530415e1;
const ER9: c_double = -0.3503288487499736816886487290;
const ER10: c_double = 0.3341791187130174790297318841;
const ER11: c_double = 0.8192320648511571246570742613e-1;
const ER12: c_double = -0.2235530786388629525884427845e-1;

#[inline]
fn c_exp(value: c_double) -> c_double {
    unsafe { exp(value) }
}

#[inline]
fn c_fabs(value: c_double) -> c_double {
    unsafe { fabs(value) }
}

#[inline]
fn c_pow(value: c_double, power: c_double) -> c_double {
    unsafe { pow(value, power) }
}

#[inline]
fn c_sqrt(value: c_double) -> c_double {
    unsafe { sqrt(value) }
}

#[inline]
fn c_min(a: c_double, b: c_double) -> c_double {
    if a < b { a } else { b }
}

#[inline]
fn c_max(a: c_double, b: c_double) -> c_double {
    if a > b { a } else { b }
}

#[inline]
fn custom_sign(x: c_double, y: c_double) -> c_double {
    if y > 0.0 { c_fabs(x) } else { -c_fabs(x) }
}

#[allow(clippy::too_many_arguments)]
/// Runs the galpy-compatible DOP853 integrator.
///
/// # Safety
///
/// `func` must be a valid callback. `dim` must be positive and `nt` must be at
/// least two. `y0`, `t`, and `result` must be aligned, mutually non-overlapping,
/// and valid for `dim`, `nt`, and `nt * dim` elements respectively. The
/// callback must accept the supplied `potential_args` and every temporary
/// `dim`-element state and acceleration buffer passed to it.
pub unsafe fn dop853(
    func: FuncPtr,
    dim: c_int,
    y0: *mut c_double,
    nt: c_int,
    _dt: c_double,
    t: *mut c_double,
    nargs: c_int,
    potential_args: *mut potentialArg,
    rtol: c_double,
    atol: c_double,
    result: *mut c_double,
    _err: *mut c_int,
) {
    let dim = usize::try_from(dim).expect("dop853: dim must be non-negative");
    let nt = usize::try_from(nt).expect("dop853: nt must be non-negative");
    assert!(dim > 0, "dop853: dim must be positive");
    assert!(nt > 1, "dop853: nt must be at least two");

    let f = func.expect("dop853: func pointer was null");
    let y0 = unsafe { core::slice::from_raw_parts_mut(y0, dim) };
    let t = unsafe { core::slice::from_raw_parts(t, nt) };
    let result = unsafe { core::slice::from_raw_parts_mut(result, nt * dim) };

    let rtol = c_exp(rtol);
    let atol = c_exp(atol);
    let safe = 0.9;
    let beta = 0.0;
    let mut facold = 1.0e-4;
    let expo1 = 1.0 / 8.0 - beta * 0.2;
    let facc1 = 1.0 / 0.333;
    let facc2 = 1.0 / 6.0;
    let hmax = t[nt - 1] - t[0];
    let pos_neg = custom_sign(1.0, hmax);

    let mut yy1 = vec![0.0; dim];
    let mut yy_temp = vec![0.0; dim];
    let mut k1 = vec![0.0; dim];
    let mut k2 = vec![0.0; dim];
    let mut k3 = vec![0.0; dim];
    let mut k4 = vec![0.0; dim];
    let mut k5 = vec![0.0; dim];
    let mut k6 = vec![0.0; dim];
    let mut k7 = vec![0.0; dim];
    let mut k8 = vec![0.0; dim];
    let mut k9 = vec![0.0; dim];
    let mut k10 = vec![0.0; dim];
    let mut rcont1 = vec![0.0; dim];
    let mut rcont2 = vec![0.0; dim];
    let mut rcont3 = vec![0.0; dim];
    let mut rcont4 = vec![0.0; dim];
    let mut rcont5 = vec![0.0; dim];
    let mut rcont6 = vec![0.0; dim];
    let mut rcont7 = vec![0.0; dim];
    let mut rcont8 = vec![0.0; dim];

    result[..dim].copy_from_slice(y0);
    let mut result_offset = dim;

    f(
        t[0],
        y0.as_mut_ptr(),
        k1.as_mut_ptr(),
        nargs,
        potential_args,
    );

    let mut dnf = 0.0;
    let mut dny = 0.0;
    for i in 0..dim {
        let sk = atol + rtol * c_fabs(y0[i]);
        let mut sqr = k1[i] / sk;
        dnf += sqr * sqr;
        sqr = y0[i] / sk;
        dny += sqr * sqr;
    }

    let mut h = custom_sign(c_min(c_sqrt(dny / dnf) * 0.01, c_fabs(hmax)), pos_neg);
    for i in 0..dim {
        k3[i] = y0[i] + h * k1[i];
    }
    f(
        t[0] + h,
        k3.as_mut_ptr(),
        k2.as_mut_ptr(),
        nargs,
        potential_args,
    );

    let mut der2 = 0.0;
    for i in 0..dim {
        let sk = atol + rtol * c_fabs(y0[i]);
        let sqr = (k2[i] - k1[i]) / sk;
        der2 += sqr * sqr;
    }
    der2 = c_sqrt(der2) / h;
    let der12 = c_max(c_fabs(der2), c_sqrt(dnf));
    let h1 = c_pow(0.01 / der12, 1.0 / 8.0);
    h = custom_sign(
        c_min(100.0 * c_fabs(h), c_min(c_fabs(h1), c_fabs(hmax))),
        pos_neg,
    );

    let mut reject = 0;
    let mut t_current = t[0];
    let mut t_old = t[0];
    let mut t_old_older;
    let mut finished_user_t_ii = 0;

    while finished_user_t_ii < nt - 1 {
        h = pos_neg * c_max(c_fabs(h), 1e3 * UROUND);

        for i in 0..dim {
            yy1[i] = y0[i] + h * A21 * k1[i];
        }
        f(
            t_current + C2 * h,
            yy1.as_mut_ptr(),
            k2.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] = y0[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        f(
            t_current + C3 * h,
            yy1.as_mut_ptr(),
            k3.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] = y0[i] + h * (A41 * k1[i] + A43 * k3[i]);
        }
        f(
            t_current + C4 * h,
            yy1.as_mut_ptr(),
            k4.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] = y0[i] + h * (A51 * k1[i] + A53 * k3[i] + A54 * k4[i]);
        }
        f(
            t_current + C5 * h,
            yy1.as_mut_ptr(),
            k5.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] = y0[i] + h * (A61 * k1[i] + A64 * k4[i] + A65 * k5[i]);
        }
        f(
            t_current + C6 * h,
            yy1.as_mut_ptr(),
            k6.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] = y0[i] + h * (A71 * k1[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
        }
        f(
            t_current + C7 * h,
            yy1.as_mut_ptr(),
            k7.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] =
                y0[i] + h * (A81 * k1[i] + A84 * k4[i] + A85 * k5[i] + A86 * k6[i] + A87 * k7[i]);
        }
        f(
            t_current + C8 * h,
            yy1.as_mut_ptr(),
            k8.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] = y0[i]
                + h * (A91 * k1[i]
                    + A94 * k4[i]
                    + A95 * k5[i]
                    + A96 * k6[i]
                    + A97 * k7[i]
                    + A98 * k8[i]);
        }
        f(
            t_current + C9 * h,
            yy1.as_mut_ptr(),
            k9.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] = y0[i]
                + h * (A101 * k1[i]
                    + A104 * k4[i]
                    + A105 * k5[i]
                    + A106 * k6[i]
                    + A107 * k7[i]
                    + A108 * k8[i]
                    + A109 * k9[i]);
        }
        f(
            t_current + C10 * h,
            yy1.as_mut_ptr(),
            k10.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] = y0[i]
                + h * (A111 * k1[i]
                    + A114 * k4[i]
                    + A115 * k5[i]
                    + A116 * k6[i]
                    + A117 * k7[i]
                    + A118 * k8[i]
                    + A119 * k9[i]
                    + A1110 * k10[i]);
        }
        f(
            t_current + C11 * h,
            yy1.as_mut_ptr(),
            k2.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            yy1[i] = y0[i]
                + h * (A121 * k1[i]
                    + A124 * k4[i]
                    + A125 * k5[i]
                    + A126 * k6[i]
                    + A127 * k7[i]
                    + A128 * k8[i]
                    + A129 * k9[i]
                    + A1210 * k10[i]
                    + A1211 * k2[i]);
        }

        t_old_older = t_old;
        t_old = t_current;
        t_current = t_current + h;

        f(
            t_current,
            yy1.as_mut_ptr(),
            k3.as_mut_ptr(),
            nargs,
            potential_args,
        );

        for i in 0..dim {
            k4[i] = B1 * k1[i]
                + B6 * k6[i]
                + B7 * k7[i]
                + B8 * k8[i]
                + B9 * k9[i]
                + B10 * k10[i]
                + B11 * k2[i]
                + B12 * k3[i];
            k5[i] = y0[i] + h * k4[i];
        }

        let mut err = 0.0;
        let mut err2 = 0.0;
        for i in 0..dim {
            let sk = atol + rtol * c_max(c_fabs(y0[i]), c_fabs(k5[i]));
            let mut erri = k4[i] - BHH1 * k1[i] - BHH2 * k9[i] - BHH3 * k3[i];
            let mut sqr = erri / sk;
            err2 += sqr * sqr;
            erri = ER1 * k1[i]
                + ER6 * k6[i]
                + ER7 * k7[i]
                + ER8 * k8[i]
                + ER9 * k9[i]
                + ER10 * k10[i]
                + ER11 * k2[i]
                + ER12 * k3[i];
            sqr = erri / sk;
            err += sqr * sqr;
        }
        let mut deno = err + 0.01 * err2;
        if deno <= 0.0 {
            deno = 1.0;
        }
        err = c_fabs(h) * err * c_sqrt(1.0 / (deno * dim as c_double));

        let fac11 = c_pow(err, expo1);
        let mut fac = fac11 / c_pow(facold, beta);
        fac = c_max(facc2, c_min(facc1, fac / safe));
        let mut hnew = h / fac;

        if err <= 1.0 {
            facold = c_max(err, 1.0e-4);
            f(
                t_current,
                k5.as_mut_ptr(),
                k4.as_mut_ptr(),
                nargs,
                potential_args,
            );

            for i in 0..dim {
                rcont1[i] = y0[i];
                let ydiff = k5[i] - y0[i];
                rcont2[i] = ydiff;
                let bspl = h * k1[i] - ydiff;
                rcont3[i] = bspl;
                rcont4[i] = ydiff - h * k4[i] - bspl;
                rcont5[i] = D41 * k1[i]
                    + D46 * k6[i]
                    + D47 * k7[i]
                    + D48 * k8[i]
                    + D49 * k9[i]
                    + D410 * k10[i]
                    + D411 * k2[i]
                    + D412 * k3[i];
                rcont6[i] = D51 * k1[i]
                    + D56 * k6[i]
                    + D57 * k7[i]
                    + D58 * k8[i]
                    + D59 * k9[i]
                    + D510 * k10[i]
                    + D511 * k2[i]
                    + D512 * k3[i];
                rcont7[i] = D61 * k1[i]
                    + D66 * k6[i]
                    + D67 * k7[i]
                    + D68 * k8[i]
                    + D69 * k9[i]
                    + D610 * k10[i]
                    + D611 * k2[i]
                    + D612 * k3[i];
                rcont8[i] = D71 * k1[i]
                    + D76 * k6[i]
                    + D77 * k7[i]
                    + D78 * k8[i]
                    + D79 * k9[i]
                    + D710 * k10[i]
                    + D711 * k2[i]
                    + D712 * k3[i];
            }

            for i in 0..dim {
                yy1[i] = y0[i]
                    + h * (A141 * k1[i]
                        + A147 * k7[i]
                        + A148 * k8[i]
                        + A149 * k9[i]
                        + A1410 * k10[i]
                        + A1411 * k2[i]
                        + A1412 * k3[i]
                        + A1413 * k4[i]);
            }
            f(
                t_old + C14 * h,
                yy1.as_mut_ptr(),
                k10.as_mut_ptr(),
                nargs,
                potential_args,
            );

            for i in 0..dim {
                yy1[i] = y0[i]
                    + h * (A151 * k1[i]
                        + A156 * k6[i]
                        + A157 * k7[i]
                        + A158 * k8[i]
                        + A1511 * k2[i]
                        + A1512 * k3[i]
                        + A1513 * k4[i]
                        + A1514 * k10[i]);
            }
            f(
                t_old + C15 * h,
                yy1.as_mut_ptr(),
                k2.as_mut_ptr(),
                nargs,
                potential_args,
            );

            for i in 0..dim {
                yy1[i] = y0[i]
                    + h * (A161 * k1[i]
                        + A166 * k6[i]
                        + A167 * k7[i]
                        + A168 * k8[i]
                        + A169 * k9[i]
                        + A1613 * k4[i]
                        + A1614 * k10[i]
                        + A1615 * k2[i]);
            }
            f(
                t_old + C16 * h,
                yy1.as_mut_ptr(),
                k3.as_mut_ptr(),
                nargs,
                potential_args,
            );

            for i in 0..dim {
                rcont5[i] =
                    h * (rcont5[i] + D413 * k4[i] + D414 * k10[i] + D415 * k2[i] + D416 * k3[i]);
                rcont6[i] =
                    h * (rcont6[i] + D513 * k4[i] + D514 * k10[i] + D515 * k2[i] + D516 * k3[i]);
                rcont7[i] =
                    h * (rcont7[i] + D613 * k4[i] + D614 * k10[i] + D615 * k2[i] + D616 * k3[i]);
                rcont8[i] =
                    h * (rcont8[i] + D713 * k4[i] + D714 * k10[i] + D715 * k2[i] + D716 * k3[i]);
                k1[i] = k4[i];
                y0[i] = k5[i];
            }

            while finished_user_t_ii < nt - 1
                && pos_neg * t[finished_user_t_ii + 1] < pos_neg * t_current
            {
                let s = (t[finished_user_t_ii + 1] - t_old) / h;
                let s1 = 1.0 - s;
                for i in 0..dim {
                    yy_temp[i] = rcont1[i]
                        + s * (rcont2[i]
                            + s1 * (rcont3[i]
                                + s * (rcont4[i]
                                    + s1 * (rcont5[i]
                                        + s * (rcont6[i] + s1 * (rcont7[i] + s * rcont8[i]))))));
                }
                result[result_offset..result_offset + dim].copy_from_slice(&yy_temp);
                result_offset += dim;
                finished_user_t_ii += 1;
            }

            hnew = if c_fabs(hnew) > c_fabs(hmax) {
                pos_neg * hmax
            } else {
                hnew
            };
            if reject != 0 {
                hnew = pos_neg * c_min(c_fabs(hnew), c_fabs(h));
            }
            reject = 0;
        } else {
            hnew = h / c_min(facc1, fac11 / safe);
            reject = 1;
            t_current = t_old;
            t_old = t_old_older;
        }

        h = hnew;
    }
}
