// GPU device code: the steppers are `#[inline(always)]` so they end up inlined
// into the generated NVVM/PTX kernels (no device-call overhead), and variable
// names (`der12`, `erri`, `fac11`, ...) follow Hairer's reference DOP853
// implementation. Several `clippy::pedantic` lints are therefore relaxed for
// the whole module.
#![allow(
    clippy::inline_always,
    clippy::too_many_arguments, // argument list follows the reference API
    clippy::too_many_lines, // stepper is kept monolithic like the reference
    clippy::similar_names, // names follow Hairer's reference implementation
    clippy::many_single_char_names // q/a/x/y/z state components
)]

#[cfg(all(target_os = "cuda", feature = "rust-cuda"))]
use cuda_std::GpuFloat;

const DIM: usize = 6;

#[allow(clippy::cast_precision_loss)] // DIM is a tiny compile-time constant
const DIM_F64: f64 = DIM as f64;
const UROUND: f64 = 2.3e-16;

const C2: f64 = 0.526_001_519_587_677_318_785_587_544_488e-1;
const C3: f64 = 0.789_002_279_381_515_978_178_381_316_732e-1;
const C4: f64 = 0.118_350_341_907_227_396_726_757_197_510;
const C5: f64 = 0.281_649_658_092_772_603_273_242_802_490;
const C6: f64 = 0.333_333_333_333_333_333_333_333_333_333;
const C7: f64 = 0.25;
const C8: f64 = 0.307_692_307_692_307_692_307_692_307_692;
const C9: f64 = 0.651_282_051_282_051_282_051_282_051_282;
const C10: f64 = 0.6;
const C11: f64 = 0.857_142_857_142_857_142_857_142_857_142;
const C14: f64 = 0.1;
const C15: f64 = 0.2;
const C16: f64 = 0.777_777_777_777_777_777_777_777_777_778;

const A21: f64 = 5.260_015_195_876_773_187_855_875_444_88e-2;
const A31: f64 = 1.972_505_698_453_789_945_445_953_291_83e-2;
const A32: f64 = 5.917_517_095_361_369_836_337_859_875_49e-2;
const A41: f64 = 2.958_758_547_680_684_918_168_929_937_75e-2;
const A43: f64 = 8.876_275_643_042_054_754_506_789_813_24e-2;
const A51: f64 = 2.413_651_341_592_666_855_023_697_986_65e-1;
const A53: f64 = -8.845_494_793_282_861e-1;
const A54: f64 = 9.248_340_032_617_920_031_157_379_665_43e-1;
const A61: f64 = 3.703_703_703_703_703_5e-2;
const A64: f64 = 1.708_286_087_294_738_712_796_044_821_73e-1;
const A65: f64 = 1.254_676_875_668_224_250_166_918_141_23e-1;
const A71: f64 = 3.710_937_5e-2;
const A74: f64 = 1.702_522_110_195_440_393_149_780_602_72e-1;
const A75: f64 = 6.021_653_898_045_596_068_502_193_972_83e-2;
const A76: f64 = -1.757_812_5e-2;
const A81: f64 = 3.709_200_011_850_479_271_087_793_198_36e-2;
const A84: f64 = 1.703_839_257_122_399_938_102_140_547_05e-1;
const A85: f64 = 1.072_620_304_463_732_846_518_091_991_68e-1;
const A86: f64 = -1.531_943_774_862_440_2e-2;
const A87: f64 = 8.273_789_163_814_022_887_584_737_660_02e-3;
const A91: f64 = 6.241_109_587_160_757_171_144_295_778_12e-1;
const A94: f64 = -3.360_892_629_446_941_4;
const A95: f64 = -8.682_193_468_417_26e-1;
const A96: f64 = 2.759_209_969_944_670_830_494_156_007_97e1;
const A97: f64 = 2.015_406_755_047_789_340_861_867_889_79e1;
const A98: f64 = -4.348_988_418_106_996e1;
const A101: f64 = 4.776_625_364_382_643_658_904_339_085_27e-1;
const A104: f64 = -2.488_114_619_971_667_7;
const A105: f64 = -5.902_908_268_368_43e-1;
const A106: f64 = 2.123_005_144_818_119_423_472_889_498_97e1;
const A107: f64 = 1.527_923_363_288_242_358_325_969_229_38e1;
const A108: f64 = -3.328_821_096_898_486e1;
const A109: f64 = -2.033_120_170_850_862_7e-2;
const A111: f64 = -9.371_424_300_859_873e-1;
const A114: f64 = 5.186_372_428_844_063_708_300_238_532_09;
const A115: f64 = 1.091_437_348_996_729_578_185_002_546_54;
const A116: f64 = -8.149_787_010_746_927;
const A117: f64 = -1.852_006_565_999_696e1;
const A118: f64 = 2.273_948_709_935_050_428_189_700_567_34e1;
const A119: f64 = 2.493_605_552_679_652_389_870_893_967_62;
const A1110: f64 = -3.046_764_471_898_219_6;
const A121: f64 = 2.273_310_147_516_538_207_923_597_684_49;
const A124: f64 = -1.053_449_546_673_725e1;
const A125: f64 = -2.000_872_058_224_862_5;
const A126: f64 = -1.795_893_186_311_88e1;
const A127: f64 = 2.794_888_452_941_996_005_084_998_088_37e1;
const A128: f64 = -2.858_998_277_135_023_5;
const A129: f64 = -8.872_856_933_530_63;
const A1210: f64 = 1.236_056_717_579_430_306_472_662_015_28e1;
const A1211: f64 = 6.433_927_460_157_635_303_559_704_840_46e-1;
const A141: f64 = 5.616_750_228_304_795_233_929_092_196_81e-2;
const A147: f64 = 2.535_002_102_166_248_110_887_947_653_33e-1;
const A148: f64 = -2.462_390_374_708_025e-1;
const A149: f64 = -1.241_914_232_638_163_7e-1;
const A1410: f64 = 1.532_917_982_787_656_8e-1;
const A1411: f64 = 8.201_052_295_634_689_884_916_666_020_57e-3;
const A1412: f64 = 7.567_897_660_545_699_761_386_035_895_84e-3;
const A1413: f64 = -8.298e-3;
const A151: f64 = 3.183_464_816_350_214_050_607_684_732_61e-2;
const A156: f64 = 2.830_090_967_236_677_552_883_229_614_02e-2;
const A157: f64 = 5.354_198_830_743_856_762_237_973_843_72e-2;
const A158: f64 = -5.492_374_857_139_099e-2;
const A1511: f64 = -1.083_473_286_972_493_2e-4;
const A1512: f64 = 3.825_710_908_356_584_129_549_201_923_23e-4;
const A1513: f64 = -3.404_650_086_874_045_6e-4;
const A1514: f64 = 1.413_124_436_746_325_002_780_746_183_66e-1;
const A161: f64 = -4.288_963_015_837_919_4e-1;
const A166: f64 = -4.697_621_415_361_164;
const A167: f64 = 7.683_421_196_062_599_041_842_409_538_78;
const A168: f64 = 4.068_989_818_397_110_079_702_135_543_31;
const A169: f64 = 3.567_271_874_552_811_092_706_695_430_21e-1;
const A1613: f64 = -1.399_024_165_159_014_5e-3;
const A1614: f64 = 2.947_514_789_152_772_4;
const A1615: f64 = -9.150_958_472_179_87;

const B1: f64 = 5.429_373_411_656_876_223_805_357_663_63e-2;
const B6: f64 = 4.450_312_892_752_408_881_441_139_505_66;
const B7: f64 = 1.891_517_899_314_500_383_042_815_990_44;
const B8: f64 = -5.801_203_960_010_585;
const B9: f64 = 3.111_643_669_578_199e-1;
const B10: f64 = -1.521_609_496_625_161e-1;
const B11: f64 = 2.013_654_008_040_303_483_747_765_375_01e-1;
const B12: f64 = 4.471_061_572_777_259_051_768_855_690_43e-2;
const BHH1: f64 = 0.244_094_488_188_976_377_952_755_905_512;
const BHH2: f64 = 0.733_846_688_281_611_857_341_361_741_547;
const BHH3: f64 = 0.220_588_235_294_117_647_058_823_529_412e-1;

const D41: f64 = -8.428_938_276_109_013;
const D46: f64 = 0.566_714_953_519_377_7;
const D47: f64 = -3.068_949_945_949_891_7;
const D48: f64 = 2.384_667_656_512_07;
const D49: f64 = 2.117_034_582_445_028;
const D410: f64 = -0.871_391_583_777_973;
const D411: f64 = 2.240_437_430_260_788_3;
const D412: f64 = 0.631_578_778_769_468_8;
const D413: f64 = -8.899_033_645_133_331e-2;
const D414: f64 = 1.814_850_552_085_472_7e1;
const D415: f64 = -9.194_632_392_478_356;
const D416: f64 = -4.436_036_387_594_894;
const D51: f64 = 1.042_750_864_257_913_4e1;
const D56: f64 = 2.422_834_917_752_581_7e2;
const D57: f64 = 1.652_004_517_172_702_8e2;
const D58: f64 = -3.745_467_547_226_902e2;
const D59: f64 = -2.211_366_685_312_530_6e1;
const D510: f64 = 7.733_432_668_472_264;
const D511: f64 = -3.067_408_473_108_939_8e1;
const D512: f64 = -9.332_130_526_430_229;
const D513: f64 = 1.569_723_812_177_084_5e1;
const D514: f64 = -3.113_940_321_956_517_8e1;
const D515: f64 = -9.352_924_358_844_48;
const D516: f64 = 3.581_684_148_639_408e1;
const D61: f64 = 1.998_505_324_200_243_3e1;
const D66: f64 = -3.870_373_087_493_518e2;
const D67: f64 = -1.891_781_381_951_675_8e2;
const D68: f64 = 5.278_081_592_054_236e2;
const D69: f64 = -1.157_390_253_995_963e1;
const D610: f64 = 6.881_232_694_696_3;
const D611: f64 = -1.000_605_096_691_083_8;
const D612: f64 = 0.777_713_779_805_344_3;
const D613: f64 = -2.778_205_752_353_508;
const D614: f64 = -6.019_669_523_126_412e1;
const D615: f64 = 8.432_040_550_667_716e1;
const D616: f64 = 1.199_229_113_618_279e1;
const D71: f64 = -2.569_393_346_270_375e1;
const D76: f64 = -1.541_897_486_902_364_3e2;
const D77: f64 = -2.315_293_791_760_455e2;
const D78: f64 = 3.576_391_179_106_141e2;
const D79: f64 = 9.340_532_418_362_432e1;
const D710: f64 = -3.745_832_313_645_163e1;
const D711: f64 = 1.040_996_495_089_623e2;
const D712: f64 = 2.984_029_342_666_05e1;
const D713: f64 = -4.353_345_659_001_114e1;
const D714: f64 = 9.632_455_395_918_828e1;
const D715: f64 = -3.917_726_167_561_544e1;
const D716: f64 = -1.497_268_362_579_856_4e2;

const ER1: f64 = 1.312_004_499_419_488e-2;
const ER6: f64 = -1.225_156_446_376_204_4;
const ER7: f64 = -0.495_758_949_657_250_2;
const ER8: f64 = 1.664_377_182_454_986_4;
const ER9: f64 = -0.350_328_848_749_973_66;
const ER10: f64 = 0.334_179_118_713_017_5;
const ER11: f64 = 8.192_320_648_511_571e-2;
const ER12: f64 = -2.235_530_786_388_629_4e-2;

#[inline(always)]
fn c_min(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

#[inline(always)]
fn c_max(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

#[inline(always)]
fn custom_sign(x: f64, y: f64) -> f64 {
    if y > 0.0 { x.abs() } else { -x.abs() }
}

#[inline(always)]
fn kepler_rhs(_t: f64, q: &[f64; DIM], a: &mut [f64; DIM]) {
    let x = q[0];
    let y = q[1];
    let z = q[2];

    a[0] = q[3];
    a[1] = q[4];
    a[2] = q[5];

    let radius = (x * x + y * y).sqrt();
    let sin_phi = y / radius;
    let cos_phi = x / radius;
    let radius_squared = radius * radius + z * z;
    let radial_force = -radius * radius_squared.powf(-1.5);
    let phi_torque = 0.0;

    a[3] = cos_phi * radial_force - 1.0 / radius * sin_phi * phi_torque;
    a[4] = sin_phi * radial_force + 1.0 / radius * cos_phi * phi_torque;
    a[5] = -z * radius_squared.powf(-1.5);
}

#[inline(always)]
unsafe fn dop853_integrate_kepler(
    y0: &mut [f64; DIM],
    t: &[f64],
    rtol: f64,
    atol: f64,
    state_out: *mut f64,
    n: usize,
    tid: usize,
) {
    let nt = t.len();
    let rtol = rtol.exp();
    let atol = atol.exp();
    let safe: f64 = 0.9;
    let beta: f64 = 0.0;
    let mut facold: f64 = 1.0e-4;
    let expo1 = 1.0 / 8.0 - beta * 0.2;
    let facc1 = 1.0 / 0.333;
    let facc2 = 1.0 / 6.0;
    let hmax = t[nt - 1] - t[0];
    let pos_neg = custom_sign(1.0, hmax);

    let mut yy1 = [0.0; DIM];
    let mut yy_temp = [0.0; DIM];
    let mut k1 = [0.0; DIM];
    let mut k2 = [0.0; DIM];
    let mut k3 = [0.0; DIM];
    let mut k4 = [0.0; DIM];
    let mut k5 = [0.0; DIM];
    let mut k6 = [0.0; DIM];
    let mut k7 = [0.0; DIM];
    let mut k8 = [0.0; DIM];
    let mut k9 = [0.0; DIM];
    let mut k10 = [0.0; DIM];
    let mut rcont1 = [0.0; DIM];
    let mut rcont2 = [0.0; DIM];
    let mut rcont3 = [0.0; DIM];
    let mut rcont4 = [0.0; DIM];
    let mut rcont5 = [0.0; DIM];
    let mut rcont6 = [0.0; DIM];
    let mut rcont7 = [0.0; DIM];
    let mut rcont8 = [0.0; DIM];

    unsafe { crate::write_time_major_state(state_out, 0, n, tid, y0) };

    kepler_rhs(t[0], y0, &mut k1);

    let mut dnf = 0.0;
    let mut dny = 0.0;
    for i in 0..DIM {
        let sk = atol + rtol * y0[i].abs();
        let mut sqr = k1[i] / sk;
        dnf += sqr * sqr;
        sqr = y0[i] / sk;
        dny += sqr * sqr;
    }

    let mut h = custom_sign(c_min((dny / dnf).sqrt() * 0.01, hmax.abs()), pos_neg);
    for i in 0..DIM {
        k3[i] = y0[i] + h * k1[i];
    }
    kepler_rhs(t[0] + h, &k3, &mut k2);

    let mut der2 = 0.0;
    for i in 0..DIM {
        let sk = atol + rtol * y0[i].abs();
        let sqr = (k2[i] - k1[i]) / sk;
        der2 += sqr * sqr;
    }
    der2 = der2.sqrt() / h;
    let der12 = c_max(der2.abs(), dnf.sqrt());
    let h1 = (0.01 / der12).powf(1.0 / 8.0);
    h = custom_sign(c_min(100.0 * h.abs(), c_min(h1.abs(), hmax.abs())), pos_neg);

    let mut reject = 0;
    let mut t_current = t[0];
    let mut t_old = t[0];
    let mut t_old_older;
    let mut finished_user_t_ii = 0;

    while finished_user_t_ii < nt - 1 {
        h = pos_neg * c_max(h.abs(), 1e3 * UROUND);

        for i in 0..DIM {
            yy1[i] = y0[i] + h * A21 * k1[i];
        }
        kepler_rhs(t_current + C2 * h, &yy1, &mut k2);

        for i in 0..DIM {
            yy1[i] = y0[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        kepler_rhs(t_current + C3 * h, &yy1, &mut k3);

        for i in 0..DIM {
            yy1[i] = y0[i] + h * (A41 * k1[i] + A43 * k3[i]);
        }
        kepler_rhs(t_current + C4 * h, &yy1, &mut k4);

        for i in 0..DIM {
            yy1[i] = y0[i] + h * (A51 * k1[i] + A53 * k3[i] + A54 * k4[i]);
        }
        kepler_rhs(t_current + C5 * h, &yy1, &mut k5);

        for i in 0..DIM {
            yy1[i] = y0[i] + h * (A61 * k1[i] + A64 * k4[i] + A65 * k5[i]);
        }
        kepler_rhs(t_current + C6 * h, &yy1, &mut k6);

        for i in 0..DIM {
            yy1[i] = y0[i] + h * (A71 * k1[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
        }
        kepler_rhs(t_current + C7 * h, &yy1, &mut k7);

        for i in 0..DIM {
            yy1[i] =
                y0[i] + h * (A81 * k1[i] + A84 * k4[i] + A85 * k5[i] + A86 * k6[i] + A87 * k7[i]);
        }
        kepler_rhs(t_current + C8 * h, &yy1, &mut k8);

        for i in 0..DIM {
            yy1[i] = y0[i]
                + h * (A91 * k1[i]
                    + A94 * k4[i]
                    + A95 * k5[i]
                    + A96 * k6[i]
                    + A97 * k7[i]
                    + A98 * k8[i]);
        }
        kepler_rhs(t_current + C9 * h, &yy1, &mut k9);

        for i in 0..DIM {
            yy1[i] = y0[i]
                + h * (A101 * k1[i]
                    + A104 * k4[i]
                    + A105 * k5[i]
                    + A106 * k6[i]
                    + A107 * k7[i]
                    + A108 * k8[i]
                    + A109 * k9[i]);
        }
        kepler_rhs(t_current + C10 * h, &yy1, &mut k10);

        for i in 0..DIM {
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
        kepler_rhs(t_current + C11 * h, &yy1, &mut k2);

        for i in 0..DIM {
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
        t_current += h;

        kepler_rhs(t_current, &yy1, &mut k3);

        for i in 0..DIM {
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
        for i in 0..DIM {
            let sk = atol + rtol * c_max(y0[i].abs(), k5[i].abs());
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
        err = h.abs() * err * (1.0 / (deno * DIM_F64)).sqrt();

        let fac11 = err.powf(expo1);
        let mut fac = fac11 / facold.powf(beta);
        fac = c_max(facc2, c_min(facc1, fac / safe));
        let mut hnew = h / fac;

        if err <= 1.0 {
            facold = c_max(err, 1.0e-4);
            kepler_rhs(t_current, &k5, &mut k4);

            for i in 0..DIM {
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

            for i in 0..DIM {
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
            kepler_rhs(t_old + C14 * h, &yy1, &mut k10);

            for i in 0..DIM {
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
            kepler_rhs(t_old + C15 * h, &yy1, &mut k2);

            for i in 0..DIM {
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
            kepler_rhs(t_old + C16 * h, &yy1, &mut k3);

            for i in 0..DIM {
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
                for i in 0..DIM {
                    yy_temp[i] = rcont1[i]
                        + s * (rcont2[i]
                            + s1 * (rcont3[i]
                                + s * (rcont4[i]
                                    + s1 * (rcont5[i]
                                        + s * (rcont6[i] + s1 * (rcont7[i] + s * rcont8[i]))))));
                }
                unsafe {
                    crate::write_time_major_state(
                        state_out,
                        finished_user_t_ii + 1,
                        n,
                        tid,
                        &yy_temp,
                    );
                };
                finished_user_t_ii += 1;
            }

            hnew = if hnew.abs() > hmax.abs() {
                pos_neg * hmax
            } else {
                hnew
            };
            if reject != 0 {
                hnew = pos_neg * c_min(hnew.abs(), h.abs());
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

/// Integrates one particle and writes its complete trajectory.
///
/// # Safety
///
/// `tid` must be less than `n`; `state0` must contain at least `n * DIM`
/// values; `times` must contain at least `nt` values with `2 <= nt <= 1024`;
/// and `state_out` must point to at least `nt * n * DIM` writable values.
/// Concurrent callers must use distinct particle indices.
pub(crate) unsafe fn integrate_particle(
    tid: usize,
    n: usize,
    nt: usize,
    state0: &[f64],
    times: &[f64],
    state_out: *mut f64,
    rtol: f64,
    atol: f64,
) {
    let t_slice = &times[..nt];
    let mut y0 = [0.0; DIM];
    let base_in = tid * DIM;
    y0.copy_from_slice(&state0[base_in..base_in + DIM]);

    unsafe { dop853_integrate_kepler(&mut y0, t_slice, rtol, atol, state_out, n, tid) };
}
