#[cfg(test)]
mod tests {
    use drift_rs::integrators::dopr54_cpu::*;
    use std::ptr;
    use libc::{self, c_double, c_int};
    use std::fs::File;
    use std::io::{self, BufRead, BufReader};
    use std::path::Path;

    extern "C" fn kepler_rhs(
        _t: c_double,
        q: *mut c_double,
        a: *mut c_double,
        _nargs: c_int,
        _pot_args: *mut potentialArg,
    ) {
        unsafe {
            let x = *q.add(0);
            let y = *q.add(1);
            let z = *q.add(2);
            let vx = *q.add(3);
            let vy = *q.add(4);
            let vz = *q.add(5);

            let r2 = x * x + y * y + z * z;
            // guard against r=0 just in case
            let r2_safe = if r2 == 0.0 { 1e-16 } else { r2 };
            let r = sqrt(r2_safe);
            let inv_r3 = 1.0 / (r2_safe * r); // 1 / r^3

            *a.add(0) = vx;
            *a.add(1) = vy;
            *a.add(2) = vz;

            *a.add(3) = -x * inv_r3;
            *a.add(4) = -y * inv_r3;
            *a.add(5) = -z * inv_r3;
        }
    }

    #[test]
    fn dopr54_cpu_matches_reference() {
        unsafe {           

            let init = parse_dopr54_dump("dopr54_init_dump.txt").expect("could not parse init dump");

            let dim = init.dim;
            let nt = init.nt;
            let nargs = init.nargs;
            assert_eq!(init.t.len(), nt as usize, "t length != nt");
            assert_eq!(init.yo.len(), dim as usize, "yo length != dim");

            let mut t = init.t.clone();
            let mut yo = init.yo.clone();

            let mut result = vec![0.0 as c_double; (nt as usize) * (dim as usize)];
            let mut err: c_int = 0;
            let pot_ptr: *mut potentialArg = ptr::null_mut();

            dopr54(
                Some(kepler_rhs),
                dim,
                yo.as_mut_ptr(),
                nt,
                init.dt_one,
                t.as_mut_ptr(),
                nargs,
                pot_ptr,
                init.rtol,
                init.atol,
                result.as_mut_ptr(),
                &mut err,
            );
            
            for step in 0..(nt as usize) {
                println!("step {}", step);
                for j in 0..(dim as usize) {
                    let v = result[step * (dim as usize) + j];
                    println!(" (0x{:016x})", v.to_bits());
                }
            }
            
            let expected: [u64; 6] = [
                0xbfead9ac890cbf34,
                0xbfe1689ef5f2f595,
                0x0000000000000000,
                0x3fe1689ef5f2da49,
                0xbfead9ac890ca8be,
                0x0000000000000000,
            ];

            let tail = &result[result.len() - 6..];

            for (i, &actual_f64) in tail.iter().enumerate() {
                let actual_bits = actual_f64.to_bits();
                let expected_bits = expected[i];

                assert_eq!(
                    actual_bits,
                    expected_bits,
                    "Mismatch in tail element {i}: got 0x{:016x}, expected 0x{:016x}",
                    actual_bits,
                    expected_bits
                );
            }

        }
    }

    struct DumpData {
        dim: c_int,
        nt: c_int,
        dt_one: c_double,
        rtol: c_double,
        atol: c_double,
        t: Vec<c_double>,
        yo: Vec<c_double>,
        nargs: c_int,
    }

    fn parse_dopr54_dump<P: AsRef<Path>>(path: P) -> io::Result<DumpData> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut dim: Option<c_int> = None;
        let mut nt: Option<c_int> = None;
        let mut dt_one: Option<c_double> = None;
        let mut rtol: Option<c_double> = None;
        let mut atol: Option<c_double> = None;
        let mut t: Vec<c_double> = Vec::new();
        let mut yo: Vec<c_double> = Vec::new();
        let mut nargs: c_int = 0;
        let mut args: Vec<c_double> = Vec::new();

        for line_res in reader.lines() {
            let line = line_res?;
            let mut parts = line.split_whitespace();
            let key = match parts.next() {
                Some(k) => k,
                None => continue,
            };
            match key {
                "dim" => {
                    if let Some(v) = parts.next() {
                        dim = Some(v.parse().unwrap());
                    }
                }
                "nt" => {
                    if let Some(v) = parts.next() {
                        nt = Some(v.parse().unwrap());
                    }
                }
                "dt_one" => {
                    if let Some(v) = parts.next() {
                        dt_one = Some(v.parse().unwrap());
                    }
                }
                "rtol" => {
                    if let Some(v) = parts.next() {
                        rtol = Some(v.parse().unwrap());
                    }
                }
                "atol" => {
                    if let Some(v) = parts.next() {
                        atol = Some(v.parse().unwrap());
                    }
                }
                "t_hex" => {
                    t.clear();
                    for v in parts {
                        t.push(parse_hex_f64(v));
                    }
                }
                "yo" => {
                    for v in parts {
                        yo.push(v.parse().unwrap());
                    }
                }

                "nargs" => {
                    if let Some(v) = parts.next() {
                        nargs = v.parse().unwrap();
                    }
                }
                "args" => {
                    for v in parts {
                        args.push(v.parse().unwrap());
                    }
                }
                _ => {}
            }
        }

        Ok(DumpData {
            dim: dim.expect("dim missing in dump"),
            nt: nt.expect("nt missing in dump"),
            dt_one: dt_one.unwrap_or(-9999.99),
            rtol: rtol.expect("rtol missing in dump"),
            atol: atol.expect("atol missing in dump"),
            t,
            yo,
            nargs,
        })
    }

    fn parse_hex_f64(s: &str) -> f64 {
        let bits = u64::from_str_radix(s, 16)
            .unwrap_or_else(|e| panic!("failed to parse hex f64 '{}': {}", s, e));
        f64::from_bits(bits)
    }    
}