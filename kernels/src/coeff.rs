use crate::handshake::{load_state,store_state};
use cuda_std::{kernel, thread};
use libm::{floor, pow, sqrt};
use crate::recipes::consume_recipe;
use crate::{expand_statics,thread_id_limit_check};
use shared::{Potential,PotentialRecipe,PotentialEnum,PotentialNames,Config,KeplerPotential};


fn calc_coeff(x0:f64,v0:f64,a0:f64,x1:f64,v1:f64,a1:f64)->(f64,f64,f64,f64,f64,f64){
    let (d,e,f) = (0.5*a0,v0,x0);
    let (gamma,mu,nu) = (x1-x0-v0-d,v1-a0-v0,a1-a0);
    let (a,b,c) = (6.*gamma-3.*mu+0.5*nu,-15.*gamma+7.*mu-nu,10.*gamma-4.*mu+0.5*nu);
    (a,b,c,d,e,f)
}

#[kernel]
pub unsafe fn coeff_kernel(
    state_out: *mut f64, //pointer
    coeff: *mut f64, //pointer
    statics : Config,
    recipe: [PotentialRecipe;10],
    supertable: *mut f64,
) {
    let (n,steps_cap,t_end,atol,rtol,fac_min,fac_max,safety,dt_min,dt_max,poll_number,time_direction) = expand_statics(statics);
    
    let tid = match thread_id_limit_check(n) {
        Some(tid) => tid,
        None => return,
    };
    for i in 0..(poll_number-1){
        let base_offset = 9*(poll_number)*tid+ i* 9;
        let x0 = *state_out.add(base_offset + 0);
        let y0 = *state_out.add(base_offset + 1);
        let z0 = *state_out.add(base_offset + 2);
        let vx0 = *state_out.add(base_offset + 3);
        let vy0 = *state_out.add(base_offset + 4);
        let vz0 = *state_out.add(base_offset + 5);
        let ax0 = *state_out.add(base_offset + 6);
        let ay0 = *state_out.add(base_offset + 7);
        let az0 = *state_out.add(base_offset + 8);
        let x1 = *state_out.add(base_offset + 9);
        let y1 = *state_out.add(base_offset + 10);
        let z1 = *state_out.add(base_offset + 11);
        let vx1 = *state_out.add(base_offset + 12);
        let vy1 = *state_out.add(base_offset + 13);
        let vz1 = *state_out.add(base_offset + 14);
        let ax1 = *state_out.add(base_offset + 15);
        let ay1 = *state_out.add(base_offset + 16);
        let az1 = *state_out.add(base_offset + 17);
        let out_offset = 18*(poll_number-1)*tid+i*18;
        let (ax,bx,cx,dx,ex,fx) = calc_coeff(x0,vx0,ax0,x1,vx1,ax1);
        let (ay,by,cy,dy,ey,fy) = calc_coeff(y0,vy0,ay0,y1,vy1,ay1);
        let (az,bz,cz,dz,ez,fz) = calc_coeff(z0,vz0,az0,z1,vz1,az1);
        *coeff.add(out_offset + 0) = ax;
        *coeff.add(out_offset + 1) = bx;
        *coeff.add(out_offset + 2) = cx;
        *coeff.add(out_offset + 3) = dx;
        *coeff.add(out_offset + 4) = ex;
        *coeff.add(out_offset + 5) = fx;
        *coeff.add(out_offset + 6) = ay;
        *coeff.add(out_offset + 7) = by;
        *coeff.add(out_offset + 8) = cy;
        *coeff.add(out_offset + 9) = dy;
        *coeff.add(out_offset + 10) = ey;
        *coeff.add(out_offset + 11) = fy;
        *coeff.add(out_offset + 12) = az;
        *coeff.add(out_offset + 13) = bz;
        *coeff.add(out_offset + 14) = cz;
        *coeff.add(out_offset + 15) = dz;
        *coeff.add(out_offset + 16) = ez;
        *coeff.add(out_offset + 17) = fz;
    }

}
