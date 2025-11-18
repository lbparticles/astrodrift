use crate::handshake::{load_state,store_state};
use cuda_std::{kernel, thread};
use libm::{floor, pow, sqrt};
use crate::recipes::consume_recipe;
use crate::{expand_statics,thread_id_limit_check};
use shared::{Potential,PotentialRecipe,PotentialEnum,PotentialNames,Config,KeplerPotential};
#[kernel]
pub unsafe fn post_kernel(
    state_out: *mut f64, //pointer
    statics : Config,
    recipe: [PotentialRecipe;10],
    supertable: *mut f64,
) {
    let (n,steps_cap,t_end,atol,rtol,fac_min,fac_max,safety,dt_min,dt_max,poll_number,time_direction) = expand_statics(statics);
    
    let tid = match thread_id_limit_check(n) {
        Some(tid) => tid,
        None => return,
    };


    for i in 0..poll_number{
        let t = t_end * (i as f64) / (poll_number as f64 - 1.);
        let offset = 9*poll_number*tid+ i* 9;
        let x0 = load_state(state_out, offset);
        let mut potential:[PotentialEnum;10] = [PotentialEnum::KeplerPotential(KeplerPotential{amp:0.0});10];

        for (i,r) in recipe.into_iter().enumerate(){
            potential[i] = consume_recipe(r,supertable);
        }
        let pot = potential[0];
        let (ax,ay,az) = pot.force(t,x0[0],x0[1],x0[2]);
        *state_out.add(offset + 6) = ax;
        *state_out.add(offset + 7) = ay;
        *state_out.add(offset + 8) = az;
    }

}
