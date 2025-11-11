[repr(C)]
enum PotentialEnum {
    potential_id,
    params,
    lut_id,
}


[repr(C)]
enum PotentialList {
    bovy14,
    plummer,
    mn,
    nfw,
    sphcutoff,
}
