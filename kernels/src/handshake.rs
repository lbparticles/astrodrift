#[inline(always)]
pub unsafe fn load_state(ptr: *const f64, base_offset: usize) -> [f64; 6] {
    [
        *ptr.add(base_offset + 0),
        *ptr.add(base_offset + 1),
        *ptr.add(base_offset + 2),
        *ptr.add(base_offset + 3),
        *ptr.add(base_offset + 4),
        *ptr.add(base_offset + 5),
    ]
}

#[inline(always)]
pub unsafe fn store_state(ptr: *mut f64, base_offset: usize, s: [f64; 6]) {
    *ptr.add(base_offset + 0) = s[0];
    *ptr.add(base_offset + 1) = s[1];
    *ptr.add(base_offset + 2) = s[2];
    *ptr.add(base_offset + 3) = s[3];
    *ptr.add(base_offset + 4) = s[4];
    *ptr.add(base_offset + 5) = s[5];
}
