#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdjacencyMatrix(pub u128);

impl AdjacencyMatrix {
    const N: usize = 11;
    const VALID_MASK: u128 = (1u128 << (Self::N * Self::N)) - 1;

    #[inline]
    const fn idx(r: usize, c: usize) -> usize {
        r * Self::N + c
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> bool {
        debug_assert!(r < Self::N && c < Self::N);
        ((self.0 >> Self::idx(r, c)) & 1) != 0
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: bool) {
        debug_assert!(r < Self::N && c < Self::N);
        let bit = 1u128 << Self::idx(r, c);
        if val {
            self.0 |= bit;
        } else {
            self.0 &= !bit;
        }
    }

    #[inline]
    pub fn row_bits(&self, r: usize) -> u128 {
        debug_assert!(r < Self::N);
        (self.0 >> (r * Self::N)) & ((1u128 << Self::N) - 1)
    }

    // Build a bitmask that has the j-th bit set for each row i where A[i, j] is 1.
    // That is, the column as a vertical bitset aligned so that row i maps to bit i.
    fn column_rows_mask(&self, col: usize) -> u16 {
        // We only need 11 bits to represent which rows have a 1 in this column.
        let mut mask: u16 = 0;
        let mut r = 0;
        // Unrolled-ish loop for clarity/perf; simple loop is fine too.
        while r < Self::N {
            let bit = ((self.0 >> Self::idx(r, col)) & 1) as u16;
            mask |= bit << r;
            r += 1;
        }
        mask
    }

    // Square over boolean semiring: C = A ⊙ A (OR/AND).
    // For each (i, j): c[i,j] = any k s.t. A[i,k] & A[k,j].
    pub fn mul_self(&self) -> AdjacencyMatrix {
        let n = Self::N;

        // Precompute row bitsets: row[i] is 11-bit word for row i
        let mut row_bits: [u16; 11] = [0; 11];
        for i in 0..n {
            row_bits[i] = (self.row_bits(i) as u16) & 0x7FF;
        }

        // Precompute column-as-rows masks: col_rows[j] has bit k set iff A[k, j] == 1
        let mut col_rows: [u16; 11] = [0; 11];
        for j in 0..n {
            col_rows[j] = self.column_rows_mask(j); // 11-bit
        }

        // For each pair (i, j), c[i,j] = (row_bits[i] & col_rows[j]) != 0
        let mut out: u128 = 0;
        for i in 0..n {
            for j in 0..n {
                let has = (row_bits[i] & col_rows[j]) != 0;
                if has {
                    out |= 1u128 << Self::idx(i, j);
                }
            }
        }

        AdjacencyMatrix(out & Self::VALID_MASK)
    }

    // Trace as a boolean: true if any diagonal entry is 1
    pub fn trace_any(&self) -> bool {
        for i in 0..Self::N {
            if self.get(i, i) {
                return true;
            }
        }
        false
    }

    // Trace as a count of 1s on the diagonal (0..=11)
    pub fn trace_count(&self) -> u32 {
        let mut cnt = 0u32;
        for i in 0..Self::N {
            if self.get(i, i) {
                cnt += 1;
            }
        }
        cnt
    }
}


#[cfg(test)]
mod tests {
    use super::AdjacencyMatrix as AM;

    #[test]
    fn basic_set_get() {
        let mut a = AM(0);
        a.set(3, 7, true);
        assert!(a.get(3, 7));
        assert!(!a.get(3, 6));
    }

    #[test]
    fn identity_square_is_identity() {
        let mut id = AM(0);
        for i in 0..11 {
            id.set(i, i, true);
        }
        let sq = id.mul_self();
        for i in 0..11 {
            for j in 0..11 {
                assert_eq!(sq.get(i, j), i == j);
            }
        }
        assert!(sq.trace_any());
        assert_eq!(sq.trace_count(), 11);
    }

    #[test]
    fn path_of_length_two() {
        // 0 -> 1, 1 -> 2, so A^2 has 0 -> 2
        let mut a = AM(0);
        a.set(0, 1, true);
        a.set(1, 2, true);
        let a2 = a.mul_self();
        assert!(a2.get(0, 2));
        assert!(!a2.get(0, 1));
        assert!(!a2.get(1, 2));
    }
}
