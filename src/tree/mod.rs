use core::fmt;

use crate::interface::Container;

#[derive(Clone, Copy, PartialEq, Eq)]
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
    pub fn last_true_column_power(&self, cap: usize) 
    -> [u8; 11] 
    {
        // assert!(cap > 0 && cap <= 255, "cap must be in 1..=255");
        let n = Self::N;
        let mut last: [u8; 11] = [0; 11];

        let mut power = *self; // A^1
        for p in 1..=cap {
            // For each column j, check if any entry in column j is true in A^p
            for j in 0..n {
                // build a quick "any bit in column j" test
                // We can scan rows, since n=11 this is cheap.
                let mut any = false;
                let mut r = 0;
                while r < n {
                    if power.get(r, j) {
                        any = true;
                        break;
                    }
                    r += 1;
                }
                if any {
                    last[j] = p as u8;
                }
            }

            if p == cap {
                break;
            }
            power = power.mul_self(); // A^(p+1)
        }

        last
    }

    // Efficient diagonal-nonzero test
    #[inline]
    fn has_nonzero_trace(&self) -> bool {
        // Diagonal bits are at indices i*11 + i, for i=0..10
        // Just scan them; n=11 so this is cheap.
        for i in 0..Self::N {
            if self.get(i, i) {
                return true;
            }
        }
        false
    }

    // General boolean matrix multiply: C = A · B over OR/AND
    pub fn mul_bool(&self, rhs: &AdjacencyMatrix) -> AdjacencyMatrix {
        let n = Self::N;

        // Row i of A as 11-bit masks
        let mut a_rows: [u16; 11] = [0; 11];
        for i in 0..n {
            a_rows[i] = (self.row_bits(i) as u16) & 0x7FF;
        }

        // For B, precompute column-as-rows masks: for each column j, bit k is B[k, j]
        let mut b_col_rows: [u16; 11] = [0; 11];
        for j in 0..n {
            b_col_rows[j] = rhs.column_rows_mask(j);
        }

        let mut out: u128 = 0;
        for i in 0..n {
            for j in 0..n {
                if (a_rows[i] & b_col_rows[j]) != 0 {
                    out |= 1u128 << Self::idx(i, j);
                }
            }
        }
        AdjacencyMatrix(out & Self::VALID_MASK)
    }

    // Final acyclicity method using A, A^2, ..., A^N
    pub fn is_acyclic_by_traces(&self) -> bool {
        let n = Self::N;

        // A^1
        if self.has_nonzero_trace() {
            return false;
        }

        // Iteratively multiply by A to get A^p for p = 2..=N
        let mut power = *self; // A^1
        for _p in 2..=n {
            power = power.mul_bool(self); // A^(p) = A^(p-1) · A
            if power.has_nonzero_trace() {
                return false;
            }
        }
        true
    }
    pub fn build(
        &self,
        containers: Box<[Option<Container>; 11]>,
    ) -> (shared::Meal, shared::InputStates) {
        let last = self.last_true_column_power(11);

        let mut with_deps: Vec<usize> = (0..11).filter(|&v| last[v] >= 1).collect();
        with_deps.sort_by_key(|&v| (last[v], v));

        let mut zeros: Vec<usize> = (0..11).filter(|&v| last[v] == 0).collect();
        zeros.sort(); // stable deterministic tail

        let mut order: [usize; 11] = [0; 11];
        let split = with_deps.len();
        for (i, v) in with_deps.into_iter().enumerate() {
            order[i] = v;
        }
        for (i, v) in zeros.into_iter().enumerate() {
            order[split + i] = v;
        }

        let mut meal_by_stage: [Option<[Option<shared::Recipe>; 11]>; 11] =
            std::array::from_fn(|_| None);
        let mut istates_by_stage: [Option<shared::InputState>; 11] =
            std::array::from_fn(|_| None);
        let mut rank: [usize; 11] = [0; 11];
        for (s, &v) in order.iter().enumerate() {
            rank[v] = s;
        }

        for v in 0..11 {
            let has_incoming = (0..11).any(|k| self.get(k, v));
            if !(last[v] >= 1 && has_incoming) {
                continue;
            }

            let s = rank[v];

            // Input state from v
            if let Some(c) = containers[v].as_ref() {
                istates_by_stage[s] = c.state.clone();
            }

            // Build the per-upstream array for this stage
            let mut arr_k: [Option<shared::Recipe>; 11] = std::array::from_fn(|_| None);
            for k in 0..11 {
                if self.get(k, v) {
                    if let Some(src) = containers[k].as_ref() {
                        if let Some(py_recipe) = src.recipe.as_ref() {
                            arr_k[k] = Some(py_recipe.inner.clone());
                        }
                    }
                }
            }
            if arr_k.iter().any(|x| x.is_some()) {
                meal_by_stage[s] = Some(arr_k);
            }
        }


        // Box to match return types
        let meal: shared::Meal = Box::new(meal_by_stage).into();
        let input_states: shared::InputStates = Box::new(istates_by_stage);

        (meal, input_states)
    }
    // pub fn build(
    //     &self,
    //     containers: Box<[Option<Container>; 11]>,
    // ) -> (
    //     shared::Meal,
    //     shared::InputStates,
    // )
    // {
    //     // // 1) Compute last power for ordering (cap = 11)
    //     let last = self.last_true_column_power(11);

    //     // 2) Build order of container indices by ascending last power, then by index
    //     let mut order: [usize; 11] = [0; 11];
    //     for i in 0..11 {
    //         order[i] = i;
    //     }
    //     order.sort_by_key(|&v| (last[v], v));

    //     // 3) Rank map: vertex -> stage index 0..10
    //     let mut rank: [usize; 11] = [0; 11];
    //     for (s, &v) in order.iter().enumerate() {
    //         rank[v] = s;
    //     }

    //     // // 4) Initialize outputs
    //     let mut deps_by_stage: [[Option<Container>; 11]; 11] =
    //         std::array::from_fn(|_| std::array::from_fn(|_| None));
    //     let mut istates_by_stage: [Option<shared::InputState>; 11] =
    //         std::array::from_fn(|_| None);

    //     // // 5) Fill per-vertex stage entries
    //     for v in 0..11 {
    //         let s = rank[v];
    //         istates_by_stage[s] = containers[v]
    //             .as_ref()
    //             .and_then(|c| c.state); // adjust to your Container API
    //         // println!("{:?}",istates_by_stage[0]);

    //         // 5b) First direct dependency (k -> v) that also exists (Some)
    //         let mut placed = false;
    //         for k in 0..11 {
    //             if self.get(k, v) {
    //                 if let Some(dep) = containers[k].as_ref() {
    //                     deps_by_stage[s][v] = Some(dep.clone());
    //                     placed = true;
    //                     break;
    //                 }
    //             }
    //         }
    //         if !placed {
    //             // leave as None
    //         }
    //     }

    //     // (deps_by_stage, istates_by_stage)
    //     return (
    //         Box::new(
    //         [None; shared::MAX_COURSES]),
    //         Box::new(
    //         [None; shared::MAX_STATES],
    //     ));
    // }
}

impl fmt::Debug for AdjacencyMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header with the raw value (trim to used 121 bits)
        let raw = self.0 & ((1u128 << (Self::N * Self::N)) - 1);
        // Print 11 rows, each with 11 columns as 0/1
        for r in 0..Self::N {
            for c in 0..Self::N {
                let bit = ((raw >> Self::idx(r, c)) & 1) as u8;
                // '0' + bit
                let ch = (b'0' + bit) as char;
                write!(f, "{ch}")?;
            }
            if r + 1 < Self::N {
                writeln!(f)?;
            }
        }
        Ok(())
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
