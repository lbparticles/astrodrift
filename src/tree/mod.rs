use core::fmt;

use crate::{
    interface::Container,
    state::{InputFrame, InputState},
};
use shared::{MAX_MODEL_COMPONENTS, MAX_RECIPES, MAX_STATES, Model, Recipe};

pub struct IntegrationPlan {
    pub model: Model,
    pub input_frame: InputFrame,
    // Dispatch outputs are stage ordered; retain stable identities so the
    // interface can restore the requested container order.
    pub container_identity_by_stage: [Option<u64>; MAX_STATES],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AdjacencyMatrix(pub u128);

impl AdjacencyMatrix {
    const N: usize = MAX_MODEL_COMPONENTS;
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

    fn topological_order(&self, present: &[bool; Self::N]) -> Option<Vec<usize>> {
        let mut indegree = [0; Self::N];
        for source in 0..Self::N {
            for dependent in 0..Self::N {
                if present[source] && present[dependent] && self.get(source, dependent) {
                    indegree[dependent] += 1;
                }
            }
        }

        let count = present.iter().filter(|&&is_present| is_present).count();
        let mut emitted = [false; Self::N];
        let mut order = Vec::with_capacity(count);
        while order.len() < count {
            let next = (0..Self::N)
                .find(|&node| present[node] && !emitted[node] && indegree[node] == 0)?;
            emitted[next] = true;
            order.push(next);
            for dependent in 0..Self::N {
                if present[dependent] && self.get(next, dependent) {
                    indegree[dependent] -= 1;
                }
            }
        }
        Some(order)
    }

    pub fn build(&self, containers: [Option<Container>; MAX_STATES]) -> Option<IntegrationPlan> {
        let present = std::array::from_fn(|index| containers[index].is_some());
        let order = self.topological_order(&present)?;
        let mut meal_by_stage: [Option<[Option<Recipe>; MAX_RECIPES]>; MAX_MODEL_COMPONENTS] =
            std::array::from_fn(|_| None);
        let mut istates_by_stage: [Option<InputState>; MAX_STATES] = std::array::from_fn(|_| None);
        let mut container_identity_by_stage = [None; MAX_STATES];
        let mut stage = 0;

        for container_label in order {
            let container = containers[container_label].as_ref()?;
            let Some(input_state) = container.state.as_ref() else {
                continue;
            };

            let mut recipes: [Option<Recipe>; MAX_RECIPES] = std::array::from_fn(|_| None);
            for (source_label, recipe) in recipes.iter_mut().enumerate() {
                if self.get(source_label, container_label)
                    && let Some(source_recipe) = containers[source_label]
                        .as_ref()
                        .and_then(|source| source.recipe.as_ref())
                {
                    *recipe = Some(source_recipe.inner);
                }
            }

            meal_by_stage[stage] = Some(recipes);
            istates_by_stage[stage] = Some(input_state.clone());
            container_identity_by_stage[stage] = Some(container.identity);
            stage += 1;
        }

        Some(IntegrationPlan {
            model: meal_by_stage.into(),
            input_frame: InputFrame(istates_by_stage),
            container_identity_by_stage,
        })
    }
}

impl fmt::Debug for AdjacencyMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header with the raw value (trim to used 121 bits)
        let raw = self.0 & Self::VALID_MASK;
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
    fn topological_order_places_dependencies_first() {
        let mut a = AM(0);
        a.set(2, 0, true);
        a.set(0, 1, true);

        let mut present = [false; 11];
        present[..3].fill(true);

        assert_eq!(a.topological_order(&present), Some(vec![2, 0, 1]));
    }

    #[test]
    fn topological_order_rejects_cycles() {
        let mut a = AM(0);
        a.set(0, 1, true);
        a.set(1, 0, true);

        let mut present = [false; 11];
        present[..2].fill(true);

        assert_eq!(a.topological_order(&present), None);
    }
}
