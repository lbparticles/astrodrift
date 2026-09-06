use core::slice;
use std::sync::Arc;

use numpy::PyReadonlyArrayDyn;
use shared::{INPUT_LENGTH, INPUT_STATE_DIM, Index, MAX_STATES, Real};

#[derive(Debug, Clone)]
pub struct InputState {
    pub num_particles: Index,
    pub data: Vec<Real>,
}

#[derive(Debug, Clone)]
pub struct InputFrame(pub [Option<Arc<InputState>>; MAX_STATES]);

#[derive(Debug, Clone)]
pub struct OutputState {
    pub num_times: Index,
    pub num_particles: Index,
    // Flattened time-major (time, particle, phase-space component) data.
    pub data: Vec<Real>,
}

#[derive(Debug, Clone)]
pub struct OutputFrame(pub [Option<OutputState>; MAX_STATES]);

impl<'a> IntoIterator for &'a InputFrame {
    type Item = &'a Option<Arc<InputState>>;
    type IntoIter = slice::Iter<'a, Option<Arc<InputState>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl InputState {
    pub fn new_zeroed() -> Self {
        InputState {
            num_particles: 0,
            data: vec![0.0; INPUT_LENGTH],
        }
    }

    pub fn from_validated_py_array(
        istate: &PyReadonlyArrayDyn<Real>,
        num_particles: Index,
    ) -> Self {
        let mut data = vec![0.0; INPUT_LENGTH];
        let istate_array = istate.as_array();
        debug_assert_eq!(istate_array.len(), num_particles * INPUT_STATE_DIM);
        debug_assert!(istate_array.len() <= INPUT_LENGTH);

        for (output, input) in data.iter_mut().zip(istate_array.iter()) {
            *output = *input;
        }

        InputState {
            num_particles,
            data,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[Real] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Real] {
        &mut self.data
    }
}

impl OutputState {
    pub fn new_zeroed(num_times: Index, num_particles: Index) -> Self {
        OutputState {
            num_times,
            num_particles,
            data: vec![0.0; num_times * num_particles * INPUT_STATE_DIM],
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[Real] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Real] {
        &mut self.data
    }
}
