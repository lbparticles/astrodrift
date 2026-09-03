use core::slice;

use numpy::PyReadonlyArrayDyn;
use shared::{Index, Real, INPUT_LENGTH, INPUT_STATE_DIM, MAX_STATES, OUTPUT_LENGTH};

#[derive(Debug,Clone)]
pub struct InputState{
    pub num_particles: Index,
    pub data: Vec<Real>
}

#[derive(Debug,Clone)]
pub struct InputFrame(pub [Option<InputState>; MAX_STATES]);

#[derive(Debug,Clone)]
pub struct OutputState{
    pub data: Vec<Real>
}

#[derive(Debug,Clone)]
pub struct OutputFrame(pub [Option<OutputState>; MAX_STATES]);

impl InputFrame {
    pub fn iter(&self) -> slice::Iter<'_, Option<InputState>> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a InputFrame {
    type Item = &'a Option<InputState>;
    type IntoIter = slice::Iter<'a, Option<InputState>>;

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

    pub fn from_py_array(istate: &PyReadonlyArrayDyn<Real>) -> Self {
        let mut data = vec![0.0; INPUT_LENGTH];

        let istate_array = istate.as_array();
        let num_particles = istate_array.len() / INPUT_STATE_DIM;

        for (i, v) in istate_array.iter().copied().enumerate() {
            if i >= INPUT_LENGTH {
                break;
            }
            data[i] = v;
        }

        InputState { num_particles, data }
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
    pub fn new_zeroed() -> Self {
        OutputState {
            data: vec![0.0; OUTPUT_LENGTH],
        }
    }

    pub fn from_py_array(istate: &PyReadonlyArrayDyn<Real>) -> Self {
        let mut data = vec![0.0; OUTPUT_LENGTH];

        for (i, v) in istate.as_array().iter().copied().enumerate() {
            if i >= OUTPUT_LENGTH {
                break;
            }
            data[i] = v;
        }

        OutputState { data }
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

