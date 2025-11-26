use numpy::PyReadonlyArrayDyn;
use shared::{INPUT_LENGTH,OUTPUT_LENGTH,MAX_STATES,Real};

#[derive(Debug,Clone)]
pub struct InputState{
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

impl InputState {
    pub fn new_zeroed() -> Self {
        InputState {
            data: vec![0.0; INPUT_LENGTH],
        }
    }

    pub fn from_py_array(istate: &PyReadonlyArrayDyn<Real>) -> Self {
        let mut data = vec![0.0; INPUT_LENGTH];

        for (i, v) in istate.as_array().iter().copied().enumerate() {
            if i >= INPUT_LENGTH {
                break;
            }
            data[i] = v;
        }

        InputState { data }
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

