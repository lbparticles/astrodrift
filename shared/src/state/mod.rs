use crate::{INPUT_LENGTH,OUTPUT_LENGTH,MAX_STATES,Real};

#[derive(Debug,Clone)]
pub struct InputState(pub [Real; INPUT_LENGTH]);
#[derive(Debug,Clone)]
pub struct InputFrame(pub [Option<InputState>; MAX_STATES]);

#[derive(Debug,Clone)]
pub struct OutputState(pub [Real; OUTPUT_LENGTH]);
#[derive(Debug,Clone)]
pub struct OutputFrame(pub [Option<OutputState>; MAX_STATES]);

