use crate::{INPUT_LENGTH,OUTPUT_LENGTH,MAX_STATES,Real};

#[derive(Debug,Clone)]
pub struct InputState(pub Box<[Real; INPUT_LENGTH]>);
#[derive(Debug,Clone)]
pub struct InputFrame(pub Box<[Option<InputState>; MAX_STATES]>);

#[derive(Debug,Clone)]
pub struct OutputState(pub Box<[Real; OUTPUT_LENGTH]>);
#[derive(Debug,Clone)]
pub struct OutputFrame(pub Box<[Option<OutputState>; MAX_STATES]>);

