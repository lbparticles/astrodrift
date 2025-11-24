
pub struct InputState(pub Box<[Real; INPUT_LENGTH]>);
pub struct InputFrame(pub Box<[Option<InputState>; MAX_STATES]>);

pub type OutputState(pub Box<[Real; OUTPUT_LENGTH]>);
pub type OutputFrame(pub Box<[Option<OutputState>; MAX_STATES]>);

