use shared::Potential;


pub struct Container<T> {
    potential: Potential<T>,
    istate: mut *f64,
    dependencies: Vec<*mut Container<T>>,
}

pub unsafe fn tree_constructor<T>(mut *Container<T>)-> Vec<*mut Container<T>> {
    // ParticleGroups depend on Backgrounds
    for &p in &particles {
        (*p).dependencies.extend_from_slice(&backgrounds);
    }

    // TestGroups depend on both Particles and Backgrounds
    for &t in &tests {
        (*t).dependencies.extend_from_slice(&particles);
        (*t).dependencies.extend_from_slice(&backgrounds);
    }

    // Return all nodes in one vector for traversal
    [backgrounds, particles, tests].concat()
}
