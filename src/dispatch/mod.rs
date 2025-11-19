pub mod cpu;
pub mod gpu;
pub mod gpu_from_cpu;

pub use gpu::gpu_dispatch;
pub use gpu_from_cpu::gpu_dispatch2;
