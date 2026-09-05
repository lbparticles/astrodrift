use cuda_core::CudaContext;
use cuda_host::embedded::{ArtifactPayloadKind, artifact_bundles_from_binary_path};
use std::ffi::CStr;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::sync::Arc;

use super::super::GPUDispatchError;

const KERNEL_BUNDLE_NAME: &str = "kernels";

// FIXME(cuda-oxide): replace this local shared-object discovery once the generated
// loader can read a caller-supplied binary path or directly bound artifact bytes.
// Its current `load()` searches the Python executable rather than `drift_rs.so`.
#[inline(never)]
fn artifact_binary_path() -> Result<PathBuf, GPUDispatchError> {
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
    let address = (artifact_binary_path as *const ()).cast::<libc::c_void>();

    // SAFETY: `address` points to this function and `info` points to writable storage.
    let found = unsafe { libc::dladdr(address, info.as_mut_ptr()) };
    if found == 0 {
        return Err(GPUDispatchError::ArtifactBinaryNotFound);
    }

    // SAFETY: a successful `dladdr` call initialized `info`.
    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() {
        return Err(GPUDispatchError::ArtifactBinaryNotFound);
    }

    // SAFETY: `dladdr` returns a NUL-terminated filename owned by the dynamic loader.
    let filename = unsafe { CStr::from_ptr(info.dli_fname) };
    if filename.to_bytes().is_empty() {
        return Err(GPUDispatchError::ArtifactBinaryNotFound);
    }

    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(
        filename.to_bytes(),
    )))
}

pub(super) fn load_module(
    context: &Arc<CudaContext>,
) -> Result<kernels::oxide::LoadedModule, GPUDispatchError> {
    let path = artifact_binary_path()?;
    let bundles =
        artifact_bundles_from_binary_path(&path).map_err(cuda_host::EmbeddedModuleError::Core)?;
    let mut matching_bundles = bundles
        .into_iter()
        .filter(|bundle| bundle.name == KERNEL_BUNDLE_NAME);
    let Some(bundle) = matching_bundles.next() else {
        return Err(GPUDispatchError::ArtifactBundleCount {
            path,
            name: KERNEL_BUNDLE_NAME,
            count: 0,
        });
    };
    let duplicate_count = matching_bundles.count();
    if duplicate_count != 0 {
        return Err(GPUDispatchError::ArtifactBundleCount {
            path,
            name: KERNEL_BUNDLE_NAME,
            count: duplicate_count + 1,
        });
    }

    let mut cubins = bundle
        .payloads
        .iter()
        .filter(|payload| payload.kind == ArtifactPayloadKind::Cubin);
    let Some(cubin) = cubins.next() else {
        return Err(GPUDispatchError::ArtifactCubinCount {
            path,
            name: KERNEL_BUNDLE_NAME,
            count: 0,
        });
    };
    let duplicate_count = cubins.count();
    if duplicate_count != 0 {
        return Err(GPUDispatchError::ArtifactCubinCount {
            path,
            name: KERNEL_BUNDLE_NAME,
            count: duplicate_count + 1,
        });
    }

    let module = context.load_module_from_image(&cubin.bytes)?;
    // SAFETY: the selected bundle is embedded by the same `kernels` crate that
    // generated this typed host API, so its kernel ABI and contracts match.
    Ok(unsafe { kernels::oxide::from_module(module) }?)
}
