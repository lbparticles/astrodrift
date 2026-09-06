import drift as dft
import numpy as np


def run_gpu_smoke() -> list[list[float]]:
    kp = dft.Potential.kepler(1.0)
    gmc_istate = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    iso_istate = np.array([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float64)
    gal = dft.bg_feature(kp)
    gmc = dft.part_group(kp, gmc_istate)
    iso = dft.test_group(iso_istate)
    sim = dft.Config(variant=dft.Variant("Compatible"))
    sim.dependency(gmc, gal)
    sim.dependency(iso, gmc, gal)
    return sim.run(gal, gmc, iso)


def test_gpu_extension_launches_embedded_kernel() -> None:
    result = run_gpu_smoke()
    assert isinstance(result, list)


if __name__ == "__main__":
    print(run_gpu_smoke())
