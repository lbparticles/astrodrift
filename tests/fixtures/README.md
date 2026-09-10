# Test fixtures

The `.fixture` files under this directory are numeric orbit dumps generated
by running native galpy v1.11.2 (`dop853_c` / `dopr54_c` methods over
`KeplerPotential`; the exact inputs are in the provenance header of each
file). They exist as reference expectations for the integrator ports in this
repository and change only when regenerated against a newer galpy.

galpy is Copyright (c) 2010, Jo Bovy, licensed under the BSD 3-Clause
License. These generated outputs are distributed under the same BSD 3-Clause
terms for conservatism. See `docs/licensing.md` for the project licensing
model.
