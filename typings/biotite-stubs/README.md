This partial typing overlay is based on the stubs shipped with Biotite 1.7.1
(BSD 3-Clause, see LICENSE.rst). Runtime Biotite is not patched.

`structure/atoms.pyi` corrects the dynamic atom annotation interface (`__getattr__`,
`__setattr__`, `_annot`) and permits sequences for annotations and index selection,
as supported by the installed runtime. The other stubs preserve Biotite's public
re-exports when Pyright overlays this package. `py.typed` marks it as partial;
modules not included here continue to use the installed package's typing.

Recheck this overlay when changing the Biotite version. Keep functional tests for
annotation assignment, integer/vector indexing, bonds, and CIF round trips.
