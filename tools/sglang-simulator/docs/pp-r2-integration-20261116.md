# R2 source integration, 2026-11-16

This branch imports the runtime source shipped in the colleague's deployed R2 wheel:
`oss://kunlun-us-west/hisim/runtime/pp-lifecycle-r2-20261111/2f9f1f61d897/artifact.tar.gz`.
The upstream collection branch is still `aa73f1d7`; its two existing local commits
`11f3d50` and `94f3a0a` are retained. The artifact is the runtime source of record;
this is a source import, not a claim that a newer upstream Git commit exists.

The existing real collector, Prefill protocol boundaries, and native WORLD barrier
observations remain. Shared runtime modules come from the R2 wheel, including its
SGLang adapters needed by the original bootstrap; these are imported baseline code,
not new SGLang behavior or a change to the Prefill-only experiment scope.
The bootstrap selects collector or simulator using their existing environment flags.
Enabling both stops interpreter startup before either installs hooks. Vendored native
PP control files and their source manifest are included in built wheels.

The follow-up local commit retains the stage barrier timestamps and per-rank/padding
calibration already running in the personal test4 simulator. Predictors, runtime
configuration, local V6D isolation, and online deployments are not changed here.

Validation reuses the R2 CPU integration test and targeted request/native boundary
checks. CPU validation does not establish TPU timing accuracy or DRAM hit accuracy.
