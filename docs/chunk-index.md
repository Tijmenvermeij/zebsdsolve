# Pattern chunk lookup

HDF5 previews no longer walk the entire pattern chunk tree. Until the full
lookup is ready, reads descend only the version-1 B-tree paths needed for the
requested patterns. Written chunks and verified missing chunks are cached
separately from unknown entries. Compressed chunks, split image axes and
partial edge chunks use the same decoding path before and after preparation.

The browser starts full lookup preparation after the first successful pattern
preview. It runs between foreground operations, through the same serialized
engine queue: normally at most eight nodes and 64 KiB in one file-read batch.
An oversized node is reread on a later turn; its explicit byte budget doubles
up to 8 MiB if necessary. A foreground request can wait for the current batch,
whose actual latency depends on storage, but never waits for the whole lookup.
Later previews and indexing automatically use the complete lookup. A partly
populated table is never reported complete.

`ze.indexPreparation` reports `ready`, `failed`, `visited`, `pending`, and
cumulative pattern-file `reads`/`bytes`. `ze.chunkIndexWarmup.state` reports
`idle`, `pending`, `ready` or `failed`; a failure also emits `chunkIndexError`
and stops automatic retries. `ze.pauseIndexPreparation()` cancels future
background turns; another successful preview can resume them. It does not
cancel indexing. `ze.cancel()` continues to cancel computational work only.

Opening or replacing a scan invalidates pending background work. Await
`ze.closeScan()` and `ze.dispose()`; both drain in-flight file reads before
freeing their data. Call `dispose()` rather than the low-level WASM destructor.
After disposal, operations reject and cancellation is a no-op. A lost GPU
stops scheduling; completed results remain available through existing exports.

Python previews use the same targeted native reader. Python does not start a
background thread against the shared solver. Prepare the full lookup explicitly:

```python
solver.open("scan.h5oina")
preview = solver.pattern(0, processed=False, full=True)
solver.prepare_index()  # optional synchronous full preparation
# Or interleave bounded work with application activity:
while not solver.prepare_index_step(max_nodes=8, max_bytes=65536):
    update_application()
print(solver.index_preparation)
```

A too-small explicit byte budget raises `ChunkIndexBudgetTooSmall` without
invalidating targeted reads; retry with a larger budget. The full Python helper
allows a single node up to 8 MiB. Structural corruption or IO failure marks the
index failed and is reported, rather than treating unknown chunks as empty.
Contiguous, compact and single-chunk datasets need no tree preparation.

Focused checks: `zig test src/io/hdf5.zig`, `tests/test_chunk_index.py`, and
`node web/tests/chunk-index-scheduler.mjs`. The Python fixture oracle uses
h5py for rank-3/rank-4 compressed, sparse, split-axis and edge chunks. No
orientation conventions or crystallographic calculations change.
