# Resources and measurement scope

## Model files and working memory

Model bundle 3.12.0 contains the following ONNX files. MB means decimal bytes;
GiB means 1,073,741,824 bytes. File sizes are not RAM requirements.

| File | Bytes | Rounded file size |
| --- | ---: | ---: |
| `armiger.int8.onnx` | 17,606,362 | 17.6 MB |
| `armiger_ultra.fp32.onnx` | 33,260,312 | 33.3 MB |

Sizes and checksums are recorded in the [pinned model manifest](../src/armiger/model_manifest.json).
The INT8 file's name and the small neural network do not establish a tiny
whole-process memory footprint.

There are different execution paths:

- **Standalone ONNX:** the `ArmigerStandaloneONNX` interface performs neural
  inference and its associated output processing. It omits the Ultra catalogue
  and swarm components. Its current-release working RAM has not been established
  by a resource benchmark in this repository.
- **GitHub Ultra package:** `ArmigerUltraPipeline` adds the packaged catalogue,
  calibration, metadata rules and optional swarm components. Selecting INT8
  retains those components; it does not select the standalone interface.
- **Complete released SDK:** directly loading the fuller Hugging Face SDK
  snapshot with its additional authority indexes and policy modules is a separate
  integration. The GitHub package's eight-file model manifest is not a manifest
  of that entire SDK. Equal neural weight versions do not establish full policy
  or output parity between these paths.

A Linux x86-64 CPU service using the complete R13.11 SDK was observed on
2026-11-02 with one persistent worker, both FP32/INT8 sessions used, and an
integration adapter that clears request-only caches. During a 628.8-second
observation with 4,301 natural classification attempts, whole-container memory
was 2,231,607,296–2,250,448,896 bytes (about 2.1 GiB). Its configured limit was
4 GiB. This is one deployment observation, not a minimum requirement, a memory
benchmark of the standalone/GitHub Ultra interfaces, or a guarantee for other
batches, versions or integrations. The observation records resources and
completion, not classification accuracy. Measure the actual path you deploy.

ONNX execution buffers, loaded sessions, Python objects, reference data and
optional libraries all contribute to working memory. Allow additional resources
for Bitmagnet, Rats Search, their databases and the host OS.

## Timing and cost

No current-release end-to-end CPU/GPU latency or throughput guarantee is made.
The `0.21 ms/item` GPU headline lacks enough published hardware, batch, model
and timing scope to reproduce it and is not endorsed here. Neural-kernel timing, full policy processing, connector work
and HTTP round trips are different measurements.

The committed M4 Pro timing receipt is a historical measurement on the model
hashes it names. Those hashes differ from model bundle 3.12.0; relabelling that
timing as a new release does not remeasure it. Historical API prices and local
cost estimates are not current price quotes or measured total ownership costs.
See [evaluation status](evaluation.md).

## Prediction boundaries

Classification reads names, file lists and supplied counters. Audio/subtitle,
junk, adult-content and honeypot outputs are metadata estimates or policy flags.
They do not inspect media payloads, scan executable contents, certify English
speech, establish lawful content or guarantee a safe or available download.
An optional connector may separately inspect an already downloaded file with
`ffprobe`; that is a different operation with its own evidence requirements.
Cryptographic checks establish the origin/integrity of update data, not model
accuracy or payload safety. Sharing is optional; see the documented
[privacy limits](distributed-updates.md).
