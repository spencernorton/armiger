# Runtime and release validation

The required GitHub check `test` includes package/model tests on Python 3.10
and 3.12, installed-wheel verification, release/quickstart contracts, and native
container acceptance on amd64 and arm64. A skipped, cancelled or failed required
job fails the aggregate. Container builds and acceptance run on pull requests.

## What the container checks prove

`ci/runtime_acceptance.py` derives commands, environment and services from the
three documented Compose files. It substitutes the newly built candidate,
uses unique project names, fresh volumes and random loopback ports, disables
automatic model updates/sharing and public crawling, and retains logs before
removing only that test project's containers and volumes. It checks that the
running Armiger image matches the candidate and uses its default UID10001.

- Standalone: start the normal command, wait for HTTP health, classify licensed
  Blender movie metadata with the real bundled Ultra model, and require movie.
- Bitmagnet: start stock Bitmagnet v0.10.0 and PostgreSQL, run stock migrations,
  require an empty database, then insert synthetic metadata. The default
  connector command must process 350 records over batch100 polls, delete two
  disguised-executable fixtures in the first page without skipping shifted
  rows, and process 50 later arrivals after exhausting the original wave.
  Readback through the real GraphQL API requires every eligible record tagged.
  Three legacy viability tags must migrate, and all six supported tier inputs
  must serialize to the exact readable canonical tags and persist upstream.
  The Torznab proxy must retrieve real stock capabilities and admit actual
  search results for the licensed fixtures.
- Rats: both native Armiger images start the documented connector entrypoint
  and perform real-model HTTP classification. On amd64, CI also builds/starts
  the supported x86 Rats backend with Manticore, reaches its real recent-search
  REST endpoint, applies a classification through the real HTTP/SQL path and
  reads the changed category back from Manticore. A colliding infohash must
  preserve the other row's metadata. On arm64 the connector HTTP entrypoint
  runs alone; the bundled Rats
  backend is x86-only and full-stack acceptance is required on the amd64 job.

The fixtures contain metadata only; no copyrighted media or public torrents
are downloaded. They qualify packaging and integration, not broad model
accuracy or every possible upstream deployment. Fast model, protocol and unit
tests remain required alongside these checks.

## Run locally

CI tooling requires Python3.11+, Docker, Buildx and Compose. The installed
Armiger library continues to support Python3.10+.

```bash
python3 -m unittest discover -s ci/tests -v
docker build -f examples/Dockerfile -t armiger-ci:local .
python3 -m ci.runtime_acceptance --image armiger-ci:local --evidence out/runtime
```

On x86, build `examples/rats-search/Dockerfile` as `rats-ci:local` and pass
`--rats-image rats-ci:local` for full backend acceptance. An ARM Mac can supply
the published x86 backend under Docker emulation while Armiger runs natively.

## Publication uses tested artifacts

On main, native acceptance jobs push only their successful local images to
run-specific candidate tags. `ci/candidates.py` verifies the pushed image's
uncompressed filesystem layers and runtime configuration against the tested
local image, and records its manifest/config digests, source and version.

After all required checks succeed, `ci/promote.py` requires all three native
receipts (Armiger amd64+arm64, Rats amd64), checks their identities again, and
creates release indexes from those exact native manifest digests. It never
rebuilds an image after acceptance. Source/version provenance must agree in every native image config; Docker
manifest lists and OCI indexes are both supported, and any index annotation
must agree with those configs. Published indexes, aliases and both OCI
Compose stacks are read back and checked against the receipts/documented
commands. Mutable latest/legacy aliases advance only after OCI readback. Logs
and JSON evidence are retained as Actions artifacts for14days;
publication proof is retained for30days. A partial or ambiguous publication
remains a failed workflow.

Existing software versions are immutable. An image-affecting source change
without a version bump fails `ci/release_contract.py`; it cannot become a green
"already released" skip. Changes confined to tests or other non-image docs can
validate an existing release without republishing. Failed registry reads are
errors, not evidence that a version is unpublished. Local setup image/OCI
versions must match the software version. External Connectors/HF guide image
references must exist and resolve to a release containing the3.4.1 startup fix.
Software version and model-bundle version remain separate.

An unchanged-version main run still verifies the existing release's platforms,
immutable alias and published stacks. It cannot turn an incomplete release
green by skipping publication. To recover a partial publication, rerun the
failed promotion job to reuse that run's original receipts. A full rebuild that
changes an already-released payload is rejected and needs a new version.

Connectors can invoke this workflow with their exact repository/commit inputs.
CI checks out a fixed Armiger revision, replaces the two pinned connector ADDs
with checksummed files from that candidate and runs the same native acceptance
suite. Consumer runs cannot publish Armiger releases.

## Completion

Before merging, inspect required checks on the current PR head and resolve
actionable findings. After merge, require successful main publication and read
the release proof before announcing the version. Add release notes describing
the user-visible fix and upgrade path. For a fixed issue, post a resolution
comment explaining the fix with a direct release-note link and verified upgrade
instructions before closing. Avoid auto-closing PR keywords when publication
and verification will occur after merge.
