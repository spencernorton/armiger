"""Classify sample metadata with a previously downloaded Armiger Ultra model."""
from armiger import ArmigerUltraPipeline

pipeline = ArmigerUltraPipeline(preload_catalog=True, rsi_mode="pull_only", opt_in_sharing=False)
result = pipeline.classify({
    "name": "Tears.of.Steel.2012.1080p.BluRay.x264-OPEN",
    "size": 2_400_000_000,
    "files": [{"path": "Tears.of.Steel.2012.mkv", "size": 2_400_000_000}],
})
print(f"Content type: {result['category']}")
