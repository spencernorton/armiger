"""End-to-end tests for DH-Waypoint, Bitmagnet, and *Arr / qBittorrent integration adapters."""
from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "examples" / "dh-waypoint"))
sys.path.insert(0, str(ROOT_DIR / "examples" / "bitmagnet"))
sys.path.insert(0, str(ROOT_DIR / "examples" / "arr-stack"))

from armiger_arr_webhook_receiver import ArmigerArrWebhookService
from armiger_dht_rsi import DHTRSIMode
from armiger_torznab_proxy import ArmigerTorznabFilter
from armiger_ultra_pipeline import ArmigerUltraPipeline
from armiger_waypoint_processor import ArmigerWaypointAdapter
from bitmagnet_armiger_bridge import BitmagnetArmigerBridge


@pytest.fixture(scope="module")
def shared_pipeline() -> ArmigerUltraPipeline:
    return ArmigerUltraPipeline.from_directory(
        ROOT_DIR,
        variant="int8",
        rsi_mode=DHTRSIMode.ANONYMOUS_CONTRIBUTE,
        preload_catalog=False,
    )


def test_dh_waypoint_processor_classify_admit_and_feedback(shared_pipeline: ArmigerUltraPipeline) -> None:
    adapter = ArmigerWaypointAdapter(shared_pipeline, use_zero_cue_catalog=False, reject_dead_swarms=False)

    classify_resp = adapter.handle_classify_batch({
        "schema": "dh-waypoint.processor.v1",
        "torrents": [
            {
                "info_hash": "1111222233334444555566667777888899990000",
                "name": "Sintel.2010.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN",
                "size": 74_500_000_000,
                "files": [{"path": "Sintel.2010.2160p.mkv", "size": 74_500_000_000}],
                "swarm": {"seeders": 240, "leechers": 15, "completed": 1200},
            },
            {
                "info_hash": "aaaa22223333444455556666777788889999bbbb",
                "name": "Big.Buck.Bunny.2008.1080p.BluRay.mkv.exe",
                "size": 14_000_000,
                "files": [{"path": "Big.Buck.Bunny.2008.1080p.BluRay.mkv.exe", "size": 14_000_000}],
                "swarm": {"seeders": 5000, "leechers": 10},
            },
        ],
    })
    assert classify_resp["schema"] == "dh-waypoint.processor.v1"
    decisions = classify_resp["decisions"]
    assert len(decisions) == 2
    assert decisions[0]["action"] == "enrich"
    assert decisions[0]["content_type"] == "movie"
    assert decisions[1]["action"] == "delete"

    admit_resp = adapter.handle_admission_batch({
        "schema": "dh-waypoint.processors.admission.v1",
        "candidates": [
            {
                "info_hash": "1111222233334444555566667777888899990000",
                "name": "Sintel.2010.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN",
                "size": 74_500_000_000,
                "files": [{"path": "Sintel.2010.2160p.mkv", "size": 74_500_000_000}],
                "seeders": 240,
                "leechers": 15,
            },
            {
                "info_hash": "aaaa22223333444455556666777788889999bbbb",
                "name": "Big.Buck.Bunny.2008.1080p.BluRay.mkv.exe",
                "size": 14_000_000,
                "files": [{"path": "Big.Buck.Bunny.2008.1080p.BluRay.mkv.exe", "size": 14_000_000}],
                "seeders": 5000,
                "leechers": 10,
            },
        ],
    })
    assert admit_resp["schema"] == "dh-waypoint.processors.admission.v1"
    assert admit_resp["decisions"][0]["admit"] is True
    assert admit_resp["decisions"][1]["admit"] is False

    fb_resp = adapter.handle_feedback_batch({
        "schema": "dh-waypoint.processors.feedback.v1",
        "events": [
            {
                "info_hash": "1111222233334444555566667777888899990000",
                "name": "Sintel.2010.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN",
                "tier": "t4_playback",
                "outcome": "verified",
                "content_type": "movie",
                "size": 74_500_000_000,
                "attributes": {"video_codec": "hevc", "audio_codec": "truehd", "audio_lang": "eng"},
            }
        ],
    })
    assert fb_resp["schema"] == "dh-waypoint.processors.feedback.v1"
    assert fb_resp["accepted"] == 1
    assert fb_resp["verified"] == 1


def test_bitmagnet_bridge_enrichment_and_malware_pruning(shared_pipeline: ArmigerUltraPipeline) -> None:
    bridge = BitmagnetArmigerBridge(shared_pipeline, use_zero_cue_catalog=False)
    res = bridge.evaluate_batch([
        {
            "infoHash": "2222333344445555666677778888999900001111",
            "name": "Cosmos.Laundromat.S01E01.1080p.WEB-DL.DDP5.1.H.264-OPEN",
            "size": 1_850_000_000,
            "seeders": 95,
            "leechers": 8,
            "files": [{"path": "Cosmos.Laundromat.S01E01.mkv", "size": 1_850_000_000}],
        },
        {
            "infoHash": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "name": "Tears.of.Steel.2012.1080p.BluRay.mkv.scr",
            "size": 8_500_000,
            "seeders": 3200,
            "leechers": 4,
            "files": [{"path": "Tears.of.Steel.2012.1080p.BluRay.mkv.scr", "size": 8_500_000}],
        },
    ])
    assert res["total"] == 2
    assert res["deletedCount"] == 1
    assert res["deleteInfoHashes"] == ["deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"]
    assert res["actions"][0]["action"] == "update"
    assert res["actions"][0]["contentType"] == "tv_show"
    assert res["actions"][1]["action"] == "delete"
    assert res["graphqlDeleteMutation"] is not None


def test_arr_torznab_proxy_filters_malware_and_enriches_clean_items(shared_pipeline: ArmigerUltraPipeline) -> None:
    torznab_filter = ArmigerTorznabFilter(
        shared_pipeline,
        require_english_audio=False,
        reject_dead_swarms=False,
        use_zero_cue_catalog=False,
    )
    sample_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:torznab="http://torznab.com/schemas/2015/feed">
  <channel>
    <title>Upstream Torznab Indexer</title>
    <item>
      <title>Sintel.2010.1080p.BluRay.x264-OPEN</title>
      <size>8500000000</size>
      <torznab:attr name="infohash" value="1111222233334444555566667777888899990000"/>
      <torznab:attr name="seeders" value="120"/>
      <torznab:attr name="peers" value="135"/>
    </item>
    <item>
      <title>Sintel.2010.1080p.BluRay.mkv.exe</title>
      <size>12000000</size>
      <torznab:attr name="infohash" value="bbbb22223333444455556666777788889999cccc"/>
      <torznab:attr name="seeders" value="4000"/>
      <torznab:attr name="peers" value="4010"/>
    </item>
  </channel>
</rss>"""

    filtered_xml, stats = torznab_filter.filter_torznab_xml(sample_xml)
    assert stats["total"] == 2
    assert stats["admitted"] == 1
    assert stats["blocked"] == 1

    root = ET.fromstring(filtered_xml)
    items = root.findall("./channel/item")
    assert len(items) == 1
    assert items[0].findtext("title") == "Sintel.2010.1080p.BluRay.x264-OPEN"


def test_arr_and_qbittorrent_webhook_receiver(shared_pipeline: ArmigerUltraPipeline) -> None:
    svc = ArmigerArrWebhookService(shared_pipeline, enable_ffprobe=False)

    # 1. Native Sonarr webhook with tvdbId
    sonarr_res = svc.process_webhook(
        "/webhooks/sonarr",
        {
            "eventType": "Download",
            "series": {"title": "Cosmos Laundromat", "tvdbId": 310123},
            "episodes": [{"seasonNumber": 1, "episodeNumber": 1}],
            "release": {
                "releaseTitle": "Cosmos.Laundromat.S01E01.1080p.WEB-DL.DDP5.1.H.264-OPEN",
                "size": 1_650_000_000,
            },
            "episodeFile": {
                "relativePath": "Season 01/Cosmos.Laundromat.S01E01.mkv",
                "size": 1_650_000_000,
                "mediaInfo": {
                    "videoCodec": "h264",
                    "audioFormat": "eac3",
                    "audioChannels": 5.1,
                    "audioLanguages": "eng",
                    "subtitles": "eng",
                },
            },
        },
    )
    assert sonarr_res["accepted"] is True
    assert sonarr_res["verified"] is True
    assert sonarr_res["category"] == "tv_show"

    # 2. Native Radarr webhook with tmdbId
    radarr_res = svc.process_webhook(
        "/webhooks/radarr",
        {
            "eventType": "Download",
            "movie": {"title": "Tears of Steel", "year": 2012, "tmdbId": 126889},
            "release": {
                "releaseTitle": "Tears.of.Steel.2012.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN",
                "size": 32_000_000_000,
            },
            "movieFile": {
                "relativePath": "Tears of Steel (2012)/Tears.of.Steel.2012.2160p.mkv",
                "size": 32_000_000_000,
            },
        },
    )
    assert radarr_res["accepted"] is True
    assert radarr_res["verified"] is True
    assert radarr_res["category"] == "movie"

    # 3. Native Lidarr webhook with mbId
    lidarr_res = svc.process_webhook(
        "/webhooks/lidarr",
        {
            "eventType": "Download",
            "artist": {"name": "Open Blend Orchestra", "mbId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"},
            "album": {"title": "Sintel Suite", "mbId": "f1e2d3c4-b5a6-7890-abcd-ef0987654321"},
            "trackFiles": [{"size": 380_000_000, "audioFormat": "flac", "bitrate": 950000}],
        },
    )
    assert lidarr_res["accepted"] is True
    assert lidarr_res["verified"] is True
    assert lidarr_res["category"] == "music"

    # 4. qBittorrent completion event + explicit ffprobe verification
    qb_res = svc.process_webhook(
        "/webhooks/qbittorrent",
        {
            "name": "Big.Buck.Bunny.2008.1080p.BluRay.x264-OPEN",
            "hash": "3333444455556666777788889999000011112222",
            "size": 8_200_000_000,
            "progress": 1.0,
            "state": "uploading",
            "content_path": "/data/media/Big.Buck.Bunny.2008.1080p.BluRay.x264-OPEN.mkv",
        },
        explicit_ffprobe={
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080},
                {"codec_type": "audio", "codec_name": "ac3", "channels": 6, "tags": {"language": "eng"}},
            ],
            "format": {"format_name": "matroska,webm", "duration": "596.5", "size": "8200000000"},
        },
    )
    assert qb_res["accepted"] is True
    assert qb_res["verified"] is True
    assert qb_res["category"] == "movie"
