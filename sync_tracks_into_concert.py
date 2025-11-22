#!/usr/bin/env python3
"""
Synchronize high-quality per-track WAVs with a concert audio track
extracted from video, and build a single .mka plus a Matroska chapter file.

Usage example (real run):

    python sync_concert_audio.py \
        --ref-audio original.m4a \
        --songs-dir ./tracks_wav \
        --output-audio concert_hq.mka \
        --chapters-xml concert_chapters.xml

Usage example (dry run, no files written – just offsets & chapters table):

    python sync_concert_audio.py \
        --ref-audio original.m4a \
        --songs-dir ./tracks_wav \
        --dry-run

Requirements:
    - Python 3.8+
    - ffmpeg available in PATH
    - pip install numpy scipy soundfile
"""

import argparse
import math
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, correlate


# --------------------------- Helpers & dataclasses --------------------------- #

@dataclass
class TrackInfo:
    index: int
    title: str
    path: Path
    data_full: np.ndarray          # shape: (frames, channels), high-quality
    duration_samples: int
    align_mono: np.ndarray         # 1D, downsampled mono for alignment (full track)
    start_sample_full: int = None  # in final sample rate
    end_sample_full: int = None    # in final sample rate


def run_ffmpeg(cmd: list):
    """Run ffmpeg and raise a clear error if it fails."""
    print("Running ffmpeg:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed with code {result.returncode}:\n"
            f"{result.stdout.decode(errors='ignore')}"
        )


def format_time_hhmmss_ns(seconds: float) -> str:
    """Format seconds as HH:MM:SS.000000000 (nanosecond precision)."""
    ns_total = int(round(seconds * 1e9))
    h, rem = divmod(ns_total, 3600 * 10**9)
    m, rem = divmod(rem, 60 * 10**9)
    s, ns = divmod(rem, 10**9)
    return f"{h:02d}:{m:02d}:{s:02d}.{ns:09d}"


def ensure_channels(data: np.ndarray, target_channels: int) -> np.ndarray:
    """
    Ensure audio has the desired number of channels.
    - If data has more channels, downmix to mono then duplicate if needed.
    - If data has fewer, duplicate channels.
    """
    if data.ndim == 1:
        data = data[:, None]

    current_channels = data.shape[1]

    if current_channels == target_channels:
        return data

    # Downmix to mono if necessary
    mono = data.mean(axis=1, keepdims=True)

    if target_channels == 1:
        return mono

    # Duplicate mono across channels
    if target_channels > 1:
        return np.repeat(mono, target_channels, axis=1)

    # Fallback
    return data


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample multi-channel audio using polyphase filtering."""
    if orig_sr == target_sr:
        return data

    if data.ndim == 1:
        data = data[:, None]

    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    resampled = resample_poly(data, up, down, axis=0)
    return resampled


def downsample_mono_for_alignment(
    data: np.ndarray, orig_sr: int, align_sr: int
) -> np.ndarray:
    """Convert multi-channel data to mono and downsample to align_sr."""
    if data.ndim == 2:
        mono = data.mean(axis=1)
    else:
        mono = data

    if orig_sr == align_sr:
        return mono.astype(np.float32)

    g = math.gcd(orig_sr, align_sr)
    up = align_sr // g
    down = orig_sr // g
    mono = mono.astype(np.float32)
    ds = resample_poly(mono, up, down)
    return ds.astype(np.float32)


def make_align_snippet(
    full_align_mono: np.ndarray,
    align_sr: int,
    clip_seconds: float,
    offset_seconds: float,
) -> Tuple[np.ndarray, int]:
    """
    Return a snippet of the alignment signal plus its offset within the track.

    - clip_seconds: desired length of snippet in seconds (e.g. 30.0)
    - offset_seconds: where to start the snippet inside the track (e.g. 0.0, 5.0)

    Returns:
        snippet (1D np.ndarray), snippet_offset_samples (int, in align_sr domain)
    """
    total_len = len(full_align_mono)
    if clip_seconds <= 0 or total_len == 0:
        return full_align_mono, 0

    clip_len = int(round(clip_seconds * align_sr))
    if clip_len >= total_len:
        # whole track is shorter than snippet length – use full track
        return full_align_mono, 0

    offset = int(round(offset_seconds * align_sr))
    if offset < 0:
        offset = 0
    if offset + clip_len > total_len:
        # shift snippet window to fit into track
        offset = max(0, total_len - clip_len)

    snippet = full_align_mono[offset : offset + clip_len]
    return snippet, offset


# --------------------------- Core functionality ----------------------------- #

def detect_song_files(songs_dir: Path) -> List[Path]:
    """Return a list of WAV file paths sorted by their numeric prefix."""
    wavs = sorted(songs_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in {songs_dir}")

    def extract_index(path: Path) -> int:
        m = re.match(r"(\d+)\s*[-_]\s*(.+)\.wav$", path.name, flags=re.IGNORECASE)
        if not m:
            raise ValueError(
                f"Filename '{path.name}' does not match pattern 'NN-Title.wav'"
            )
        return int(m.group(1))

    wavs.sort(key=extract_index)
    return wavs


def get_song_format(song_path: Path):
    """Inspect first song to get sample rate, channels and subtype."""
    info = sf.info(str(song_path))
    sr = info.samplerate
    channels = info.channels
    subtype = info.subtype  # eg 'PCM_16', 'PCM_24'
    return sr, channels, subtype


def load_tracks(
    wav_paths: List[Path],
    target_sr: int,
    target_channels: int,
    align_sr: int,
) -> List[TrackInfo]:
    """Load all WAV tracks and build alignment signals."""
    tracks: List[TrackInfo] = []

    for path in wav_paths:
        m = re.match(r"(\d+)\s*[-_]\s*(.+)\.wav$", path.name, flags=re.IGNORECASE)
        if not m:
            print(f"Skipping '{path.name}', name does not match 'NN-Title.wav'")
            continue

        index = int(m.group(1))
        title_raw = m.group(2)
        title = title_raw.replace("_", " ").strip()

        print(f"Loading track {index:02d}: {title} ({path.name})")
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)

        if sr != target_sr:
            print(
                f"  Resampling {path.name} from {sr} Hz to {target_sr} Hz for final mix"
            )
            data = resample_audio(data, sr, target_sr)
            sr = target_sr

        data = ensure_channels(data, target_channels)
        duration_samples = data.shape[0]

        align_mono = downsample_mono_for_alignment(data, sr, align_sr)

        tracks.append(
            TrackInfo(
                index=index,
                title=title,
                path=path,
                data_full=data,
                duration_samples=duration_samples,
                align_mono=align_mono,
            )
        )

    tracks.sort(key=lambda t: t.index)
    return tracks


def extract_ref_audio_to_flac(
    ref_audio_path: Path,
    tmp_dir: Path,
    target_sr: int,
    target_channels: int,
    ffmpeg_binary: str,
) -> Path:
    """
    Use ffmpeg to extract/convert reference audio into a FLAC file.

    We avoid WAV here because very long concerts can hit the 4 GiB WAV limit,
    which truncates the audio. FLAC avoids that and is handled fine by
    soundfile/libsndfile.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / "ref_audio_converted.flac"

    cmd = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(ref_audio_path),
        "-vn",
        "-ac",
        str(target_channels),
        "-ar",
        str(target_sr),
        "-c:a",
        "flac",
        str(out_path),
    ]
    run_ffmpeg(cmd)
    return out_path


def align_tracks(
    ref_align_mono: np.ndarray,
    tracks: List[TrackInfo],
    align_sr: int,
    full_sr: int,
    max_search_gap_s: float = 180.0,
    clip_seconds: float = 30.0,
    clip_offset_seconds: float = 0.0,
    backtrack_seconds: float = 30.0,
):
    """
    For each song, find best alignment offset in reference using cross-correlation
    of a snippet (not the entire song).

    - ref_align_mono: mono reference at align_sr
    - tracks: TrackInfo list with .align_mono prepared (full track)
    """
    cursor = 0  # search start (in align samples)

    for t in tracks:
        song_align_full = t.align_mono
        song_len_align = len(song_align_full)
        if song_len_align == 0:
            raise RuntimeError(f"Empty alignment signal for track {t.title}")

        snippet, snippet_offset = make_align_snippet(
            song_align_full, align_sr, clip_seconds, clip_offset_seconds
        )
        snippet_len = len(snippet)

        extra = int(max_search_gap_s * align_sr)
        backtrack = int(backtrack_seconds * align_sr)

        seg_start = max(0, cursor - backtrack)
        seg_end = min(len(ref_align_mono), seg_start + song_len_align + extra)
        seg_len = seg_end - seg_start

        if seg_len <= 0:
            raise RuntimeError(
                f"Reference ends before track {t.index} '{t.title}' search window; nothing left to align."
            )

        # If the remaining reference window is shorter than the snippet,
        # shrink the snippet instead of aborting. This is especially useful
        # for the last track near the end of the reference.
        if seg_len < snippet_len:
            print(
                f"[WARN] Search window ({seg_len} samples) for track {t.index} "
                f"'{t.title}' is smaller than snippet ({snippet_len} samples). "
                f"Using shortened snippet."
            )
            snippet = snippet[:seg_len]
            snippet_len = seg_len

        ref_seg = ref_align_mono[seg_start:seg_end]

        print(
            f"Aligning track {t.index:02d} '{t.title}' "
            f"using a {clip_seconds:.1f}s snippet (offset {clip_offset_seconds:.1f}s) "
            f"in ref[{seg_start}:{seg_end}] (len={seg_len})"
        )

        ref_z = ref_seg - ref_seg.mean()
        snip_z = snippet - snippet.mean()

        corr = correlate(ref_z, snip_z, mode="valid", method="fft")

        best_offset_in_seg = int(np.argmax(corr))
        best_offset_align_snip = seg_start + best_offset_in_seg

        # Convert snippet offset to full track start in alignment samples
        start_align = best_offset_align_snip - snippet_offset
        if start_align < 0:
            start_align = 0

        start_sec = start_align / align_sr
        start_sample_full = int(round(start_sec * full_sr))
        end_sample_full = start_sample_full + t.duration_samples

        t.start_sample_full = start_sample_full
        t.end_sample_full = end_sample_full

        # Advance cursor to after the end of this track in alignment samples
        end_align = start_align + song_len_align
        cursor = end_align

        print(
            f"  -> snippet best at {best_offset_align_snip/align_sr:0.3f}s, "
            f"track start at {format_time_hhmmss_ns(start_sec)} "
            f"(sample {start_sample_full} at {full_sr} Hz)"
        )


def build_composite_mix(
    ref_full: np.ndarray, tracks: List[TrackInfo]
) -> np.ndarray:
    """
    Build the final composite:
    - Start from the entire reference (already in target_sr & channels)
    - For each track, replace the corresponding segment with the HQ song audio.
    """
    composite = np.array(ref_full, copy=True)
    total_frames = composite.shape[0]

    for t in tracks:
        if t.start_sample_full is None or t.end_sample_full is None:
            raise RuntimeError(f"Track {t.index} '{t.title}' is not aligned.")

        start = t.start_sample_full
        end = t.end_sample_full
        song_frames = t.duration_samples

        if start < 0 or start >= total_frames:
            print(
                f"WARNING: Track {t.index} '{t.title}' start {start} "
                f"outside reference length {total_frames}; skipping."
            )
            continue

        if end > total_frames:
            print(
                f"WARNING: Track {t.index} '{t.title}' overruns the reference "
                f"(end {end} > {total_frames}); truncating."
            )
            end = total_frames
            song_frames = end - start

        print(
            f"Replacing ref[{start}:{end}] with "
            f"track {t.index:02d} '{t.title}' ({song_frames} frames)"
        )

        composite[start:end, :] = t.data_full[:song_frames, :]

    return composite


def write_chapters_xml(tracks: List[TrackInfo], sr: int, out_path: Path):
    """Write a Matroska chapters XML file with one chapter per song."""
    import xml.etree.ElementTree as ET

    root = ET.Element("Chapters")
    edition = ET.SubElement(root, "EditionEntry")

    edition_uid = ET.SubElement(edition, "EditionUID")
    edition_uid.text = str(random.getrandbits(64))

    for t in tracks:
        if t.start_sample_full is None or t.end_sample_full is None:
            raise RuntimeError(f"Track {t.index} '{t.title}' is not aligned.")

        start_sec = t.start_sample_full / sr
        end_sec = t.end_sample_full / sr

        atom = ET.SubElement(edition, "ChapterAtom")

        ts = ET.SubElement(atom, "ChapterTimeStart")
        ts.text = format_time_hhmmss_ns(start_sec)

        te = ET.SubElement(atom, "ChapterTimeEnd")
        te.text = format_time_hhmmss_ns(end_sec)

        disp = ET.SubElement(atom, "ChapterDisplay")
        cs = ET.SubElement(disp, "ChapterString")
        cs.text = t.title

        clang = ET.SubElement(disp, "ChapterLanguage")
        clang.text = "und"

        ci = ET.SubElement(disp, "ChapLanguageIETF")
        ci.text = "und"

        cuid = ET.SubElement(atom, "ChapterUID")
        cuid.text = str(random.getrandbits(64))

    tree = ET.ElementTree(root)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    print(f"Wrote chapters XML to {out_path}")


def print_dry_run_summary(tracks: List[TrackInfo], sr: int, ref_frames: int):
    """Print detected offsets and chapter times without writing any files."""
    total_duration_sec = ref_frames / sr
    print("\n=== DRY RUN: Alignment summary (no files written) ===")
    print(f"Reference duration: {format_time_hhmmss_ns(total_duration_sec)} ({total_duration_sec:.3f} s)\n")
    print("{:<4} {:<30} {:>20} {:>20} {:>14}".format(
        "No.", "Title", "Start (HH:MM:SS)", "End (HH:MM:SS)", "Start (s)"
    ))
    print("-" * 96)

    for t in tracks:
        if t.start_sample_full is None or t.end_sample_full is None:
            raise RuntimeError(f"Track {t.index} '{t.title}' is not aligned.")

        start_sec = t.start_sample_full / sr
        end_sec = t.end_sample_full / sr
        start_str = format_time_hhmmss_ns(start_sec)
        end_str = format_time_hhmmss_ns(end_sec)

        print("{:<4} {:<30} {:>20} {:>20} {:>14.3f}".format(
            t.index,
            t.title[:30],
            start_str,
            end_str,
            start_sec,
        ))

    print("\nRe-run WITHOUT '--dry-run' to actually build the merged audio and XML chapters.\n")


def sanity_check_alignment(tracks: List[TrackInfo], sr: int, max_gap_warn_s: float):
    """
    Sanity check:
      - warn if any track overlaps the previous one
      - warn if the gap between consecutive tracks is larger than max_gap_warn_s
    """
    print("\n=== Sanity check: gaps & overlaps between tracks ===")

    tracks_sorted = sorted(
        [t for t in tracks if t.start_sample_full is not None and t.end_sample_full is not None],
        key=lambda t: t.start_sample_full,
    )

    if not tracks_sorted:
        print("No tracks with alignment data to check.")
        return

    print("{:<4} {:<30} {:>20} {:>20} {:>12}".format(
        "No.", "Title", "Start", "End", "Gap (s)"
    ))
    print("-" * 92)

    prev = None
    for t in tracks_sorted:
        start_sec = t.start_sample_full / sr
        end_sec = t.end_sample_full / sr
        gap_sec = None
        status = ""

        if prev is not None:
            gap_samples = t.start_sample_full - prev.end_sample_full
            gap_sec = gap_samples / sr

            if gap_samples < 0:
                status = f"WARNING: OVERLAP {gap_sec:.3f}s with {prev.index:02d}-{prev.title}"
            elif gap_sec > max_gap_warn_s:
                status = f"WARNING: LARGE GAP {gap_sec:.3f}s after {prev.index:02d}-{prev.title}"
            else:
                status = f"{gap_sec:.3f}s"

        start_str = format_time_hhmmss_ns(start_sec)
        end_str = format_time_hhmmss_ns(end_sec)

        print("{:<4} {:<30} {:>20} {:>20} {:>12}".format(
            t.index,
            t.title[:30],
            start_str,
            end_str,
            "" if gap_sec is None else status,
        ))

        prev = t

    print(
        f"\nSanity check complete. Gaps > {max_gap_warn_s:.1f}s or any overlaps are flagged above.\n"
    )


# ----------------------------------- main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Sync per-track WAVs with concert audio and build a single MKA."
    )
    parser.add_argument(
        "--ref-audio",
        required=True,
        help="Original audio from video (.mka, .mkv, .mp4, .flac, .wav, .mp3, etc.)",
    )
    parser.add_argument(
        "--songs-dir",
        required=True,
        help="Directory containing numbered WAV tracks (e.g. '01-Intro.wav')",
    )
    parser.add_argument(
        "--output-audio",
        default="concert_merged.mka",
        help="Output .mka audio filename",
    )
    parser.add_argument(
        "--chapters-xml",
        default="chapters.xml",
        help="Output Matroska chapters XML filename",
    )
    parser.add_argument(
        "--tmp-dir",
        default="tmp_sync",
        help="Temporary working directory (for intermediate lossless audio files)",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Name/path of ffmpeg binary (default: ffmpeg in PATH)",
    )
    parser.add_argument(
        "--align-sr",
        type=int,
        default=8000,
        help="Sample rate used internally for alignment (mono, low to speed up).",
    )
    parser.add_argument(
        "--max-search-gap",
        type=float,
        default=180.0,
        help="Max seconds allowed between consecutive songs in the reference (search window).",
    )
    parser.add_argument(
        "--align-clip-seconds",
        type=float,
        default=30.0,
        help="Length (in seconds) of the snippet used for alignment from each song.",
    )
    parser.add_argument(
        "--align-clip-offset",
        type=float,
        default=0.0,
        help="Offset (in seconds) inside each song where the alignment snippet starts.",
    )
    parser.add_argument(
        "--align-backtrack",
        type=float,
        default=30.0,
        help="How many seconds backwards the search window may move relative to the previous song end.",
    )
    parser.add_argument(
        "--sanity-max-gap",
        type=float,
        default=300.0,
        help="Warn if the gap between consecutive tracks exceeds this many seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only detect and print alignment; do NOT write any output audio or XML files.",
    )
    args = parser.parse_args()

    ref_audio_path = Path(args.ref_audio)
    songs_dir = Path(args.songs_dir)
    tmp_dir = Path(args.tmp_dir)
    output_audio_path = Path(args.output_audio)
    chapters_xml_path = Path(args.chapters_xml)

    # 1) Detect song files and infer target audio format from the first song.
    wav_paths = detect_song_files(songs_dir)
    first_song = wav_paths[0]
    target_sr, target_channels, song_subtype = get_song_format(first_song)

    print(
        f"Detected song format: {target_sr} Hz, {target_channels} ch, subtype={song_subtype}"
    )

    # 2) Extract/convert reference audio into a FLAC matching song SR/channels.
    ref_lossless_path = extract_ref_audio_to_flac(
        ref_audio_path=ref_audio_path,
        tmp_dir=tmp_dir,
        target_sr=target_sr,
        target_channels=target_channels,
        ffmpeg_binary=args.ffmpeg_bin,
    )

    # 3) Load reference audio and prepare alignment signal.
    ref_full, ref_sr = sf.read(str(ref_lossless_path), dtype="float32", always_2d=True)
    if ref_sr != target_sr:
        print(
            f"WARNING: Reference sample rate {ref_sr} != target {target_sr}, resampling."
        )
        ref_full = resample_audio(ref_full, ref_sr, target_sr)
        ref_sr = target_sr

    ref_full = ensure_channels(ref_full, target_channels)

    print(
        f"Loaded reference audio: {ref_lossless_path.name}, "
        f"{ref_full.shape[0]} frames, {ref_sr} Hz, {ref_full.shape[1]} channels"
    )

    ref_align_mono = downsample_mono_for_alignment(ref_full, ref_sr, args.align_sr)
    print(
        f"Prepared mono alignment signal for reference: "
        f"len={len(ref_align_mono)}, align_sr={args.align_sr}"
    )

    # 4) Load tracks and build their alignment signals.
    tracks = load_tracks(
        wav_paths=wav_paths,
        target_sr=target_sr,
        target_channels=target_channels,
        align_sr=args.align_sr,
    )
    if not tracks:
        raise RuntimeError("No tracks loaded. Check your WAV filenames.")

    # 5) Align each track in the reference timeline.
    align_tracks(
        ref_align_mono=ref_align_mono,
        tracks=tracks,
        align_sr=args.align_sr,
        full_sr=target_sr,
        max_search_gap_s=args.max_search_gap,
        clip_seconds=args.align_clip_seconds,
        clip_offset_seconds=args.align_clip_offset,
        backtrack_seconds=args.align_backtrack,
    )

    # 6) Sanity check (gaps and overlaps)
    sanity_check_alignment(
        tracks=tracks,
        sr=target_sr,
        max_gap_warn_s=args.sanity_max_gap,
    )

    # ---- DRY RUN MODE: only print alignment, no files written ---- #
    if args.dry_run:
        print_dry_run_summary(tracks=tracks, sr=target_sr, ref_frames=ref_full.shape[0])
        return

    # 7) Build composite mix (songs replaced by HQ, missing parts from ref).
    composite = build_composite_mix(ref_full=ref_full, tracks=tracks)

    # 8) Write composite to a temporary WAV with same subtype as songs.
    tmp_dir.mkdir(parents=True, exist_ok=True)
    merged_wav_path = tmp_dir / "concert_merged.wav"
    print(
        f"Writing merged high-quality WAV ({composite.shape[0]} frames, "
        f"{target_sr} Hz, {composite.shape[1]} ch) to {merged_wav_path}"
    )
    sf.write(
        str(merged_wav_path),
        composite,
        target_sr,
        subtype=song_subtype,
    )

    # 9) Wrap WAV in an MKA container, copying audio as-is.
    cmd_mka = [
        args.ffmpeg_bin,
        "-y",
        "-i",
        str(merged_wav_path),
        "-c:a",
        "copy",
        str(output_audio_path),
    ]
    run_ffmpeg(cmd_mka)
    print(f"Wrote final .mka audio to {output_audio_path}")

    # 10) Generate chapters XML.
    write_chapters_xml(tracks=tracks, sr=target_sr, out_path=chapters_xml_path)

    print("\nDone.")
    print("You can now mux the new .mka into the original video (e.g. with mkvmerge).")


if __name__ == "__main__":
    main()
