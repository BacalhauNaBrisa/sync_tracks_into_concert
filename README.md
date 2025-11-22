# sync_tracks_into_concert

Synchronize high-quality **per-track WAVs** with a **full-concert audio track** (extracted from video) and build a single `.mka` plus a Matroska **chapters XML** file.

Typical use case:

- You have a concert video with OK audio (`.mkv`, `.mp4`, `.m4a`, etc.).
- You also have a mixed/mastered audio album (one **WAV per song**, numbered in order).
- You want to **replace the concert’s audio** with the album-quality tracks, perfectly aligned with the original show (including interludes and crowd noise), and create **chapters** per song.

This tool:

- Detects each song’s position in the reference concert audio via **cross-correlation**.
- Replaces those segments with the corresponding **high-quality WAV**.
- Keeps any remaining parts (intros/outros/crowd noise) from the original reference audio.
- Generates a **Matroska chapters XML** with one chapter per song.

You can then mux the resulting `.mka` into your video using tools like **MKVToolNix / mkvmerge**.

---

## Features

- Automatic alignment of per-track WAVs against a long reference audio file.
- Robust to small timing differences (drifts, fades, etc.), using:
  - Downsampled mono alignment signal.
  - Short snippet per track (configurable).
  - Sliding search window with backtracking.
- Handles very long concerts safely by using **FLAC** for intermediate reference audio (avoids the 4 GiB WAV limit).
- Generates:
  - A single **merged `.mka`** audio file.
  - A **chapters XML** file for Matroska containers.
- **Dry-run mode** to print detected offsets & chapter times without writing any files.
- **Sanity check** to warn about:
  - Overlaps between aligned tracks.
  - Gaps larger than a configurable threshold.

---

## Requirements

- **Python** 3.8+
- **ffmpeg** available in `PATH` (or provide a custom path with `--ffmpeg-bin`).
- Python packages:
  - `numpy`
  - `scipy`
  - `soundfile`

Install dependencies (example with `pip`):

```bash
    pip install numpy scipy soundfile
```

Install ffmpeg (examples):

macOS (Homebrew):
```bash
    brew install ffmpeg
```
Ubuntu / Debian:
```bash
    sudo apt-get update
    sudo apt-get install ffmpeg
```
Naming convention for WAV tracks

The script expects each per-track WAV to be named:

NN-Title.wav
NN - Title.wav
NN_Title.wav

Examples:

    01-Mars for the Rich.wav

    02-Converge.wav

    03_Witchcraft.wav

Where:

    NN = track number (1, 2, 3, …).

    The rest is the track title (used for chapter names).

Only .wav files are read from --songs-dir.
Basic usage
1. Real run (build .mka + chapters XML)
```bash
python sync_concert_audio.py \
  --ref-audio original_concert.mka \
  --songs-dir ./tracks_wav \
  --output-audio concert_merged.mka \
  --chapters-xml concert_chapters.xml
```
Where:

    original_concert.mka is the audio extracted from the concert video
    (could also be .mkv, .mp4, .m4a, .flac, .wav, .mp3, etc.).

    ./tracks_wav is a folder containing NN-Title.wav files.

The script will:

    Convert the reference audio to a lossless FLAC at the same sample rate & channels as the WAVs (to avoid WAV size limits).

    Build a downsampled mono alignment signal.

    Align each track in order and compute start/end positions.

    Perform a sanity check on gaps/overlaps.

    Create a merged high-quality .wav, wrap it into .mka, and write:

        concert_merged.mka

        concert_chapters.xml

2. Dry run (no files written)

Use this to verify the alignment before generating the merged audio:
```bash
python sync_concert_audio.py \
  --ref-audio original_concert.mka \
  --songs-dir ./tracks_wav \
  --dry-run
```
Output includes:

    Approximate reference duration.

    Table with, for each track:

        Start / End time (HH:MM:SS.fffffffff)

        Start time in seconds

    Sanity check table with:

        Gaps / overlaps between consecutive tracks.

        Warnings for large gaps or overlaps.

If everything looks good, re-run without --dry-run.
Command-line options

--ref-audio PATH           (required)
    Original audio from video (.mka, .mkv, .mp4, .m4a, .flac, .wav, .mp3, etc.)

--songs-dir PATH           (required)
    Directory containing numbered WAV tracks (e.g. '01-Intro.wav', '02-Song.wav', ...)

--output-audio PATH
    Output .mka audio filename (default: concert_merged.mka)

--chapters-xml PATH
    Output Matroska chapters XML filename (default: chapters.xml)

--tmp-dir PATH
    Temporary working directory for intermediate lossless audio files
    (default: tmp_sync)

--ffmpeg-bin CMD
    Name/path of ffmpeg binary (default: "ffmpeg" in PATH)

--align-sr INT
    Sample rate used internally for alignment (mono).
    Lower = faster alignment; default: 8000 Hz

--max-search-gap FLOAT
    Maximum seconds allowed between consecutive songs in the reference
    when searching for the next track. Default: 180.0

--align-clip-seconds FLOAT
    Length (in seconds) of the snippet used from each song for alignment.
    Default: 30.0

--align-clip-offset FLOAT
    Offset (in seconds) inside each song where the alignment snippet starts.
    Default: 0.0 (from the very beginning of each track)

--align-backtrack FLOAT
    How many seconds backwards the search window may move relative to the
    previous song’s end when searching for the next song.
    Default: 30.0

--sanity-max-gap FLOAT
    Warn if the gap between consecutive tracks exceeds this many seconds.
    Default: 300.0

--dry-run
    Only detect and print alignment; do NOT write any output audio or XML files.

How alignment works (high level)

    The script extracts the reference audio to FLAC with the same sample rate and channels as your WAV tracks.

    It builds a downsampled mono version of:

        The reference (full concert).

        Each track (full song).

    For each track, it:

        Takes a configurable snippet from the track (e.g. 10–30 seconds).

        Searches for that snippet in a sliding window over the reference, starting from where the previous track ended (with an optional backtrack).

        Uses cross-correlation to find the best match.

        Converts that match position back into sample indices in the full-resolution reference.

    For the last tracks, if the remaining reference window is shorter than the snippet, the snippet is automatically shortened rather than failing.

    Finally:

        The corresponding segments in the reference are replaced with the high-quality track audio.

        The rest of the reference (e.g. crowd noise between songs) is preserved.

Sanity checks

After alignment, the script prints a sanity check table that:

    Lists each track in time order.

    Shows:

        Start / End timestamps.

        The gap between this track and the previous one.

        Whether there is an overlap or a large gap (above --sanity-max-gap).

This is useful to:

    Detect misalignments (e.g. one track placed in the wrong section).

    Verify that the spacing between songs is reasonable.

If something looks off:

    You can re-run with different alignment parameters:

        --align-clip-seconds (shorter/longer snippets).

        --align-clip-offset (start alignment a bit into the song).

        --max-search-gap and --align-backtrack to widen/narrow search windows.

Troubleshooting
ffmpeg not found

If you see an error like:

ffmpeg command failed with code 1: ...

or

No such file or directory: 'ffmpeg'

You might need to:

    Install ffmpeg (see Requirements), or

    Provide the explicit path:
```bash
python sync_concert_audio.py \
  --ref-audio original_concert.mka \
  --songs-dir ./tracks_wav \
  --ffmpeg-bin /usr/local/bin/ffmpeg \
  --dry-run
```
No .wav files found

Ensure your --songs-dir is correct and that files are:

    Real .wav files (not .flac, .mp3, etc.).

    Named with a numeric prefix, e.g.:

    01-Intro.wav
    02-Track Name.wav
    03-Another_Track.wav

Last track fails to align

This project already includes a fallback for the end of the reference:

    If the remaining reference segment is shorter than the snippet, the snippet is automatically shortened instead of failing with “Not enough reference left…”.

If a track is truly missing in the reference (e.g. edited out in the video), alignment may still be ambiguous. In that case, you can:

    Temporarily comment that track out from the folder and re-run.

    Or adjust --align-clip-offset and --align-clip-seconds to use a more distinctive passage of the song.
