#!/usr/bin/env python3
"""
transcribe.py - turn audio or video files into text, locally, with Whisper.
Optionally labels who is speaking (Speaker 1, Speaker 2, ... or real names).

Setup (once):
    pip install faster-whisper
    pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"    (only if you have an NVIDIA GPU)
    pip install sherpa-onnx                                    (only for --speakers)

Usage:
    python transcribe.py recording.mp3
    python transcribe.py "C:\\Recordings"                 (every audio/video file in a folder)
    python transcribe.py interview.m4a --model medium     (more accurate, slower)
    python transcribe.py clip.mp4 --language en --formats txt srt
    python transcribe.py                                  (asks you for a file - drag it in)

Who's speaking:
    python transcribe.py meeting.mp3 --speakers                       (works out how many people)
    python transcribe.py meeting.mp3 --speakers 2                     (you know it's 2 - more accurate)
    python transcribe.py meeting.mp3 --speakers 2 --names "Aaron,Sean"
        Names are given in order of who speaks FIRST in the recording.

Outputs are written next to each input file (or to --output-dir):
    name.txt              transcript (one paragraph per speaker turn with --speakers)
    name.timestamps.txt   same, with [mm:ss] at the start of each line
    name.srt / name.vtt   subtitles

Models download automatically on first run (cached afterwards).
Whisper models: tiny, base, small, medium, large-v3, turbo
  - small  : good default, fast on most laptops
  - medium : noticeably better with accents / background noise
  - turbo  : near large-v3 quality, much faster (best if you have an NVIDIA GPU)
"""

import argparse
import importlib.util
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


def add_nvidia_dll_paths():
    """On Windows, let CTranslate2 find cuBLAS/cuDNN installed via pip:
        pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
    Those wheels put their DLLs in site-packages/nvidia/<lib>/bin, which
    Windows doesn't search by default.
    """
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("nvidia")
    if not spec or not spec.submodule_search_locations:
        return
    for root in spec.submodule_search_locations:
        for bin_dir in Path(root).glob("*/bin"):
            if bin_dir.is_dir():
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(str(bin_dir))
                except (OSError, AttributeError):
                    pass


add_nvidia_dll_paths()

GPU_LIB_HINT = (
    "  GPU libraries missing - falling back to CPU (slower).\n"
    "  To use your NVIDIA card, run once:\n"
    '      pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"'
)

MEDIA_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp",
}
ALL_FORMATS = ["txt", "srt", "vtt", "timestamps"]

# Speaker-detection models (sherpa-onnx, from its GitHub releases)
SPEAKER_CACHE = Path.home() / ".cache" / "transcribe-speakers"
_REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEG_URL = f"{_REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL = f"{_REL}/speaker-recongition-models/nemo_en_titanet_small.onnx"


def is_gpu_lib_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("cublas", "cudnn", "cuda", ".dll", "libcu"))


# ---------------------------------------------------------------- formatting

def fmt_time(seconds: float, sep: str = ",") -> str:
    """Seconds -> HH:MM:SS,mmm (srt) or HH:MM:SS.mmm (vtt)."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_short(seconds: float) -> str:
    """Seconds -> mm:ss (or h:mm:ss for long recordings)."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------- inputs / audio

def collect_inputs(paths):
    files = []
    for raw in paths:
        p = Path(raw.strip().strip('"').strip("'")).expanduser()
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in MEDIA_EXTS))
        elif p.is_file():
            files.append(p)
        else:
            print(f"  ! Not found: {p}")
    return files


def pick_device():
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def decode_audio(path: Path, sampling_rate: int = 16000):
    """Decode any audio/video file to 16 kHz mono float32, as Whisper expects.

    Done here rather than by faster-whisper, whose built-in decoder breaks
    with newer PyAV releases ("unexpected keyword argument 'metadata_errors'").
    """
    import av
    import numpy as np

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    chunks = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("no audio track in this file")
        stream = container.streams.audio[0]
        try:
            for frame in container.decode(stream):
                frame.pts = None
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray())
        except av.error.InvalidDataError:
            pass  # damaged tail of a file: keep what decoded fine
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray())

    if not chunks:
        raise ValueError("couldn't decode any audio")
    audio = np.concatenate(chunks, axis=1).reshape(-1)
    return audio.astype(np.float32) / 32768.0


# ---------------------------------------------------------------- speakers

def _download(url: str, dest: Path):
    print(f"  Downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            got += len(block)
            if total:
                print(f"\r    {got / total:4.0%}", end="", flush=True)
    print()
    tmp.replace(dest)


def speaker_models():
    """Download (first time) and return paths to the two speaker models."""
    SPEAKER_CACHE.mkdir(parents=True, exist_ok=True)
    seg = SPEAKER_CACHE / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    emb = SPEAKER_CACHE / "nemo_en_titanet_small.onnx"
    if not seg.exists():
        archive = SPEAKER_CACHE / "segmentation.tar.bz2"
        _download(SEG_URL, archive)
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(SPEAKER_CACHE, filter="data")
            except TypeError:  # Python < 3.12
                tar.extractall(SPEAKER_CACHE)
        archive.unlink()
    if not emb.exists():
        _download(EMB_URL, emb)
    return seg, emb


class SpeakerFinder:
    def __init__(self, num_speakers, threshold):
        try:
            import sherpa_onnx
        except ImportError:
            sys.exit("Speaker detection needs sherpa-onnx. Run:  pip install sherpa-onnx")
        seg, emb = speaker_models()
        threads = max(1, min(8, os.cpu_count() or 4))
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)),
                num_threads=threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers if num_speakers else -1, threshold=threshold
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not cfg.validate():
            raise RuntimeError("speaker model config is invalid (models may be corrupt - "
                               f"delete {SPEAKER_CACHE} and retry)")
        self.sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def find(self, audio):
        """Return [(start, end, speaker_id), ...] sorted by time."""
        def progress(done, total):
            print(f"\r  Finding speakers... {done / max(total, 1):4.0%}", end="", flush=True)
            return 0
        result = self.sd.process(audio, callback=progress).sort_by_start_time()
        print()
        return [(r.start, r.end, r.speaker) for r in result]


def speaker_for(start, end, turns):
    """Speaker whose turns overlap [start, end] most; nearest turn if none overlap."""
    best, best_overlap = None, 0.0
    nearest, nearest_gap = None, float("inf")
    for ts, te, spk in turns:
        overlap = min(end, te) - max(start, ts)
        if overlap > best_overlap:
            best, best_overlap = spk, overlap
        gap = max(ts - end, start - te, 0.0)
        if gap < nearest_gap:
            nearest, nearest_gap = spk, gap
    return best if best is not None else nearest


def smooth(words):
    """Fix one- or two-word blips: A A [B] A A -> all A (usually a mis-hit at a turn edge)."""
    runs = []
    for w in words:
        if runs and runs[-1][0] == w["speaker"]:
            runs[-1][1].append(w)
        else:
            runs.append([w["speaker"], [w]])
    for i in range(1, len(runs) - 1):
        spk, ws = runs[i]
        short = len(ws) <= 2 and (ws[-1]["end"] - ws[0]["start"]) < 1.0
        if short and runs[i - 1][0] == runs[i + 1][0] != spk:
            for w in ws:
                w["speaker"] = runs[i - 1][0]
            runs[i][0] = runs[i - 1][0]


class SpeakerNames:
    """Speaker 1, 2, ... numbered in order of first appearance, or the names given."""

    def __init__(self, names):
        self.names = names or []
        self.order = {}

    def __call__(self, spk):
        if spk is None:
            return None
        if spk not in self.order:
            self.order[spk] = len(self.order)
        i = self.order[spk]
        return self.names[i] if i < len(self.names) else f"Speaker {i + 1}"


def pieces_from_words(words, label):
    """Split one Whisper segment's words into runs by speaker."""
    pieces = []
    for w in words:
        name = label(w["speaker"])
        if pieces and pieces[-1]["speaker"] == name:
            pieces[-1]["end"] = w["end"]
            pieces[-1]["text"] += w["text"]
        else:
            pieces.append({"start": w["start"], "end": w["end"], "text": w["text"], "speaker": name})
    for p in pieces:
        p["text"] = p["text"].strip()
    return [p for p in pieces if p["text"]]


def merge_turns(pieces):
    """Join consecutive pieces from the same speaker into one paragraph."""
    turns = []
    for p in pieces:
        if turns and p["speaker"] is not None and turns[-1]["speaker"] == p["speaker"]:
            turns[-1]["end"] = p["end"]
            turns[-1]["text"] += " " + p["text"]
        else:
            turns.append(dict(p))
    return turns


# ---------------------------------------------------------------- output

def write_outputs(pieces, base: Path, formats):
    has_speakers = any(p.get("speaker") for p in pieces)

    def say(p):
        return f"{p['speaker']}: {p['text']}" if p.get("speaker") else p["text"]

    turns = merge_turns(pieces) if has_speakers else pieces
    gap = "\n\n" if has_speakers else "\n"
    written = []
    if "txt" in formats:
        out = base.with_suffix(".txt")
        out.write_text(gap.join(say(t) for t in turns) + "\n", encoding="utf-8")
        written.append(out)
    if "timestamps" in formats:
        out = base.parent / f"{base.name}.timestamps.txt"
        out.write_text(
            gap.join(f"[{fmt_short(t['start'])}] {say(t)}" for t in turns) + "\n",
            encoding="utf-8",
        )
        written.append(out)
    if "srt" in formats:
        out = base.with_suffix(".srt")
        lines = []
        for i, p in enumerate(pieces, 1):
            lines += [str(i), f"{fmt_time(p['start'])} --> {fmt_time(p['end'])}", say(p), ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        written.append(out)
    if "vtt" in formats:
        out = base.with_suffix(".vtt")
        lines = ["WEBVTT", ""]
        for p in pieces:
            text = f"<v {p['speaker']}>{p['text']}" if p.get("speaker") else p["text"]
            lines += [f"{fmt_time(p['start'], '.')} --> {fmt_time(p['end'], '.')}", text, ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        written.append(out)
    return written


# ---------------------------------------------------------------- main

def speakers_arg(v):
    if v.lower() == "auto":
        return 0
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more, or 'auto'")
    return n


def main():
    ap = argparse.ArgumentParser(description="Transcribe audio/video files to text with Whisper.")
    ap.add_argument("inputs", nargs="*", help="Audio/video files or folders")
    ap.add_argument("--model", default="small", help="tiny, base, small, medium, large-v3, turbo (default: small)")
    ap.add_argument("--language", default=None, help="e.g. en. Leave out to auto-detect")
    ap.add_argument("--formats", nargs="+", default=ALL_FORMATS, choices=ALL_FORMATS,
                    help="Which outputs to write (default: all)")
    ap.add_argument("--output-dir", default=None, help="Write outputs here instead of next to each file")
    ap.add_argument("--overwrite", action="store_true", help="Redo files that already have a .txt transcript")
    ap.add_argument("--no-vad", action="store_true",
                    help="Don't skip silence/music sections (try this if words go missing)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if an NVIDIA GPU is present")
    ap.add_argument("--speakers", nargs="?", const=0, type=speakers_arg, default=None, metavar="N",
                    help="Label who is speaking. Give the number of people if you know it (more accurate)")
    ap.add_argument("--names", default=None,
                    help='Comma-separated names in order of who speaks first, e.g. "Aaron,Sean"')
    ap.add_argument("--speaker-threshold", type=float, default=0.8,
                    help="Only when the number of speakers isn't given: lower = splits into more "
                         "speakers, higher = fewer (default 0.8)")
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()] if args.names else []
    if names and args.speakers is None:
        args.speakers = len(names)  # --names implies --speakers
    use_speakers = args.speakers is not None

    inputs = args.inputs
    if not inputs:
        entered = input("Drag an audio/video file or folder here and press Enter: ").strip()
        if not entered:
            sys.exit("Nothing given.")
        inputs = [entered]

    files = collect_inputs(inputs)
    if not files:
        sys.exit("No audio/video files found.")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper isn't installed. Run:  pip install faster-whisper")

    finder = None
    if use_speakers:
        print("Loading speaker detection (first run downloads ~47 MB)...")
        finder = SpeakerFinder(args.speakers, args.speaker_threshold)

    device, compute_type = pick_device()
    if args.cpu:
        device, compute_type = "cpu", "int8"

    def load(dev, ct):
        print(f"Loading model '{args.model}' on {dev.upper()} (first run downloads it)...")
        return WhisperModel(args.model, device=dev, compute_type=ct)

    try:
        model = load(device, compute_type)
    except Exception as e:
        if device != "cuda" or not is_gpu_lib_error(e):
            raise
        print(f"  ! {e}\n{GPU_LIB_HINT}")
        device = "cpu"
        model = load("cpu", "int8")

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def run(model, audio, turns):
        seg_iter, info = model.transcribe(
            audio,
            language=args.language,
            vad_filter=not args.no_vad,
            beam_size=5,
            word_timestamps=turns is not None,  # word timing lets us split mid-sentence speaker changes
        )
        if not args.language:
            print(f"  Language: {info.language} ({info.language_probability:.0%} sure)")
        label = SpeakerNames(names)
        all_words, pieces = [], []
        for seg_no, seg in enumerate(seg_iter):  # the real work happens while iterating
            pct = min(seg.end / info.duration, 1.0) if info.duration else 0
            if turns is not None and seg.words:
                words = [{"start": w.start, "end": w.end, "text": w.word, "seg": seg_no,
                          "speaker": speaker_for(w.start, w.end, turns)} for w in seg.words]
                smooth(words)
                all_words.extend(words)
                for p in pieces_from_words(words, label):
                    print(f"  {pct:4.0%} [{fmt_short(p['start'])}] {p['speaker']}: {p['text']}")
            else:
                text = seg.text.strip()
                if text:
                    pieces.append({"start": seg.start, "end": seg.end, "text": text, "speaker": None})
                    print(f"  {pct:4.0%} [{fmt_short(seg.start)}] {text}")
        if turns is not None:
            # Re-smooth across segment edges, then rebuild pieces per original segment.
            smooth(all_words)
            label = SpeakerNames(names)
            pieces, seg_words = [], []
            for i, w in enumerate(all_words):
                seg_words.append(w)
                if i == len(all_words) - 1 or all_words[i + 1]["seg"] != w["seg"]:
                    pieces.extend(pieces_from_words(seg_words, label))
                    seg_words = []
        return pieces, label

    failed = []
    for n, f in enumerate(files, 1):
        base = (out_dir or f.parent) / f.stem
        if not args.overwrite and base.with_suffix(".txt").exists() and "txt" in args.formats:
            print(f"\n[{n}/{len(files)}] Skipping {f.name} (transcript exists; use --overwrite)")
            continue

        print(f"\n[{n}/{len(files)}] {f.name}")
        t0 = time.time()
        try:
            audio = decode_audio(f)
            print(f"  Length:   {fmt_short(len(audio) / 16000)}")
            turns = None
            if finder:
                turns = finder.find(audio)
                found = len({s for _, _, s in turns})
                print(f"  Speakers: {found} found")
                if not turns:
                    print("  ! No speech found by speaker detection - transcribing without labels.")
                    turns = None
            try:
                pieces, label = run(model, audio, turns)
            except Exception as e:
                # CUDA libraries are only loaded when decoding starts, so a
                # missing cuBLAS/cuDNN shows up here rather than at load time.
                if device != "cuda" or not is_gpu_lib_error(e):
                    raise
                print(f"  ! {e}\n{GPU_LIB_HINT}")
                device = "cpu"
                del model
                model = load("cpu", "int8")
                t0 = time.time()
                pieces, label = run(model, audio, turns)

            if not pieces:
                print("  ! No speech detected.")
                continue

            written = write_outputs(pieces, base, args.formats)
            print(f"  Done in {fmt_short(time.time() - t0)} -> " + ", ".join(p.name for p in written))
            if turns is not None and not names and len(label.order) > 1:
                print("  Tip: add  --names \"First,Second,...\"  (in order of who speaks first) "
                      "to replace Speaker 1, 2...")
        except Exception as e:
            print(f"  ! Failed: {e}")
            failed.append(f.name)

    print("\nAll done." if not failed else f"\nFinished with {len(failed)} failure(s): {', '.join(failed)}")


if __name__ == "__main__":
    main()
