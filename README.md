# transcribe

Turn audio and video files into text on your own computer with Whisper (via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)). It can also label
who is speaking. Nothing is uploaded anywhere.

## Setup

Requires Python 3.9+.

```
pip install -r requirements.txt
```

The requirements file installs everything except the GPU libraries. If you have
an NVIDIA graphics card, also install these to use it (it's much faster):

```
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

Without them the script still works. It reports that the libraries are missing
and runs on the CPU.

## Usage

```
python transcribe.py recording.mp3
python transcribe.py "D:\Recordings"                  # every audio/video file in a folder
python transcribe.py recording.mp3 --model turbo      # best choice with an NVIDIA GPU
python transcribe.py                                  # asks for a file - drag it into the window
```

### Who's speaking

```
python transcribe.py meeting.mp3 --speakers                        # works out how many people
python transcribe.py meeting.mp3 --speakers 2                      # you know it's 2 (more accurate)
python transcribe.py meeting.mp3 --speakers 2 --names "Aaron,Sean"
```

Give `--names` in the order people **first speak** in the recording.

### Output

Files are written next to each recording (or to `--output-dir`):

| File | Contents |
|---|---|
| `name.txt` | Transcript. One paragraph per speaker turn with `--speakers` |
| `name.timestamps.txt` | Same, with `[mm:ss]` at the start of each line |
| `name.srt` / `name.vtt` | Subtitles (speaker names included) |

Recordings that already have a `.txt` transcript are skipped, so you can re-run
on a folder. Add `--overwrite` to redo them.

## Options

| Option | What it does |
|---|---|
| `--model` | `tiny`, `base`, `small` (default), `medium`, `large-v3`, `turbo` |
| `--language en` | Skip language auto-detection |
| `--formats txt srt` | Only write some outputs (`txt`, `srt`, `vtt`, `timestamps`) |
| `--output-dir DIR` | Write outputs somewhere else |
| `--overwrite` | Redo files that already have a transcript |
| `--no-vad` | Don't skip silence/music. Try this if words go missing |
| `--cpu` | Don't use the GPU |
| `--speakers [N]` | Label speakers. Give the number of people if known |
| `--names "A,B"` | Names instead of Speaker 1, 2… (in order of first speaking) |
| `--speaker-threshold` | When the count isn't given: lower = more speakers, higher = fewer (default 0.8) |

## Notes

- Models download automatically on first use and are cached: Whisper models from
  Hugging Face, speaker models (~47 MB) from the
  [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) into
  `~/.cache/transcribe-speakers`.
- Speaker detection runs on the CPU, at roughly 6 seconds per minute of audio.
  It struggles with overlapping speech and very short interjections.
- The script decodes audio itself rather than using faster-whisper's decoder,
  which breaks with newer PyAV versions (`unexpected keyword argument 'metadata_errors'`).
