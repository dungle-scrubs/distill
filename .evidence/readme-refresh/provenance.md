# Opening-example provenance

`examples/cache-demo.mp4` is a six-second synthetic recording made for the
README. The table contains illustrative numbers, not measured Distill
performance. The narration says, "We measured response times. The table shows
the results." It does not speak the table's values, so recovering those values
requires reading the image.

The slide was rendered with Pillow using macOS Arial. The narration uses the
macOS Samantha voice at 145 words per minute. FFmpeg combines a still slide and
the narration into H.264 video with AAC audio. The sample contains no source
material from a user's recording.

`output.md` is an unedited markdown artifact captured from Distill 0.4.0,
pipeline 78, with the default Whisper `small` model and Tesseract. Vision was
disabled with `--no-caption-frames`. The example sets `--artifact-dir ./.distill`
so its reading command works outside a Git repository too. The README shows an excerpt of that output,
with `...` standing for omitted lines, including the untrusted-data fences.
The full artifact preserves those fences and their warning preamble.

To reproduce from a source checkout, run:

```bash
uv run distill process-local-video examples/cache-demo.mp4 --no-caption-frames
```

The JSON result reports the markdown artifact under `artifact_path`. Processing
dates and transcript boundaries can differ across environments.
