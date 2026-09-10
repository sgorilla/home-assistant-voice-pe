# Browser-controlled corpus recording

Draft preparation for [parent issue #51](https://github.com/sgorilla/pippa-home-assistant/issues/51).
**Not built, flashed, run or physically qualified.** The companion custom browser
page and receiver live in parent `wakeword-training/device-capture/` on branch
`codex/geek/51-web-device-capture`.

Use `pippa-web-recording.yaml` as an opt-in wrapper around the existing CRNN
measurement package. Without `va_client.wake_recording`, capture code is excluded
and no recording API actions are installed. Preserve the private environment
inputs used by that measurement configuration; no device secrets belong in Git.

The wrapper exposes encrypted native API actions `start_wake_recording` (sink URL,
per-take ID, ephemeral bearer token, duration) and `stop_wake_recording`. The page
calls them through its receiver. Firmware reuses the existing single WebSocket
client, 64 KiB MicTxRing and bounded sender; it does not add a parallel audio
transport, HTTP server or network write inside the microphone callback.

Corpus capture pauses inference and rejects conversational/playback activity.
It acquires a listener before stopping microWakeWord, then releases it only
after the original wake source owns its listener again, or reports a bounded
restoration failure. This is recording mode, not simultaneous wake-quality testing.

Capture uses ESPHome MicrophoneSource conversion derived from the final mWW
configuration. The base realtime package selects channel1/gain4; the CRNN
measurement package overrides that to channel0/gain1. Recording itself changes
neither setting. Final validation checks the physical microphone and mono PCM16
at 16 kHz; active capture also checks for source format/gain changes. Neither the
source configuration nor this draft establishes current installed firmware identity.

The sink receives authenticated protocol-v1 metadata, binary messages with a
big-endian 64-bit sample offset followed by PCM16LE, and a strict completion
footer. Queue/send loss, mute, source changes, inference restart, connection loss
or timeout invalidate the take. Raw bytes and manifests are retained on the host.
`callback_max_us` times only the queue callback, after source conversion; it is
not a measurement of total capture CPU or upstream I2S continuity.

The pinned WebSocket library's stop/destroy operations may wait indefinitely.
Retirement runs on one temporary low-priority worker so it cannot block the
main or microphone loop. A stalled retirement leaves the backend offline until
the original handle is safely released; no second client is created. A failed
wake restoration latches subsequent capture off until device recovery. A complete
audio footer is not evidence that backend/wake restoration succeeded afterward.

Required next validation, off the planning host:

1. Validate and compile the final private package, checking exact source derivation,
   action schema and unchanged behavior without the opt-in configuration.
2. Validate the host receiver with fake protocol/API fixtures. Check wrong tokens,
   interrupted frames, duplicate offsets, missing footers, source changes and
   create-only recording/review persistence.
3. Under the separately coordinated device trial, use a few diagnostic takes first.
   Compare wake/capture conversion; check callback/transport/queue/memory metrics,
   hardware mute, no sputter/stall, and microphone/wake/backend recovery after
   normal stop, duration limit, slow receiver, disconnect and socket-retirement
   failure. Capture setup/recovery is part of the load assessment, not only PCM send.
4. Record the actual firmware/model/frontend identities and canary disposition.
   Only then collect the new session-disjoint training/development/acceptance data.

No compilation, tests, syntax/config checks, device/API connection, audio processing,
firmware installation, Home Assistant action or recording ran during preparation.
