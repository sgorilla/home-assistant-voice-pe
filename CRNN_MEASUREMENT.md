# Pippa CRNN target measurement overlay

`pippa-crnn-measurement.yaml` is an issue #19 physical-feasibility overlay, not
a production model promotion. It is based on firmware commit
`8babb3414fab843b16af7a7d68418dbf637c6262` and keeps the live Voice PE behavior
except for the wake-model/runtime substitutions required by the measurement.

## Bound inputs

- Candidate: `models/pippa/candidate-16x8.tflite`, 165,960 bytes, SHA-256
  `5841499a1a0420d4910a2f6b24e9e29ff271f2abeff65c64fd64f7d61d9bd770`.
- Descriptor: `models/pippa/pippa-16x8.json`, SHA-256
  `733ecd2e44c06fb6e4ecaf8364337558e2b6fea5d070dd1b70a7b687a5bcb2d2`.
- ABI: int8 features `[1,3,40]`, int8 recurrent state `[1,112]`, int8
  probability, 10 ms feature step, cutoff byte 183, sliding window 5.
- External-state `micro_wake_word` overlay: exact files reviewed at
  `pippa-home-assistant` commit
  `2ae0523c21d7c2aa68bffc1e73f86e6c62b458da`.
- `va_client`: `8babb3414fab843b16af7a7d68418dbf637c6262`.
- Voice-kit component: `0579e7b9d8504264719c593474c85447253c9dc1`.

The descriptor's 200,000-byte tensor arena remains a conservative manifest
fallback. Host fixture allocation and quality results are not substitutes for
the separately reported physical Voice PE measurements.

## Build contract

The build pins `sgorilla/esp-tflite-micro` commit
`6e5817c54e0b325d9ebbda64d3a3f71034a9e9ba`, derived from registry component
`espressif/esp-tflite-micro=1.3.3~1` at upstream commit
`78e5532e682d3863a3c2985c3a74eed0a9ebaa61`. The functional patch started from
these exact source hashes:

- `tensorflow/lite/micro/kernels/fully_connected_common.cc`:
  `388bf81053aac4ce6d031e4995631deebeda020b3e02a9a80351074ebcfa960d`
- `tensorflow/lite/micro/kernels/esp_nn/fully_connected.cc`:
  `c23b68b0dceb783a37fa9eda5c3a4c4b53ae8977abde0f8f32c67f7ae2acce98`

The reviewed host patch originally produced these hashes:

- `fully_connected_common.cc`:
  `a91b2f52731c28c0a3dd8bb6740f7a1780d6a70ab5b3ab0de36abb6b85edcc5a`
- `esp_nn/fully_connected.cc`:
  `b71ca8533e16a352957b7b35b8b7ad2329e5eee1ad242884bea7fb03f94857d5`

Those bytes pass the x86 host fixture proof but fail the Xtensa compile because
`int32_t` and `int` are distinct pointer types on that target. A target-portable
derivative applies the same `reinterpret_cast<const int *>` already used by the
stock generic TFLM path at the two new per-channel call sites. It preserves the
common-file hash above and changes the ESP-NN wrapper hash to:

- `esp_nn/fully_connected.cc`:
  `3f2de506cf4a2bcc29c509744d20ef5671b74b595149bb587cd32f6028127b75`

Its preparation manifest SHA-256 is
`b58d303de3ee92d8d6dc500ad6d3d879539f9bd8431b83e8b6cda13afbd2fb39`;
the target-portable generator SHA-256 is
`8f7171c6ff5dab2cdb0c2ebfb8dbe2b58ee96feb99755c257f846ec754fb85ec`.
A minimal ESP32-S3 Xtensa build consumed those exact sources and passed. A
complete Voice PE build then consumed the same target-portable bytes and the
selective kernel policy and passed under ESPHome 2026.8.2 / ESP-IDF 5.5.5. The
off-device validation build used placeholder credentials and is deliberately
not flashable: its 3,375,840-byte
OTA image has SHA-256
`34415508636484e2950189b0a93a8a7442380aa00973f5f0b96dc2b886d3d33e`.
It used 176,635 of 341,760 linked DIRAM bytes (51.7%) and left 58% of the
application partition free. The live ESPHome Builder rebuilt the bound sources
with the existing device credentials for the physical canary.

Commit `5c2441292b07c9ee3a6286674d0036e38c32152e` contains only the target-portable
fully-connected math additions and their provenance. Its child commit
`6e5817c54e0b325d9ebbda64d3a3f71034a9e9ba` records the physical kernel policy:
only the Conv2D and DepthwiseConv2D wrappers receive `ESP_NN=1`; AveragePool2D
and the remaining wrappers execute their reference fallbacks. The measurement
runtime deliberately omits its former global `ESP_NN` build flag so it cannot
override that per-source policy.

## Physical kernel result

The same pinned 20-step closed-loop oracle was run on each physical image:

1. The stock all-ESP-NN image diverged from the expected probability/state
   trajectory.
2. The all-reference image matched all 20 probability bytes and all 112 state
   bytes at every step, but roughly 45 ms invocations exceeded the 30 ms model
   budget and repeatedly reset the microphone ring.
3. Conv2D-only acceleration remained 20/20 byte-exact and restored real-time
   processing.
4. Conv2D plus DepthwiseConv2D acceleration also remained 20/20 byte-exact and
   sustained 167 invocations per five-second score window without an observed
   ring reset in the capture.

For this graph, dependency revision, and ESP32-S3 target, the bisection isolates
the arithmetic divergence to the optimized AveragePool2D path. The final
selective image is the durable measurement policy.

The final selective image subsequently completed a 10,000-inference physical
runtime window (cumulative invocation count 200,000): p50 16.5 ms, p95 17.5 ms,
p99 18.0 ms, maximum 48.965 ms, and 3/10,000 invocations over the 30 ms runtime
budget. The percentile buckets were not censored, no report was dropped, and
`ring_full_total` remained zero. The maximum observed invocation-start gap was
40.699 ms. The raw cadence counter classified 7,485/9,997 intervals as greater
than exactly 30,000 us; because that threshold has no scheduling tolerance, it
must not be interpreted as 7,485 dropped feature windows. This was a quiet/idle
runtime measurement, not a worst-load speech-plus-playback qualification.

Physical speech confirms that the model can trigger the real wake/session path,
but the candidate is not ready for promotion. A rough activation sweep at the
byte-183 operating point produced about one observed activation in ten intended
utterances. A separate non-triggering score capture at byte 230 ranged from
5/255 to 201/255; only two five-second aggregate windows exceeded byte 183.
Higher-pitched delivery appeared more likely to score, but not reliably. A
subsequent state-reset trial was inconclusive because the five intended
utterances did not arrive at comparable microphone levels. These are
model/input-domain and test-design findings, not remaining kernel-parity
failures, and they do not support a final threshold choice.

## Intended model state

The package merge removes and re-adds the live model entries in this order:

1. `pippa` — enabled, internal measurement candidate.
2. `stop` — retained and managed by the existing reply/follow-up phase logic.
3. `okay_nabu` — retained but explicitly disabled.

The familiar Okay Nabu ID is retained for a reversible comparison. The
API-connect actions override any persisted model state by enabling Pippa and
disabling Okay Nabu explicitly. The Okay Nabu-specific sensitivity selector is
removed from this fixed-cutoff measurement configuration so it cannot
accidentally imply that it tunes Pippa.

## Measurement instrumentation

Only this measurement overlay defines `PIPPA_CRNN_METRICS` and
`PIPPA_CRNN_SELF_TEST`. At boot, the self-test runs records 184 through 203 from
the pinned host/reference trajectory before microphone inference starts. It
retains results, recreates the interpreter so fixture state cannot leak into
live inference, and reports from the main loop after logging is available.

For the external-state model, `CRNN_ARENA` reports the tensor and
resource-variable arenas' used and allocated bytes once. `CRNN_SCORE` reports a
five-second aggregate of microphone peak, feature range/change, raw score, and
sliding-average score. `CRNN_PERF` is handed from the inference task to the main
loop and logged after each 10,000 warm invocations; there is no per-invoke
logging or allocation.

`runtime_*` measures only `MicroInterpreter::Invoke()` plus recurrent-state
feedback. `runtime_budget_us` is the model stride multiplied by the 10 ms
feature step (30,000 us here); `runtime_budget_overruns` therefore describes
inference-runtime budget overruns, not missed audio cadence. Cold first invokes
after model loads are excluded from the histogram and reported separately.

The 128-bin histogram uses 500 us bins. `p50_bucket_us`, `p95_bucket_us`, and
`p99_bucket_us` identify the selected bin boundary; the final bin is open-ended
at 63,500 us. Bits 0, 1, and 2 of `percentile_censored_mask` respectively mark
a p50, p95, or p99 that landed in that censored final bin.

True cadence evidence uses consecutive external-state invocation start times.
`cadence_interval_n` is the number of valid intervals, `cadence_max_gap_us` is
their raw maximum, and `cadence_gap_over_threshold` counts intervals strictly
greater than the separately logged `cadence_threshold_us` (30,000 us here).
The previous-start anchor is cleared on every streaming-state reset, so an
interval never crosses a known stream boundary. `ring_full_total` remains the
cumulative count of rejected microphone chunks that triggered a ring reset.

The Home Assistant diagnostic controls can change Pippa's cutoff or request a
thread-safe recurrent-state reset without rebuilding. The cutoff is restored
to the descriptor's byte-183 default after every reboot.
