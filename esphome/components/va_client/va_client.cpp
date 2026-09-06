#include "va_client.h"
#include "automation.h"

#include "esphome/core/log.h"
#include "esphome/components/audio/audio.h"

#include <algorithm>
#include <cstring>

#include <esp_websocket_client.h>
#include <esp_event.h>
#include <esp_heap_caps.h>

namespace esphome {
namespace va_client {

static const char *const TAG = "va_client";

// Free-function trampoline. esp-idf event registration takes a C function
// pointer; we recover the VaClient* from the user_data slot.
static void va_ws_event_handler(void *handler_args, esp_event_base_t /*base*/, int32_t event_id, void *event_data) {
  auto *self = static_cast<VaClient *>(handler_args);
  if (self == nullptr)
    return;
  self->on_ws_event(event_id, event_data);
}

// Parse the unsigned integer immediately following `key` in `msg` (key includes
// the quotes + colon, e.g. "\"follow_up_ms\":"). Returns true and sets `out` if a
// run of digits was found right after the key. Keeps us out of a JSON parser for
// the few small ints the backend sends in `hello`.
static bool parse_uint_after_key(const std::string &msg, const char *key, uint32_t &out) {
  size_t p = msg.find(key);
  if (p == std::string::npos)
    return false;
  p += std::strlen(key);
  uint32_t v = 0;
  bool any = false;
  while (p < msg.size() && msg[p] >= '0' && msg[p] <= '9') {
    v = v * 10u + static_cast<uint32_t>(msg[p] - '0');
    p++;
    any = true;
  }
  if (!any)
    return false;
  out = v;
  return true;
}

void VaClient::setup() {
  ESP_LOGCONFIG(TAG, "Setting up VA Client...");

  if (this->mic_ != nullptr) {
    this->mic_->add_data_callback(
        [this](const std::vector<uint8_t> &data) { this->on_mic_data_(data); });
  } else {
    ESP_LOGE(TAG, "Microphone not configured");
  }

  // Allocate the audio ring buffer in PSRAM (8 MB available, internal RAM
  // is only 320 KB and we don't want to starve wifi/mww).
  this->audio_buf_ = static_cast<uint8_t *>(
      heap_caps_malloc(kAudioBufBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (this->audio_buf_ == nullptr) {
    ESP_LOGE(TAG, "Failed to allocate %u-byte audio buffer in PSRAM", (unsigned) kAudioBufBytes);
  } else {
    ESP_LOGCONFIG(TAG, "Allocated %u-byte audio ring buffer in PSRAM", (unsigned) kAudioBufBytes);
  }

  // Allocate the mic pre-roll ring in PSRAM (kPreRollMs of 16 kHz int16 mono).
  this->preroll_capacity_samples_ = (size_t) kPreRollMs * (kMicSampleRate / 1000);
  this->preroll_buf_ = static_cast<int16_t *>(
      heap_caps_malloc(this->preroll_capacity_samples_ * sizeof(int16_t),
                       MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (this->preroll_buf_ == nullptr) {
    ESP_LOGE(TAG, "Failed to allocate %u-sample mic pre-roll buffer in PSRAM",
             (unsigned) this->preroll_capacity_samples_);
    this->preroll_capacity_samples_ = 0;
  } else {
    ESP_LOGCONFIG(TAG, "Allocated %u ms mic pre-roll buffer in PSRAM (%u samples)",
                  (unsigned) kPreRollMs, (unsigned) this->preroll_capacity_samples_);
  }

  // Network I/O must not run in the high-priority I2S callback: va_client's
  // callback is registered before microWakeWord's callback, so a blocked send
  // also withholds that same frame from wake inference. The mic task writes to
  // this PSRAM queue; loop() copies and sends one bounded chunk at a time.
  this->mic_tx_storage_ = static_cast<uint8_t *>(
      heap_caps_malloc(kMicTxBufferBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  this->mic_tx_send_buf_ = static_cast<uint8_t *>(
      heap_caps_malloc(kMicTxSendChunkBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (this->mic_tx_storage_ == nullptr || this->mic_tx_send_buf_ == nullptr) {
    ESP_LOGE(TAG, "Failed to allocate microphone TX queue (%u + %u bytes)",
             (unsigned) kMicTxBufferBytes, (unsigned) kMicTxSendChunkBytes);
    if (this->mic_tx_storage_ != nullptr) {
      heap_caps_free(this->mic_tx_storage_);
      this->mic_tx_storage_ = nullptr;
    }
    if (this->mic_tx_send_buf_ != nullptr) {
      heap_caps_free(this->mic_tx_send_buf_);
      this->mic_tx_send_buf_ = nullptr;
    }
    this->mark_failed();
    return;
  }
  this->mic_tx_ring_.set_storage(this->mic_tx_storage_, kMicTxBufferBytes);
  ESP_LOGCONFIG(TAG, "Allocated %u-byte nonblocking microphone TX queue in PSRAM",
                (unsigned) kMicTxBufferBytes);

  // Tell the resampler what format we'll feed it. The resampler converts to
  // its yaml-configured output format (48k 16-bit) before passing to the
  // mixer → i2s leaf. Start the speaker task once so play() calls just push
  // into its ring buffer instead of racing to re-create the i2s channel.
  if (this->speaker_ != nullptr) {
    audio::AudioStreamInfo info(/*bits_per_sample=*/16, /*channels=*/1, /*sample_rate=*/24000);
    this->speaker_->set_audio_stream_info(info);
    this->speaker_->start();
  }

  this->connect_();
}

void VaClient::loop() {
  // Always service input transport before playback. Some playback paths return
  // early while priming the speaker chain; microphone audio must not wait on
  // that unrelated output state.
  this->drain_mic_tx_();
  this->log_mic_health_();

  // Drain the audio ring buffer into the speaker. speaker.play() accepts
  // only what fits in its own ring (returns the count actually queued).
  if (this->speaker_ != nullptr && this->audio_buf_ != nullptr) {
    // Snapshot ring state under the lock — head/tail/fill are all
    // mutated from the WS task on the other core.
    portENTER_CRITICAL(&this->ring_mux_);
    size_t head = this->audio_head_;
    size_t tail = this->audio_tail_;
    size_t fill = this->audio_fill_;
    portEXIT_CRITICAL(&this->ring_mux_);
    if (fill > 0) {
      // Resampler cold-start SILENCE-PRIME (crackle fix). The resampler does NOT
      // idle-timeout (verified vs ESPHome source): resample(stop_gracefully=false)
      // never returns FINISHED, and its output mixer-source is timeout:never, so the
      // chain stays WARM between normal replies. It goes COLD only after an explicit
      // `speaker.stop: media_resampling_speaker` (yaml interrupt / "stop" word / wake /
      // follow-up), which tears the task down to STATE_STOPPED. The next reply then
      // cold-starts a fresh AudioResampler whose windowed-sinc FIR begins from a zero
      // state → a startup-transient click. A PSRAM prebuffer can't fix it (the transient
      // is downstream of the ring). Fix: when cold, feed kChainPrimeMs of silence BEFORE
      // the first real sample so the FIR settles to a clean zero output first. We detect
      // "cold" two ways: the resampler actually reporting is_stopped() (true exactly
      // post-speaker.stop — the precise signal) OR, as a backup, nothing fed for
      // > kChainColdMs. is_stopped() closes the window the timer alone misses: a reply
      // whose audio lands < kChainColdMs after a speaker.stop (the residual click). A
      // needless prime on an already-warm chain is harmless (60ms of silence); both
      // signals are only ever true at a real cold reply-start, never mid-speech. The
      // real audio waits safely in PSRAM (and builds a small cushion) until priming done.
      {
        const uint32_t now_ms = millis();
        const bool resampler_cold = this->speaker_->is_stopped() ||
                                    this->last_fed_ms_ == 0 ||
                                    (now_ms - this->last_fed_ms_) > kChainColdMs;
        if (this->chain_prime_remaining_ == 0 && resampler_cold) {
          this->chain_prime_remaining_ =
              (size_t) kChainPrimeMs * (kPlaybackSampleRate / 1000) * 2;  // ms→bytes (mono 16-bit)
          ESP_LOGD(TAG, "resampler cold — priming %u bytes of silence before reply",
                   (unsigned) this->chain_prime_remaining_);
        }
        if (this->chain_prime_remaining_ > 0) {
          static const uint8_t kSilence[480] = {0};  // 10ms @24k mono16; fed in chunks
          size_t want = std::min(this->chain_prime_remaining_, sizeof(kSilence));
          size_t fed = this->speaker_->play(kSilence, want);
          if (fed > 0) {
            this->chain_prime_remaining_ -= fed;
            this->last_fed_ms_ = now_ms;  // count silence as "fed" so cold-check clears
          }
          // Hold real-audio drain until the chain is warmed. Real audio stays in
          // PSRAM. Re-enter loop() next tick to continue/finish priming.
          return;
        }
      }
      // Jitter buffer priming gate. After the ring was empty (reply start or a
      // post-underflow gap) hold playback until either the prebuffer cushion has
      // accumulated (fill >= target) or a deadline elapses (so real-time, non-
      // burst audio still starts promptly). Holding here lets the downstream
      // chain start with a cushion so a network gap doesn't dry it out → no
      // crackle. Skipped entirely when playback_prebuffer_ms_ == 0 (disabled).
      if (this->playback_priming_) {
        const size_t target =
            (size_t) this->playback_prebuffer_ms_ * (kPlaybackSampleRate / 1000) * 2;
        if (fill >= target ||
            (millis() - this->prime_started_ms_) >= this->playback_prebuffer_ms_) {
          this->playback_priming_ = false;
          ESP_LOGD(TAG, "prebuffer ready (%u bytes) — playback start", (unsigned) fill);
        } else {
          return;  // keep accumulating; don't drain (and don't false-flag underrun)
        }
      }
      // Detector 3: downstream underrun. If the resampler/mixer/i2s chain
      // ran out of bytes to play while we *still* have PSRAM queued,
      // something hiccupped downstream — the user hears silence or a
      // brief stuck-sample glitch. Log the first occurrence per reply so
      // we know whether bad audio in a turn correlates with this.
      if (!this->underrun_logged_this_turn_ && !this->speaker_->has_buffered_data()) {
        ESP_LOGW(TAG, "downstream underrun: %u bytes queued in PSRAM but speaker chain is dry",
                 (unsigned) fill);
        this->underrun_logged_this_turn_ = true;
      }
      // Contiguous slice we can hand to play() without copying: from head
      // to either the end of the buffer or the tail.
      size_t contiguous = (head < tail) ? (tail - head) : (kAudioBufBytes - head);
      if (contiguous > fill)
        contiguous = fill;
      // play() runs OUTSIDE the critical section: it can take milliseconds
      // (resampler ring may be full, mixer blocks). Holding ring_mux_
      // across it would block the writer and cause audio underrun.
      size_t accepted = this->speaker_->play(this->audio_buf_ + head, contiguous);
      if (accepted > 0) {
        this->last_fed_ms_ = millis();  // keep the chain "warm" for cold-detection
        portENTER_CRITICAL(&this->ring_mux_);
        this->audio_head_ = (this->audio_head_ + accepted) % kAudioBufBytes;
        this->audio_fill_ -= accepted;
        portEXIT_CRITICAL(&this->ring_mux_);
        static uint32_t dbg_last = 0;
        uint32_t now = millis();
        if (now - dbg_last >= 500) {
          ESP_LOGD(TAG, "drained %u bytes (%u still queued)", (unsigned) accepted,
                   (unsigned) (fill - accepted));
          dbg_last = now;
        }
      }
    }
  }
  // If a follow-up window was deferred while audio was draining, wait for
  // the downstream chain (resampler + mixer + i2s + DAC tail) to actually
  // finish playing before firing the deferred LED-idle / chime trigger.
  // Just because our PSRAM queue is empty doesn't mean the user has heard
  // the audio yet — and an "сейчас посмотрю" preamble before a tool call
  // would drain the ring mid-turn, so we can't act on audio_fill_==0
  // alone.
  //
  // Primary signal: speaker_->is_stopped(). The resampling speaker
  // transitions to STOPPED only after every byte we wrote has actually
  // gone out through the i2s pipeline.
  //
  // Fallback: kSpeakerStopTimeoutMs (3 s). If something wedges and the
  // speaker never reports STOPPED, we still progress so the LED doesn't
  // lock in `replying`.
  if (this->followup_pending_ && this->audio_fill_ == 0 &&
      !this->waiting_for_speaker_stop_) {
    this->waiting_for_speaker_stop_ = true;
    this->speaker_stop_wait_started_ms_ = millis();
  }
  if (this->waiting_for_speaker_stop_) {
    // Use has_buffered_data() instead of is_stopped(): the resampler only
    // transitions to STATE_STOPPED once its downstream (mixer source)
    // reports stopped, but our mixer sources are configured `timeout:
    // never` and stay RUNNING forever, so is_stopped() would never fire
    // and we'd always hit the fallback. has_buffered_data() walks the
    // chain (resampler ring + mixer source ring) and returns false as
    // soon as both have drained — exactly what we want.
    //
    // Note: this does *not* cover the i2s 500ms ring + ~100ms DAC tail
    // downstream of the mixer. We fire ~500ms before true silence. For
    // the LED that's imperceptible; for the request_follow_up chime,
    // yaml's wait_until !is_announcing + i2s tail delay already absorbs
    // any small overlap with the fading TTS tail.
    const bool speaker_drained =
        (this->speaker_ != nullptr) && !this->speaker_->has_buffered_data();
    const bool timed_out =
        (millis() - this->speaker_stop_wait_started_ms_) >= kSpeakerStopTimeoutMs;
    if (speaker_drained || timed_out) {
      if (timed_out && !speaker_drained) {
        ESP_LOGW(TAG,
                 "speaker still had buffered data after %u ms — "
                 "proceeding anyway (fallback)",
                 (unsigned) kSpeakerStopTimeoutMs);
      }
      this->waiting_for_speaker_stop_ = false;
      const bool was_request = this->request_follow_up_pending_;
      this->followup_pending_ = false;
      this->request_follow_up_pending_ = false;
      if (was_request) {
        // Request-driven path: speaker has drained, now hand off to yaml
        // for the chime → wait_until !is_announcing → commit_followup_mic
        // sequence (announcement lane is separate from the TTS lane we
        // just waited on, so the chime won't collide with our tail).
        this->open_followup_window_(0);  // emit deferred LED idle + latency log; no mic
        this->followup_armed_ = true;
        for (auto *t : this->followup_opened_triggers_) {
          t->trigger();
        }
      } else {
        // Natural-idle path: emit the deferred LED idle, and — if the backend
        // configured a follow-up window (followup_ms_ > 0) — open the mic for
        // that long so the user can answer back without a wake word.
        this->open_followup_window_(this->followup_ms_);
      }
    }
  }
}

void VaClient::connect_() {
  if (this->ws_handle_ != nullptr) {
    // Already initialised; just (re)start. A synchronous start failure must
    // reschedule — otherwise the reconnect chain stalls silently and the
    // device stays offline until a reboot.
    esp_err_t err = esp_websocket_client_start(static_cast<esp_websocket_client_handle_t>(this->ws_handle_));
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "esp_websocket_client_start (restart) failed: %d — rescheduling", (int) err);
      this->schedule_reconnect_();
    }
    return;
  }

  esp_websocket_client_config_t cfg = {};
  cfg.uri = this->url_.c_str();
  cfg.disable_auto_reconnect = true;  // we drive reconnects ourselves with exponential backoff
  cfg.reconnect_timeout_ms = 5000;    // ignored because disable_auto_reconnect=true

  esp_websocket_client_handle_t handle = esp_websocket_client_init(&cfg);
  if (handle == nullptr) {
    ESP_LOGE(TAG, "esp_websocket_client_init failed");
    this->schedule_reconnect_();
    return;
  }
  this->ws_handle_ = handle;

  esp_err_t err = esp_websocket_register_events(handle, WEBSOCKET_EVENT_ANY, va_ws_event_handler, this);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_websocket_register_events failed: %d", (int) err);
  }

  ESP_LOGI(TAG, "Connecting to %s", this->url_.c_str());
  err = esp_websocket_client_start(handle);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_websocket_client_start failed: %d", (int) err);
    this->schedule_reconnect_();
  }
}

void VaClient::schedule_reconnect_() {
  // esp_websocket_client emits multiple events per failure (DISCONNECTED,
  // CLOSED, sometimes ERROR). Coalesce them into a single reconnect.
  if (this->reconnect_pending_) {
    return;
  }
  this->reconnect_pending_ = true;

  // One per *failure* (coalesced), not per individual WS event. Once we
  // cross the threshold fire the audible-error trigger exactly once until
  // a successful connect resets the counter.
  this->consecutive_failures_++;
  if (this->consecutive_failures_ >= kRepeatedFailureThreshold &&
      !this->repeated_failure_fired_) {
    this->repeated_failure_fired_ = true;
    ESP_LOGW(TAG, "%u consecutive reconnect failures — firing on_repeated_failure",
             (unsigned) this->consecutive_failures_);
    this->defer([this]() {
      for (auto *t : this->repeated_failure_triggers_) {
        t->trigger();
      }
    });
  }

  uint32_t delay = this->reconnect_delay_ms_;
  ESP_LOGI(TAG, "Scheduling reconnect in %u ms", (unsigned) delay);
  // Backoff schedule: 1s -> 2s -> 5s -> 10s (capped).
  if (this->reconnect_delay_ms_ < 2000) {
    this->reconnect_delay_ms_ = 2000;
  } else if (this->reconnect_delay_ms_ < 5000) {
    this->reconnect_delay_ms_ = 5000;
  } else {
    this->reconnect_delay_ms_ = 10000;
  }
  this->set_timeout("va_reconnect", delay, [this]() {
    this->reconnect_pending_ = false;
    this->connect_();
  });
}

void VaClient::fail_control_channel_(const char *operation) {
  // A timed-out control frame leaves backend protocol state unknown. Fail
  // closed immediately. Pinned esp_websocket_client 1.7.0 has no nonblocking
  // abort API: both stop() and close() eventually wait with portMAX_DELAY, so
  // invoking either as recovery could recreate the permanent main-loop hang
  // this path is designed to prevent. A real disconnect enters the normal
  // reconnect path; the dirty flag is cleared only after the next connection's
  // mandatory start marker is fully sent. Until then the device stays alive
  // and visibly offline instead of accepting audio on uncertain protocol state.
  const bool already_dirty =
      this->control_channel_dirty_.exchange(1, std::memory_order_acq_rel) != 0;
  this->ws_connected_.store(false, std::memory_order_release);
  this->session_starting_.store(false, std::memory_order_release);
  this->streaming_.store(false, std::memory_order_release);
  this->clear_mic_tx_();

  if (already_dirty)
    return;

  ESP_LOGE(TAG, "%s left control state uncertain — channel held fail-closed",
           operation);
  this->defer([this]() {
    this->cancel_timeout("va_no_speech");
    this->cancel_timeout("va_followup");
    this->cancel_timeout("va_followup_open");
    this->cancel_timeout("va_tts_tail");
    this->followup_pending_ = false;
    this->waiting_for_speaker_stop_ = false;
    this->request_follow_up_pending_ = false;
    this->followup_armed_ = false;
    this->idle_emit_pending_ = false;
    this->current_phase_.store(static_cast<uint8_t>(Phase::IDLE));
    this->fire_phase_led_("idle");
  });
}

void VaClient::on_ws_event(int32_t event_id, void *event_data) {
  auto *data = static_cast<esp_websocket_event_data_t *>(event_data);
  switch (event_id) {
    case WEBSOCKET_EVENT_CONNECTED: {
      ESP_LOGI(TAG, "WS connected");
      // Do not publish a usable connection until its mandatory clean-start
      // marker is fully written. A previous control failure may have left this
      // flag dirty across the disconnect specifically so no interim DATA can
      // resurrect the old protocol state.
      this->ws_connected_.store(false, std::memory_order_release);
      const char start_msg[] = "{\"type\":\"start\"}";
      auto handle = static_cast<esp_websocket_client_handle_t>(this->ws_handle_);
      const int start_sent = esp_websocket_client_send_text(
          handle, start_msg, sizeof(start_msg) - 1,
          pdMS_TO_TICKS(kWsControlSendTimeoutMs));
      if (start_sent != static_cast<int>(sizeof(start_msg) - 1) ||
          !esp_websocket_client_is_connected(handle)) {
        ESP_LOGW(TAG, "start marker send failed (sent=%d expected=%u)",
                 start_sent, (unsigned) (sizeof(start_msg) - 1));
        this->fail_control_channel_("start marker failure");
        break;
      }

      this->control_channel_dirty_.store(0, std::memory_order_release);
      this->ws_connected_.store(true, std::memory_order_release);
      this->reconnect_delay_ms_ = 1000;  // reset backoff on a clean open
      // Don't reset the failure counter / fired flag yet — a flap-and-die
      // link would spam chimes. Only re-arm after the connection has held
      // for kStableConnectionMs without a disconnect.
      this->set_timeout("va_stable_connection", kStableConnectionMs, [this]() {
        if (this->ws_connected_.load(std::memory_order_acquire)) {
          this->consecutive_failures_ = 0;
          this->repeated_failure_fired_ = false;
          ESP_LOGD(TAG, "WS stable for %u ms — error chime re-armed",
                   (unsigned) kStableConnectionMs);
        }
      });
      this->set_phase_("idle");
      break;
    }
    case WEBSOCKET_EVENT_DATA: {
      if (data == nullptr || data->data_ptr == nullptr || data->data_len <= 0)
        break;
      // op_code: 0x01 = text, 0x02 = binary, 0x00 = continuation of the prior
      // frame. esp_websocket_client splits long messages, so we must track the
      // type from the first chunk and feed continuations to the same handler.
      uint8_t op = data->op_code;
      if (op == 0x01) {
        this->last_data_was_binary_ = false;
        this->handle_text_(data->data_ptr, static_cast<size_t>(data->data_len));
      } else if (op == 0x02) {
        this->last_data_was_binary_ = true;
        this->handle_binary_(reinterpret_cast<const uint8_t *>(data->data_ptr),
                             static_cast<size_t>(data->data_len));
      } else if (op == 0x00) {
        // Continuation. Route based on the type of the in-flight message.
        if (this->last_data_was_binary_) {
          this->handle_binary_(reinterpret_cast<const uint8_t *>(data->data_ptr),
                               static_cast<size_t>(data->data_len));
        } else {
          this->handle_text_(data->data_ptr, static_cast<size_t>(data->data_len));
        }
      }
      break;
    }
    case WEBSOCKET_EVENT_DISCONNECTED:
    case WEBSOCKET_EVENT_CLOSED:
    case WEBSOCKET_EVENT_ERROR: {
      if (this->ws_connected_.load(std::memory_order_acquire)) {
        ESP_LOGW(TAG, "WS disconnected (event %d)", (int) event_id);
      }
      this->ws_connected_.store(false, std::memory_order_release);
      this->session_starting_.store(false, std::memory_order_release);
      this->streaming_.store(false, std::memory_order_release);
      this->clear_mic_tx_();
      this->cancel_timeout("va_no_speech");
      // Connection broke before the stability window elapsed — keep the
      // failure counter and the fired flag. A flapping link won't earn
      // a fresh chime.
      this->cancel_timeout("va_stable_connection");
      this->set_phase_("idle");
      this->schedule_reconnect_();
      break;
    }
    default:
      break;
  }
}

void VaClient::handle_text_(const char *data, size_t len) {
  std::string msg(data, len);
  ESP_LOGD(TAG, "WS text: %s", msg.c_str());

  if (msg.find("\"type\":\"error\"") != std::string::npos) {
    ESP_LOGW(TAG, "Server reported error: %s", msg.c_str());
    this->session_starting_.store(false, std::memory_order_release);
    this->streaming_.store(false, std::memory_order_release);
    this->clear_mic_tx_();
    this->cancel_timeout("va_no_speech");
    // Without an audible cue the user just sees the LED go idle and
    // assumes the assistant ignored them. Reuse the on_repeated_failure
    // trigger — it already plays error_cloud_expired and the failure
    // mode is identical from the user's perspective ("something went
    // wrong, try again"). We deliberately don't bump consecutive_failures_
    // here; that counter is for WS reachability, not server-side errors.
    // Marshalled via defer(): handle_text_ runs on the WS task and ESPHome
    // triggers (→ the yaml play_sound script) are not thread-safe.
    this->defer([this]() {
      for (auto *t : this->repeated_failure_triggers_) {
        t->trigger();
      }
    });
    this->set_phase_("idle");
    return;
  }

  if (msg.find("\"type\":\"hello\"") != std::string::npos) {
    // Handshake ack from the backend. It may carry follow-up tuning so the
    // device behaviour is configurable from the add-on (no reflash): a reconnect
    // after an add-on restart re-reads these.
    //   "follow_up_ms":N            — how long the mic stays open after a reply
    //                                 so the user can answer without a wake word.
    //   "follow_up_open_delay_ms":N — delay before that mic opens, to let the
    //                                 reply's i2s/DAC tail finish playing out.
    uint32_t v = 0;
    if (parse_uint_after_key(msg, "\"follow_up_ms\":", v)) {
      if (v > kFollowupMsMax)
        v = kFollowupMsMax;
      this->followup_ms_ = v;
      ESP_LOGI(TAG, "hello: follow-up window = %u ms (%s)", (unsigned) v,
               v == 0 ? "disabled, turn-based" : "mic stays open after replies");
    }
    if (parse_uint_after_key(msg, "\"follow_up_open_delay_ms\":", v)) {
      if (v > kFollowupOpenDelayMaxMs)
        v = kFollowupOpenDelayMaxMs;
      this->followup_open_delay_ms_ = v;
      ESP_LOGI(TAG, "hello: follow-up mic-open delay = %u ms", (unsigned) v);
    }
    if (parse_uint_after_key(msg, "\"wake_open_delay_ms\":", v)) {
      if (v > kFollowupOpenDelayMaxMs)  // reuse the same 5 s ceiling
        v = kFollowupOpenDelayMaxMs;
      this->wake_open_delay_ms_ = v;
      ESP_LOGI(TAG, "hello: wake mic-open delay = %u ms", (unsigned) v);
    }
    if (parse_uint_after_key(msg, "\"playback_prebuffer_ms\":", v)) {
      if (v > kPlaybackPrebufferMaxMs)
        v = kPlaybackPrebufferMaxMs;
      this->playback_prebuffer_ms_ = v;
      ESP_LOGI(TAG, "hello: playback prebuffer (jitter buffer) = %u ms (%s)", (unsigned) v,
               v == 0 ? "disabled" : "cushion before playback");
    }
    return;
  }

  if (this->control_channel_dirty_.load(std::memory_order_acquire))
    return;

  // No valid turn phase or follow-up request can result from the new session
  // until its wake marker is fully on the socket and queued PCM is released.
  // Messages in this short interval belong to the previous turn. Connection
  // hello and explicit error messages were handled above and are never hidden.
  if (this->session_starting_.load(std::memory_order_acquire)) {
    ESP_LOGD(TAG, "ignoring stale WS text while wake marker is being sent");
    return;
  }

  if (msg.find("\"type\":\"request_follow_up\"") != std::string::npos) {
    // Server's model called the request_follow_up tool — it asked a
    // question and wants the user to answer without saying a wake word.
    // Defer to loop()'s waiting_for_speaker_stop_ logic so we only fire
    // the chime + arm the mic after speaker_->is_stopped() returns true
    // (i.e. the i2s pipeline has finished playing the question's audio).
    // Setting both pending flags is idempotent — loop() handles both the
    // "already drained" and "still queued" cases uniformly via the
    // speaker-state poll.
    ESP_LOGI(TAG, "request_follow_up — waiting for speaker drain (%u bytes queued)",
             (unsigned) this->audio_fill_);
    this->followup_pending_ = true;
    this->request_follow_up_pending_ = true;
    return;
  }

  // Substring match on `"value":"<phase>"` — keeps us out of a JSON parser
  // until M3 needs richer payloads.
  static const char *const kPhases[] = {"listening", "thinking", "replying", "idle"};
  for (const char *p : kPhases) {
    std::string needle = std::string("\"value\":\"") + p + "\"";
    if (msg.find(needle) != std::string::npos) {
      this->set_phase_(p);
      return;
    }
  }
}

void VaClient::handle_binary_(const uint8_t *data, size_t len) {
  if (this->speaker_ == nullptr || len < 2 || this->audio_buf_ == nullptr)
    return;
  if (this->control_channel_dirty_.load(std::memory_order_acquire))
    return;
  // While a fresh wake marker is being sent, any binary audio necessarily
  // belongs to the superseded turn: new microphone PCM is still transport-
  // gated, so the backend cannot yet have generated a legitimate new reply.
  if (this->session_starting_.load(std::memory_order_acquire))
    return;
  // After a "stop"/barge-in (send_interrupt) the backend is still streaming the
  // rest of the already-generated reply. Drop it so the cancelled reply goes
  // silent instead of refilling the PSRAM ring we just flushed. Cleared on the
  // next "idle"/"listening" phase (see set_phase_).
  if (this->suppress_incoming_audio_)
    return;
  const uint32_t now_ms = millis();
  if (this->turn_t_first_audio_out_ == 0 && this->turn_t_wake_ != 0) {
    this->turn_t_first_audio_out_ = now_ms;
  }
  // Detector 1: WS frame inter-arrival jitter. Normal cadence is ~20 ms
  // per frame (OpenAI streams realtime). A gap > kWsGapWarnMs means the
  // bridge stalled, network blip, or OpenAI burst late — anything that
  // could starve the downstream chain. Log immediately so the gap is
  // adjacent to whatever the user reports hearing.
  if (this->last_binary_ms_ != 0) {
    const uint32_t gap = now_ms - this->last_binary_ms_;
    if (gap > kWsGapWarnMs) {
      this->ws_gap_count_++;
      if (gap > this->ws_gap_max_ms_) this->ws_gap_max_ms_ = gap;
      ESP_LOGW(TAG, "ws audio gap: %u ms (ring fill %u bytes)",
               (unsigned) gap, (unsigned) this->audio_fill_);
    }
  }
  this->last_binary_ms_ = now_ms;
  // PCM16 mono @ 24 kHz, append to ring buffer. loop() drains.
  // Snapshot audio_fill_ under the lock — it's modified by loop() on the
  // other core and we can't trust a torn read.
  size_t free_space;
  portENTER_CRITICAL(&this->ring_mux_);
  free_space = kAudioBufBytes - this->audio_fill_;
  portEXIT_CRITICAL(&this->ring_mux_);
  if (len > free_space) {
    ESP_LOGW(TAG, "audio buffer overflow: dropping %u bytes (have %u free of %u total)",
             (unsigned) (len - free_space), (unsigned) free_space, (unsigned) kAudioBufBytes);
    len = free_space;
    if (len == 0)
      return;
  }
  // Apply user-controlled volume from external_media_player before writing to
  // the ring. volume_ is set from yaml on every media_player volume / mute
  // change. With vol ≤ 1 there is no mathematical way to overflow int16, but
  // we keep a defensive saturation + clipped_samples_ counter so any future
  // gain re-introduction shows up in the per-turn summary instead of silently
  // distorting.
  size_t pairs = len / 2;
  if (pairs > 0) {
    auto *in = reinterpret_cast<const int16_t *>(data);
    // Scale into tts_buf_ — NOT mono_buf_: that vector is owned by the mic
    // task (on_mic_data_ refills it on every mic frame, and the mic never
    // stops while mWW runs), whereas we're on the WS task. Sharing one vector
    // raced the two tasks (concurrent resize/realloc + interleaved writes),
    // putting mic samples / freed memory into the TTS ring — audible as hiss.
    this->tts_buf_.resize(pairs);
    float vol = this->volume_;
    if (vol < 0.0f) vol = 0.0f;
    else if (vol > 1.0f) vol = 1.0f;
    // Q15 fixed point so the inner loop stays integer-only.
    int32_t scale = static_cast<int32_t>(vol * 32768.0f);
    uint32_t clipped = 0;
    for (size_t i = 0; i < pairs; i++) {
      int32_t v = (static_cast<int32_t>(in[i]) * scale) >> 15;
      if (v > 32767) { v = 32767; clipped++; }
      else if (v < -32768) { v = -32768; clipped++; }
      this->tts_buf_[i] = static_cast<int16_t>(v);
    }
    this->clipped_samples_ += clipped;
    data = reinterpret_cast<const uint8_t *>(this->tts_buf_.data());
    // len is unchanged (pairs * 2 == len rounded down; trailing odd byte ignored).
    len = pairs * 2;
  }
  // Two-part write: from tail to end, then wrap to start.
  // We need a stable snapshot of audio_tail_ for the memcpy destination,
  // then commit tail + fill atomically with the writes so the reader on
  // the other core never sees a new tail before the memcpy completed.
  // Doing the memcpy *inside* the critical section is the simplest way
  // to guarantee that ordering — len is at most a few KB per WS frame
  // and PSRAM memcpy is ~10–20 µs, well under any audio deadline.
  portENTER_CRITICAL(&this->ring_mux_);
  const bool was_empty = (this->audio_fill_ == 0);
  size_t tail = this->audio_tail_;
  size_t first = std::min(len, kAudioBufBytes - tail);
  std::memcpy(this->audio_buf_ + tail, data, first);
  if (first < len) {
    std::memcpy(this->audio_buf_, data + first, len - first);
  }
  this->audio_tail_ = (tail + len) % kAudioBufBytes;
  this->audio_fill_ += len;
  portEXIT_CRITICAL(&this->ring_mux_);
  // Jitter buffer: arm priming only when the ring was empty AND the downstream
  // chain is dry — i.e. a true reply start or a real underflow. Mid-reply the
  // ring routinely flips empty (loop() drains each WS clump on arrival) while
  // the downstream chain still holds ~600 ms of audio; re-arming there did
  // nothing but spam "prebuffer ready" every ~50 ms and could hold a small
  // trailing chunk for the full prebuffer deadline. has_buffered_data() is a
  // counter read, safe enough from the WS task. Only when enabled.
  if (was_empty && this->playback_prebuffer_ms_ > 0 && !this->playback_priming_ &&
      !this->speaker_->has_buffered_data()) {
    this->prime_started_ms_ = now_ms;
    this->playback_priming_ = true;
  }
  // No per-chunk log — fires 50+ times per reply at DEBUG and drowns the
  // log. The throttled drain log in loop() gives enough visibility into
  // queue depth.
}

void VaClient::on_mic_data_(const std::vector<uint8_t> &samples) {
  // Snapshot the queue generation before touching this frame. A session
  // boundary that occurs while we downmix must invalidate the eventual write.
  const uint32_t callback_generation =
      this->mic_tx_generation_.load(std::memory_order_acquire);
  const uint32_t callback_started_us = micros();
  this->mic_callback_count_.fetch_add(1, std::memory_order_relaxed);
  this->last_mic_callback_ms_.store(millis(), std::memory_order_relaxed);
  auto finish_callback = [this, callback_started_us]() {
    const uint32_t elapsed_us = micros() - callback_started_us;
    uint32_t previous_max = this->mic_callback_max_us_.load(std::memory_order_relaxed);
    while (elapsed_us > previous_max &&
           !this->mic_callback_max_us_.compare_exchange_weak(
               previous_max, elapsed_us, std::memory_order_relaxed)) {
    }
  };

  // i2s_mics yields interleaved stereo int32 frames: [L0_low,L0_high, R0_low,R0_high, L1..].
  // Each frame = 8 bytes (2ch × 4 bytes). We want one channel converted to
  // int16 mono. Real audio sits in the high 16 bits (ADC pads up to int32).
  if (samples.size() < 8) {
    finish_callback();
    return;
  }

  const auto *in32 = reinterpret_cast<const int32_t *>(samples.data());
  size_t total_int32 = samples.size() / 4;
  size_t mono_samples = total_int32 / 2;  // half belong to this channel
  size_t offset = this->mic_channel_ & 0x1;

  this->mono_buf_.resize(mono_samples);
  uint32_t frame_peak = 0;
  for (size_t i = 0; i < mono_samples; i++) {
    int32_t s = in32[i * 2 + offset];
    const int16_t mono = static_cast<int16_t>(s >> 16);
    this->mono_buf_[i] = mono;
    const uint32_t magnitude =
        mono < 0 ? static_cast<uint32_t>(-static_cast<int32_t>(mono))
                 : static_cast<uint32_t>(mono);
    frame_peak = std::max(frame_peak, magnitude);
  }
  uint32_t previous_peak = this->mic_signal_peak_.load(std::memory_order_relaxed);
  while (frame_peak > previous_peak &&
         !this->mic_signal_peak_.compare_exchange_weak(
             previous_peak, frame_peak, std::memory_order_relaxed)) {
  }
  if (frame_peak == 0)
    this->mic_zero_frame_count_.fetch_add(1, std::memory_order_relaxed);

  // Streaming gate. When no session is active (or the socket is down), retain
  // only the rolling history. This callback performs memory copies only; doing
  // a WebSocket send here blocks microWakeWord from receiving the same frame.
  // The session opens via start_session() (wake handler) and closes on
  // "phase":"idle" from the server (response.done).
  if (!this->streaming_.load(std::memory_order_acquire) ||
      !this->ws_connected_.load(std::memory_order_acquire) ||
      this->ws_handle_ == nullptr) {
    this->preroll_push_(this->mono_buf_.data(), this->mono_buf_.size());
    finish_callback();
    return;
  }

  // Validate the callback's generation while holding the same lock as
  // clear_mic_tx_(). The outer gate alone is insufficient: a callback can pass
  // it, get pre-empted by a new-session clear, then otherwise append stale PCM
  // after that clear. If a boundary raced us, retain this historical frame in
  // the pre-roll ring; the next callback applies the new replay/discard policy.
  const size_t live_bytes = this->mono_buf_.size() * sizeof(int16_t);
  bool boundary_raced = false;
  bool replay_preroll = false;
  bool discard_preroll = false;
  bool queued = false;
  size_t replay_count = 0;
  size_t replay_bytes = 0;

  portENTER_CRITICAL(&this->mic_tx_mux_);
  if (callback_generation != this->mic_tx_generation_.load(std::memory_order_relaxed) ||
      !this->streaming_.load(std::memory_order_relaxed) ||
      !this->ws_connected_.load(std::memory_order_relaxed)) {
    boundary_raced = true;
  } else {
    replay_preroll =
        this->preroll_replay_pending_.exchange(false, std::memory_order_acq_rel);
    if (replay_preroll) {
      this->preroll_discard_pending_.store(false, std::memory_order_release);
      replay_count =
          (this->preroll_buf_ != nullptr && this->preroll_capacity_samples_ > 0)
              ? this->preroll_count_
              : 0;
      replay_bytes = replay_count * sizeof(int16_t);
      const size_t cap = this->preroll_capacity_samples_;
      if (this->mic_tx_ring_.free() >= replay_bytes + live_bytes) {
        size_t oldest = 0;
        size_t first = 0;
        size_t second = 0;
        if (replay_count > 0) {
          oldest = (this->preroll_head_ + cap - replay_count) % cap;
          first = std::min(replay_count, cap - oldest);
          second = replay_count - first;
        }
        queued =
            (first == 0 ||
             this->mic_tx_ring_.push_all(
                 reinterpret_cast<const uint8_t *>(this->preroll_buf_ + oldest),
                 first * sizeof(int16_t))) &&
            (second == 0 ||
             this->mic_tx_ring_.push_all(
                 reinterpret_cast<const uint8_t *>(this->preroll_buf_),
                 second * sizeof(int16_t))) &&
            this->mic_tx_ring_.push_all(
                reinterpret_cast<const uint8_t *>(this->mono_buf_.data()), live_bytes);
      }
    } else {
      discard_preroll =
          this->preroll_discard_pending_.exchange(false, std::memory_order_acq_rel);
      queued = this->mic_tx_ring_.push_all(
          reinterpret_cast<const uint8_t *>(this->mono_buf_.data()), live_bytes);
    }

    if (queued) {
      this->mic_tx_high_water_bytes_ =
          std::max(this->mic_tx_high_water_bytes_, this->mic_tx_ring_.size());
    }
  }
  portEXIT_CRITICAL(&this->mic_tx_mux_);

  if (boundary_raced) {
    this->preroll_push_(this->mono_buf_.data(), this->mono_buf_.size());
    finish_callback();
    return;
  }

  if (replay_preroll) {
    if (queued) {
      this->replay_enqueued_samples_.store(static_cast<uint32_t>(replay_count),
                                           std::memory_order_release);
    } else {
      this->mic_tx_dropped_bytes_.fetch_add(
          static_cast<uint32_t>(replay_bytes + live_bytes),
          std::memory_order_relaxed);
    }
    this->preroll_count_ = 0;
    this->preroll_head_ = 0;
  } else {
    // Chime/button/follow-up paths intentionally discard prior audio. Reset in
    // the mic task (the ring's sole owner) to avoid a cross-task buffer race.
    if (discard_preroll) {
      this->preroll_count_ = 0;
      this->preroll_head_ = 0;
    }
    if (!queued) {
      this->mic_tx_dropped_bytes_.fetch_add(static_cast<uint32_t>(live_bytes),
                                            std::memory_order_relaxed);
    }
  }

  finish_callback();
}

void VaClient::clear_mic_tx_() {
  portENTER_CRITICAL(&this->mic_tx_mux_);
  this->mic_tx_ring_.clear();
  this->mic_tx_generation_.fetch_add(1, std::memory_order_release);
  portEXIT_CRITICAL(&this->mic_tx_mux_);
}

void VaClient::drain_mic_tx_() {
  const uint32_t replay_samples =
      this->replay_enqueued_samples_.exchange(0, std::memory_order_acq_rel);
  if (replay_samples > 0) {
    ESP_LOGI(TAG, "queued %u ms mic pre-roll (%u samples) without blocking I2S",
             (unsigned) (replay_samples * 1000 / kMicSampleRate),
             (unsigned) replay_samples);
  }

  if (this->session_starting_.load(std::memory_order_acquire) ||
      !this->streaming_.load(std::memory_order_acquire) ||
      !this->ws_connected_.load(std::memory_order_acquire) ||
      this->ws_handle_ == nullptr || this->mic_tx_send_buf_ == nullptr) {
    return;
  }

  size_t send_length = 0;
  uint32_t generation = 0;
  portENTER_CRITICAL(&this->mic_tx_mux_);
  generation = this->mic_tx_generation_.load(std::memory_order_relaxed);
  send_length = this->mic_tx_ring_.peek(this->mic_tx_send_buf_, kMicTxSendChunkBytes);
  portEXIT_CRITICAL(&this->mic_tx_mux_);
  if (send_length == 0)
    return;

  auto handle = static_cast<esp_websocket_client_handle_t>(this->ws_handle_);
  const uint32_t send_started_us = micros();
  const int sent = esp_websocket_client_send_bin(
      handle, reinterpret_cast<const char *>(this->mic_tx_send_buf_),
      static_cast<int>(send_length), 10 / portTICK_PERIOD_MS);
  this->mic_tx_send_max_us_ =
      std::max(this->mic_tx_send_max_us_, micros() - send_started_us);
  if (sent <= 0) {
    this->mic_tx_send_failures_++;
    return;  // Retain the bytes for a later bounded retry.
  }

  portENTER_CRITICAL(&this->mic_tx_mux_);
  // A session boundary can clear the ring from the WS task while send_bin()
  // waits. Consume only if this is still the same queue generation.
  if (generation == this->mic_tx_generation_.load(std::memory_order_relaxed)) {
    this->mic_tx_ring_.consume(std::min(static_cast<size_t>(sent), send_length));
  }
  portEXIT_CRITICAL(&this->mic_tx_mux_);

  if (static_cast<size_t>(sent) < send_length) {
    this->mic_tx_send_failures_++;
  }
}

void VaClient::log_mic_health_() {
  const uint32_t now = millis();
  if (now - this->last_mic_health_log_ms_ < kMicHealthLogMs)
    return;
  this->last_mic_health_log_ms_ = now;

  const uint32_t callbacks = this->mic_callback_count_.load(std::memory_order_relaxed);
  const uint32_t callback_delta = callbacks - this->last_mic_health_callback_count_;
  this->last_mic_health_callback_count_ = callbacks;
  const uint32_t last_callback = this->last_mic_callback_ms_.load(std::memory_order_relaxed);
  const uint32_t callback_age = last_callback == 0 ? UINT32_MAX : now - last_callback;
  const uint32_t callback_max_us =
      this->mic_callback_max_us_.exchange(0, std::memory_order_acq_rel);
  const uint32_t signal_peak =
      this->mic_signal_peak_.exchange(0, std::memory_order_acq_rel);
  const uint32_t zero_frames =
      this->mic_zero_frame_count_.exchange(0, std::memory_order_acq_rel);
  const uint32_t dropped =
      this->mic_tx_dropped_bytes_.exchange(0, std::memory_order_acq_rel);
  const uint32_t send_failures = this->mic_tx_send_failures_;
  this->mic_tx_send_failures_ = 0;
  const uint32_t send_max_us = this->mic_tx_send_max_us_;
  this->mic_tx_send_max_us_ = 0;
  const bool control_dirty =
      this->control_channel_dirty_.load(std::memory_order_acquire) != 0;

  size_t queued = 0;
  size_t high_water = 0;
  portENTER_CRITICAL(&this->mic_tx_mux_);
  queued = this->mic_tx_ring_.size();
  high_water = this->mic_tx_high_water_bytes_;
  // Preserve the current fill as the starting point for the next interval.
  this->mic_tx_high_water_bytes_ = queued;
  portEXIT_CRITICAL(&this->mic_tx_mux_);

  if (callback_delta == 0 || callback_age > 100 ||
      (callback_delta > 0 && signal_peak == 0) || dropped > 0 ||
      send_failures > 0 || control_dirty) {
    ESP_LOGW(TAG,
             "mic health: callbacks=%u/%ums age=%ums callback_max=%uus peak=%u "
             "zero_frames=%u tx=%uB high=%uB dropped=%uB send_failures=%u "
             "send_max=%uus control_dirty=%s",
             (unsigned) callback_delta, (unsigned) kMicHealthLogMs,
             (unsigned) callback_age, (unsigned) callback_max_us,
             (unsigned) signal_peak, (unsigned) zero_frames,
             (unsigned) queued, (unsigned) high_water, (unsigned) dropped,
             (unsigned) send_failures, (unsigned) send_max_us,
             control_dirty ? "yes" : "no");
  } else {
    ESP_LOGD(TAG,
             "mic health: callbacks=%u/%ums age=%ums callback_max=%uus peak=%u "
             "zero_frames=%u tx=%uB high=%uB dropped=0B send_failures=0 "
             "send_max=%uus control_dirty=no",
             (unsigned) callback_delta, (unsigned) kMicHealthLogMs,
             (unsigned) callback_age, (unsigned) callback_max_us,
             (unsigned) signal_peak, (unsigned) zero_frames,
             (unsigned) queued, (unsigned) high_water,
             (unsigned) send_max_us);
  }
}

void VaClient::preroll_push_(const int16_t *data, size_t n) {
  if (this->preroll_buf_ == nullptr || this->preroll_capacity_samples_ == 0)
    return;
  const size_t cap = this->preroll_capacity_samples_;
  for (size_t i = 0; i < n; i++) {
    this->preroll_buf_[this->preroll_head_] = data[i];
    if (++this->preroll_head_ >= cap)
      this->preroll_head_ = 0;
    if (this->preroll_count_ < cap)
      this->preroll_count_++;
  }
}

VaClient::Phase VaClient::phase_from_string_(const std::string &phase) {
  if (phase == "listening")
    return Phase::LISTENING;
  if (phase == "thinking")
    return Phase::THINKING;
  if (phase == "replying")
    return Phase::REPLYING;
  return Phase::IDLE;
}

const char *VaClient::phase_name_(Phase p) {
  switch (p) {
    case Phase::LISTENING:
      return "listening";
    case Phase::THINKING:
      return "thinking";
    case Phase::REPLYING:
      return "replying";
    default:
      return "idle";
  }
}

void VaClient::set_phase_(const std::string &phase) {
  if (this->control_channel_dirty_.load(std::memory_order_acquire))
    return;
  if (this->session_starting_.load(std::memory_order_acquire)) {
    ESP_LOGD(TAG, "ignoring stale phase=%s while wake marker is being sent",
             phase.c_str());
    return;
  }

  // Don't dedupe — we want yaml-side control_leds to re-render even on
  // identical phase if other inputs (e.g. va WS connection state) have
  // changed since the last emission.
  const Phase prev = static_cast<Phase>(this->current_phase_.load());
  this->current_phase_.store(static_cast<uint8_t>(phase_from_string_(phase)));
  ESP_LOGD(TAG, "Phase -> %s", phase.c_str());

  // Post-stop `thinking` guard. After a local "stop" the mic gate is closed
  // (send_interrupt set post_stop_guard_), so no new turn can begin until a
  // wake (start_session clears it). A `thinking` arriving here is the server
  // VAD's end-of-turn for the utterance we just cancelled — acting on it
  // strands the LED in `thinking` until the backend's 15 s watchdog (observed
  // 2026-06-14: "stop" mid-question -> 15 s stuck thinking). Ignore it, keeping
  // the prior phase. Scoped to `thinking`: a web search's replying->thinking
  // has no stop (guard stays false), and a reply-drain emits no `thinking`.
  if (this->post_stop_guard_ && phase == "thinking") {
    this->current_phase_.store(static_cast<uint8_t>(prev));  // don't advance to the stale value
    ESP_LOGI(TAG, "ignoring stale 'thinking' after stop (no wake since)");
    return;
  }

  // Lift the post-"stop" incoming-audio suppression ONLY on "listening" — a
  // genuine fresh user turn whose reply is legitimate. We deliberately do NOT
  // lift on "idle": the backend keeps streaming the cancelled reply's audio in
  // real-time (OpenAI generated it faster than playback; a `response.cancel`
  // after the response already finished is a no-op), so it can drain for many
  // seconds AFTER the stop. An "idle" can arrive mid-drain — notably the
  // backend's thinking-watchdog force-idle — and lifting suppression there
  // un-mutes the still-arriving tail, which then plays as an "answer out of
  // nowhere" (observed live 2026-06-13 10:11). A real next reply is always
  // preceded by "listening" (the user speaks), so that stays the only safe
  // gate; until then the tail keeps being dropped. We also do NOT clear on
  // "thinking"/"replying" — those can still belong to the reply we cancelled.
  if (this->suppress_incoming_audio_ && phase == "listening") {
    this->suppress_incoming_audio_ = false;
    ESP_LOGI(TAG, "incoming-audio suppression lifted on phase=listening");
  }

  // Streaming gate state machine:
  //   listening  → mic on (user is being heard)
  //   thinking   → mic stays on. `thinking` only means "the VAD thinks the user
  //                stopped", but with semantic_vad it can flap listening↔thinking
  //                at the start of a turn while the user is still talking. No bot
  //                audio plays during thinking (no echo risk), so keep streaming
  //                until the reply genuinely begins — otherwise a spurious
  //                `thinking` cuts the mic, the backend's input watchdog
  //                force-ends the turn with no transcript, and the turn hangs.
  //   replying   → mic off (bot is speaking; gate to avoid picking up our own
  //                TTS in case the XMOS AEC isn't perfect). barge_in keeps it on.
  //   idle       → mic on for kFollowupMs so the user can answer a question
  //                without re-triggering the wake word. Timer expiry closes
  //                the session.
  if (phase == "listening") {
    if (!this->streaming_.load(std::memory_order_acquire)) {
      ESP_LOGI(TAG, "phase=listening — mic streaming on");
      this->streaming_.store(true, std::memory_order_release);
    }
    // Handsfree barge-in cut-over: a `listening` arriving while we still have
    // TTS queued means the backend's server VAD heard the user talk over the
    // reply and already cancelled the OpenAI response. Drop the audio still in
    // our PSRAM ring so playback stops immediately instead of finishing the
    // now-cancelled sentence. We do NOT send a WS interrupt here — the backend
    // initiated this — we just stop local playback.
    if (this->barge_in_ && this->audio_fill_ > 0) {
      portENTER_CRITICAL(&this->ring_mux_);
      this->audio_head_ = 0;
      this->audio_tail_ = 0;
      this->audio_fill_ = 0;
      portEXIT_CRITICAL(&this->ring_mux_);
      this->idle_emit_pending_ = false;
      ESP_LOGI(TAG, "phase=listening during reply — barge-in, flushed TTS queue");
    }
    if (this->turn_t_listening_ == 0 && this->turn_t_wake_ != 0) {
      this->turn_t_listening_ = millis();
    }
    // Server heard us — watchdog no longer needed.
    this->cancel_timeout("va_no_speech");
    this->cancel_timeout("va_followup");
  } else if (phase == "thinking" || phase == "replying") {
    // Gate the mic off only once the bot actually starts speaking (`replying`).
    // With handsfree barge-in we don't gate at all (the server VAD + XMOS AEC
    // arbitrate talk-over). Crucially we do NOT gate on `thinking`: semantic_vad
    // can flap listening↔thinking at the start of a turn while the user is still
    // speaking, and cutting the mic on those spurious flaps starves the backend
    // of audio → its input watchdog force-ends the turn with no transcript → the
    // turn hangs in thinking. No bot audio plays during thinking, so there's no
    // echo cost to keeping the mic open until the reply genuinely starts.
    if (phase == "replying" && this->streaming_.load(std::memory_order_acquire) &&
        !this->barge_in_) {
      ESP_LOGI(TAG, "phase=replying — mic streaming off");
      this->streaming_.store(false, std::memory_order_release);
      this->clear_mic_tx_();
    }
    if (phase == "thinking" && this->turn_t_thinking_ == 0 && this->turn_t_wake_ != 0) {
      this->turn_t_thinking_ = millis();
    }
    this->cancel_timeout("va_followup");
    this->cancel_timeout("va_followup_open");
    this->cancel_timeout("va_tts_tail");
    this->cancel_timeout("va_no_speech");
    this->followup_pending_ = false;
    this->waiting_for_speaker_stop_ = false;
    this->request_follow_up_pending_ = false;
    this->followup_armed_ = false;
    this->idle_emit_pending_ = false;  // new turn began, drop any held idle
  } else if (phase == "idle") {
    // Turn boundary: reset the WS-gap reference so the silence between THIS
    // reply and the NEXT turn's reply (~7 s across a follow-up exchange, where
    // start_session() — the other reset point — is never called) isn't logged
    // as a bogus "ws audio gap". Only intra-reply gaps are real signal.
    this->last_binary_ms_ = 0;
    // Only open a follow-up window if we just finished a real turn —
    // i.e. the previous phase was thinking or replying. Otherwise we'd
    // open the window for every spurious idle (initial WS hello, post-
    // disconnect idle, etc), spamming "follow-up window open" logs and
    // opening the mic for 5s every time the device just reconnects.
    const bool turn_just_ended = prev == Phase::THINKING || prev == Phase::REPLYING;
    if (!turn_just_ended) {
      // Plain idle (boot, reconnect, etc) — no follow-up. Fall through to
      // the regular trigger fire so the LED updates.
      //
      // An idle that arrives while the mic gate is still OPEN means an orphaned
      // stream that nothing else will close — shut it. Two real ways to reach
      // here with streaming_ still true:
      //   • idle straight from `listening`: the turn died without a reply (a
      //     backend force-idle on rate-limit / thinking-watchdog — the backend
      //     suppresses `thinking` after declaring a turn dead, so `listening`
      //     is exactly where the device sits — or a WS drop mid-listening).
      //   • idle from `idle` (prev==IDLE) with the mic open: the brief post-wake
      //     "waiting for the first phase" window, or an open follow-up window,
      //     when the WS drops (on_ws_event fires set_phase_("idle") on
      //     disconnect). Without this the gate stays open and the mic resumes
      //     streaming the room the instant we reconnect — and the backend's
      //     mic-resume buffer clear can't help because the stream never paused.
      // Closing here can't cut a LIVE follow-up window short: the only idle that
      // reaches an open window is exactly such a disconnect — the backend sends
      // no idle while it's waiting for the user to answer.
      if (this->streaming_.load(std::memory_order_acquire)) {
        ESP_LOGI(TAG, "idle while mic open (prev=%s) — closing orphaned mic gate",
                 phase_name_(prev));
        this->streaming_.store(false, std::memory_order_release);
        this->clear_mic_tx_();
        this->cancel_timeout("va_no_speech");
      }
    } else if (this->suppress_followup_) {
      // send_interrupt() set this — user explicitly asked us to stop.
      // Close the session cleanly: streaming off, no follow-up, fall through
      // to the regular trigger fire so the LED goes idle.
      this->suppress_followup_ = false;
      this->streaming_.store(false, std::memory_order_release);
      this->clear_mic_tx_();
      this->followup_pending_ = false;
      this->waiting_for_speaker_stop_ = false;
      this->request_follow_up_pending_ = false;
      this->followup_armed_ = false;
      this->cancel_timeout("va_tts_tail");
      this->idle_emit_pending_ = false;
    } else if (this->audio_fill_ == 0) {
      // Stale-`idle` guard. prev==REPLYING with NO audio played since the last
      // wake (turn_t_first_audio_out_==0) means this `idle` belongs to a reply
      // that was stopped and then superseded by a new wake while it was still
      // draining: start_session() reset turn_t_first_audio_out_ and the old
      // tail is suppressed, so nothing actually played in THIS session. Opening
      // a follow-up here lights a spurious `listening` window over the fresh
      // wake session (observed live 2026-06-14: web-search → "stop" → bare wake
      // → ~4 s `listening` flicker; device log "Phase -> idle (was replying)").
      // Ignore it: the wake's no-speech watchdog is still armed (we never
      // reached `listening`, so it was never cancelled) and owns the session —
      // it idles the LED and closes the mic. current_phase_ is already IDLE
      // (set at the top of set_phase_), so the next message's `prev` is fine.
      // Tight by construction: a reply that really played sets
      // turn_t_first_audio_out_ (suppression lifts on `listening`), and a
      // follow-up chain keeps it set (only start_session resets it), so no
      // legitimate follow-up is skipped; the dead-turn path (prev==THINKING,
      // watchdog already cancelled) is deliberately left untouched.
      if (prev == Phase::REPLYING && this->turn_t_first_audio_out_ == 0) {
        ESP_LOGI(TAG, "stale replying-idle after wake (no audio this session) "
                      "— ignoring, no follow-up");
        // Distinguished by the mic gate: streaming_==true is a BARE WAKE (mic
        // still open, never reached `replying` this session) → the no-speech
        // watchdog is armed and owns the idle + mic close, so leave the LED on
        // the wake state. streaming_==false means the turn DID reach `replying`
        // (mic gated) but the audio never played — e.g. suppress_incoming_audio_
        // was still set from an earlier stop, so turn_t_first_audio_out_ stayed
        // 0. The watchdog was cancelled when the turn reached `listening`, so
        // nothing else fires idle → fall through to the LED trigger below, else
        // the ring strands on `replying` (observed live 2026-06-14: rapid stops
        // left suppression on, the search reply was dropped, the LED hung).
        if (this->streaming_.load(std::memory_order_acquire)) {
          return;  // bare wake: va_no_speech owns the idle + mic
        }
        // else: fall through to fire the idle LED (still no follow-up).
      } else {
        // Server says response.done and the device has actually played out.
        // Open the follow-up window (mic on so user can answer a question).
        this->open_followup_window_(this->followup_ms_);
      }
      // fall through to fire the trigger normally below (LED -> idle); the
      // follow-up window, if any, fires its own `listening` LED after its delay.
    } else {
      // Server says response.done, but we still have seconds of TTS queued
      // in PSRAM + downstream rings. Two things wait on the queue:
      //   1) the LED transition to idle (otherwise it goes off while the
      //      device is still speaking)
      //   2) opening the follow-up mic window (echo + false VAD trigger)
      // Mark both pending; the drain handler in loop() releases them
      // together after the speaker actually finishes.
      ESP_LOGI(TAG, "phase=idle but %u bytes still queued; LED + follow-up deferred",
               (unsigned) this->audio_fill_);
      this->followup_pending_ = true;
      this->idle_emit_pending_ = true;
      return;  // suppress immediate trigger fire — open_followup_window_ will fire it later
    }
  }

  // set_phase_ may be called from the websocket task; ESPHome triggers and
  // most component APIs are not thread-safe. Marshal the side effects onto
  // the main loop via defer().
  std::string phase_copy = phase;
  this->defer([this, phase_copy]() {
    // We deliberately do NOT call speaker->stop() on "listening" anymore:
    // the speaker task runs continuously after setup() and play() just
    // appends to its ring buffer. Stop/start churn was creating multiple
    // speaker_task instances racing for the i2s channel ("Parent bus is
    // busy"). For barge-in/interrupt we'll add a buffer-flush API in M3.
    for (auto *t : this->phase_triggers_) {
      t->trigger(phase_copy);
    }
  });
}

void VaClient::start_session() { this->start_session_(false); }

void VaClient::start_session_with_preroll() { this->start_session_(true); }

void VaClient::start_session_(bool replay_preroll) {
  if (this->session_starting_.exchange(1, std::memory_order_acq_rel) != 0) {
    ESP_LOGW(TAG, "start_session() ignored — wake marker send already in progress");
    return;
  }

  // Close the old producer before any bounded control send. In particular, a
  // residual-reply interrupt can wait up to kWsControlSendTimeoutMs; leaving
  // streaming_ open during that wait would enqueue the user's same-breath
  // command into the old TX queue, which send_interrupt() then clears. With
  // the gate closed, those callbacks roll into pre-roll and survive for the
  // fresh session instead.
  this->streaming_.store(false, std::memory_order_release);
  this->clear_mic_tx_();
  this->replay_enqueued_samples_.store(0, std::memory_order_relaxed);

  // Open the streaming window. on_mic_data_ will start forwarding frames to
  // the server until "phase":"idle" comes back (response.done). Without this
  // gate, OpenAI Realtime's server VAD would respond to any speech in the
  // room — wake word would be cosmetic.

  // Belt-and-suspenders barge-in. The yaml wake handler calls send_interrupt()
  // when it observes voice_assistant_phase == replying, but two windows slip
  // past that check:
  //   1) server already sent phase=idle yet PSRAM still has seconds of TTS
  //      queued (idle_emit_pending_). yaml's voice_assistant_phase has been
  //      reset to idle and the wake handler takes the "fresh session" path —
  //      no interrupt — so the new reply overlaps with the tail of the old.
  //   2) wake fires mid-reply on a long answer where the server is still
  //      generating tokens; without an interrupt, OpenAI keeps streaming TTS
  //      we'll never play, burning tokens.
  // The bridge treats interrupt as cheap when there's nothing to cancel
  // (response_cancel_not_active is in its benignCodes set), and
  // input_audio_buffer.clear is safe here because the new turn's queue is
  // already cleared above and its frames cannot drain until the wake marker is
  // sent.
  const Phase phase_now = static_cast<Phase>(this->current_phase_.load());
  size_t audio_fill_snapshot = 0;
  portENTER_CRITICAL(&this->ring_mux_);
  audio_fill_snapshot = this->audio_fill_;
  portEXIT_CRITICAL(&this->ring_mux_);
  const bool residual_reply =
      audio_fill_snapshot > 0 ||
      this->idle_emit_pending_ ||
      phase_now == Phase::REPLYING ||
      phase_now == Phase::THINKING;
  if (residual_reply) {
    ESP_LOGI(TAG, "start_session: interrupting residual reply (phase=%s, fill=%u)",
             phase_name_(phase_now), (unsigned) audio_fill_snapshot);
    if (!this->send_interrupt()) {
      this->session_starting_.store(false, std::memory_order_release);
      this->streaming_.store(false, std::memory_order_release);
      this->clear_mic_tx_();
      this->fire_phase_led_("idle");
      this->fail_control_channel_("residual interrupt failure");
      ESP_LOGW(TAG, "start_session() aborted — prior turn was not cancelled");
      return;
    }
  }

  // A no-chime wake replays the rolling buffer to preserve a command spoken in
  // the same breath as the wake word. Every other caller discards it so chime
  // leakage or arbitrary pre-button room audio is never submitted as speech.
  this->preroll_replay_pending_.store(replay_preroll, std::memory_order_release);
  this->preroll_discard_pending_.store(!replay_preroll, std::memory_order_release);

  // New wake word preempts every pending or active follow-up state from the
  // previous turn, even if the control marker below cannot be delivered.
  this->followup_pending_ = false;
  this->waiting_for_speaker_stop_ = false;
  this->request_follow_up_pending_ = false;
  this->followup_armed_ = false;
  this->idle_emit_pending_ = false;
  this->suppress_followup_ = false;
  // A genuine new turn starts here — drop the post-stop `thinking` guard. Set
  // this after the residual-reply send_interrupt() above may have re-armed it.
  this->post_stop_guard_ = false;
  this->cancel_timeout("va_no_speech");
  this->cancel_timeout("va_followup");
  this->cancel_timeout("va_followup_open");
  this->cancel_timeout("va_tts_tail");

  // Initialize every state a fast server response might touch before sending
  // the marker. Otherwise an early phase=listening could cancel no watchdog,
  // only for us to install a fresh watchdog after it had already been heard.
  this->turn_t_wake_ = millis();
  this->turn_t_listening_ = 0;
  this->turn_t_thinking_ = 0;
  this->turn_t_first_audio_out_ = 0;
  this->last_binary_ms_ = 0;
  this->ws_gap_count_ = 0;
  this->ws_gap_max_ms_ = 0;
  this->clipped_samples_ = 0;
  this->underrun_logged_this_turn_ = false;

  // Watchdog: if server doesn't hear us within kNoSpeechTimeoutMs, abort the
  // session so we're not stuck with the mic open after a misfire.
  this->set_timeout("va_no_speech", kNoSpeechTimeoutMs, [this]() {
    ESP_LOGI(TAG, "no speech detected for %u ms — aborting session",
             (unsigned) kNoSpeechTimeoutMs);
    if (this->ws_connected_.load(std::memory_order_acquire) &&
        this->ws_handle_ != nullptr) {
      const char m[] = "{\"type\":\"interrupt\"}";
      auto handle = static_cast<esp_websocket_client_handle_t>(this->ws_handle_);
      const int sent = esp_websocket_client_send_text(
          handle, m, sizeof(m) - 1, pdMS_TO_TICKS(kWsControlSendTimeoutMs));
      if (sent != static_cast<int>(sizeof(m) - 1)) {
        ESP_LOGW(TAG, "no-speech interrupt send failed (sent=%d expected=%u)",
                 sent, (unsigned) (sizeof(m) - 1));
        this->fail_control_channel_("no-speech interrupt failure");
      }
    }
    this->streaming_.store(false, std::memory_order_release);
    this->clear_mic_tx_();
    this->turn_t_wake_ = 0;
    // Force LED back to idle from yaml side.
    this->defer([this]() {
      for (auto *t : this->phase_triggers_) {
        t->trigger("idle");
      }
    });
  });

  // The producer may now queue pre-roll + live PCM, but drain_mic_tx_ remains
  // gated by session_starting_. start_session_ and drain_mic_tx_ both run on
  // ESPHome's main loop, while the flag safely coordinates the I2S/WS tasks.
  this->streaming_.store(true, std::memory_order_release);
  const bool wake_sent = this->send_wake_();
  if (!wake_sent || this->session_starting_.load(std::memory_order_acquire) == 0) {
    // Disconnect and server-error handlers clear session_starting_ while the
    // bounded send is in flight. Treat that exactly like a failed send.
    this->session_starting_.store(false, std::memory_order_release);
    this->streaming_.store(false, std::memory_order_release);
    this->clear_mic_tx_();
    this->replay_enqueued_samples_.store(0, std::memory_order_relaxed);
    this->preroll_replay_pending_.store(false, std::memory_order_release);
    this->preroll_discard_pending_.store(true, std::memory_order_release);
    this->cancel_timeout("va_no_speech");
    this->turn_t_wake_ = 0;
    this->fire_phase_led_("idle");
    ESP_LOGW(TAG, "start_session() aborted — wake marker was not delivered");
    if (!wake_sent)
      this->fail_control_channel_("wake marker failure");
    return;
  }

  this->session_starting_.store(false, std::memory_order_release);
  ESP_LOGI(TAG, "start_session() — streaming on%s",
           replay_preroll ? " (queued pre-roll armed)" : "");
}

void VaClient::open_followup_window_(uint32_t duration_ms) {
  // If a phase=idle LED transition was held back while audio drained, fire
  // it now so the LED goes to idle in sync with the speaker actually going
  // quiet (instead of as soon as the server emitted response.done).
  if (this->idle_emit_pending_) {
    this->idle_emit_pending_ = false;
    this->defer([this]() {
      for (auto *t : this->phase_triggers_) {
        t->trigger("idle");
      }
    });
    // Per-turn latency summary. Anchors are zero if we skipped a milestone
    // (e.g. interrupt mid-reply); show "?" so the line stays readable.
    if (this->turn_t_wake_ != 0) {
      uint32_t now = millis();
      auto fmt = [](uint32_t from, uint32_t to) -> std::string {
        if (from == 0 || to == 0 || to < from)
          return "?";
        return std::to_string(to - from) + "ms";
      };
      ESP_LOGI(TAG,
               "turn latency: wake→listening=%s listening→thinking=%s "
               "thinking→first_audio=%s first_audio→played_out=%s "
               "total=%s",
               fmt(this->turn_t_wake_, this->turn_t_listening_).c_str(),
               fmt(this->turn_t_listening_, this->turn_t_thinking_).c_str(),
               fmt(this->turn_t_thinking_, this->turn_t_first_audio_out_).c_str(),
               fmt(this->turn_t_first_audio_out_, now).c_str(),
               fmt(this->turn_t_wake_, now).c_str());
      // Audio-quality summary: only logged if anything anomalous fired.
      // A clean turn produces no line — keeps the noise floor low.
      if (this->ws_gap_count_ > 0 || this->clipped_samples_ > 0 ||
          this->underrun_logged_this_turn_) {
        ESP_LOGW(TAG,
                 "turn audio: ws_gaps=%u (max=%ums) clipped_samples=%u underrun=%s",
                 (unsigned) this->ws_gap_count_,
                 (unsigned) this->ws_gap_max_ms_,
                 (unsigned) this->clipped_samples_,
                 this->underrun_logged_this_turn_ ? "yes" : "no");
      }
      this->turn_t_wake_ = 0;  // mark turn as logged
    }
  }
  if (duration_ms == 0) {
    // Follow-up disabled for this call: turn-based behaviour like the
    // original pipeline. Leave the mic closed; user must say a wake word
    // for the next turn. (The LED idle was already emitted above / by the
    // set_phase_ tail.)
    this->streaming_.store(false, std::memory_order_release);
    this->clear_mic_tx_();
    return;
  }
  // Follow-up dialog window. We do NOT open the mic immediately: has_buffered_
  // data() (the drain signal that got us here) goes false ~500 ms before true
  // silence — there's still the i2s ring + DAC tail playing out. Opening the
  // mic now would let that tail leak back in (XMOS AEC ~10x) and false-trigger
  // the server VAD. So wait followup_open_delay_ms_ (from the backend) for the
  // tail to clear, THEN open the mic and show `listening` so the user can see
  // the device is waiting for them to answer (without a wake word). The LED
  // stays idle (emitted just above) during this short gap. Any new turn / wake /
  // stop / interrupt cancels "va_followup_open" before it fires (see set_phase_,
  // start_session, send_interrupt), so a new turn won't reopen the mic under it.
  const uint32_t open_delay = this->followup_open_delay_ms_;
  ESP_LOGI(TAG, "follow-up: mic opens in %u ms, then listening for %u ms",
           (unsigned) open_delay, (unsigned) duration_ms);
  this->set_timeout("va_followup_open", open_delay, [this, duration_ms]() {
    if (!this->ws_connected_.load(std::memory_order_acquire)) {
      // The reply drained into a dead connection (WS dropped mid-reply).
      // Opening the mic would show a "listening" LED while on_mic_data_
      // drops every frame — a window the user talks into for nothing.
      // Leave the LED on idle; a fresh wake after reconnect starts clean.
      ESP_LOGI(TAG, "follow-up window skipped — WS disconnected");
      return;
    }
    ESP_LOGI(TAG, "follow-up window open (mic on, listening for %u ms)", (unsigned) duration_ms);
    this->clear_mic_tx_();
    this->preroll_replay_pending_.store(false, std::memory_order_release);
    this->preroll_discard_pending_.store(true, std::memory_order_release);
    this->streaming_.store(true, std::memory_order_release);
    this->fire_phase_led_("listening");  // blue ring: user may answer now
    this->set_timeout("va_followup", duration_ms, [this]() {
      if (this->streaming_.load(std::memory_order_acquire)) {
        ESP_LOGI(TAG, "follow-up window expired — mic streaming off");
        this->streaming_.store(false, std::memory_order_release);
        this->clear_mic_tx_();
        this->send_mic_flush_();        // drop any uncommitted partial utterance
        this->fire_phase_led_("idle");  // no answer came; back to idle
      }
    });
  });
}

void VaClient::send_mic_flush_() {
  // The mic gate just closed mid-stream because a follow-up window timed out.
  // If the user had started (but not finished) speaking, that audio sits
  // UNCOMMITTED in OpenAI's input_audio_buffer; left there, a later wake's
  // audio "completes" it and the model answers a stale half-sentence. Drop it
  // NOW, at the cut-off, so no reactive clear-on-wake is needed (that disturbed
  // the server VAD and caused garbage commits). This timer only fires when the
  // user did NOT trigger speech — `listening` cancels va_followup — so it can
  // never drop a valid command. Cheap no-op when the buffer was empty.
  if (this->ws_connected_.load(std::memory_order_acquire) &&
      this->ws_handle_ != nullptr) {
    const char msg[] = "{\"type\":\"flush\"}";
    auto handle = static_cast<esp_websocket_client_handle_t>(this->ws_handle_);
    const uint32_t started_ms = millis();
    const int sent = esp_websocket_client_send_text(
        handle, msg, sizeof(msg) - 1, pdMS_TO_TICKS(kWsControlSendTimeoutMs));
    const uint32_t elapsed_ms = millis() - started_ms;
    if (sent == static_cast<int>(sizeof(msg) - 1)) {
      ESP_LOGI(TAG, "follow-up window closed — sent flush in %u ms "
                    "(drop uncommitted mic audio)",
               (unsigned) elapsed_ms);
    } else {
      ESP_LOGW(TAG, "follow-up flush send failed in %u ms (sent=%d expected=%u)",
               (unsigned) elapsed_ms, sent, (unsigned) (sizeof(msg) - 1));
      this->fail_control_channel_("follow-up flush failure");
    }
  }
}

bool VaClient::send_wake_() {
  // Tell the backend a fresh wake started (dangling-VAD guard, A). Until the
  // user actually speaks this turn, OpenAI's server VAD can still fire an
  // end-of-turn for a PREVIOUS utterance that never closed (the reply gated the
  // mic mid-sentence) — committing it auto-creates a garbage answer to an empty
  // turn. The backend uses this signal to suppress that stale thinking + cancel
  // the racing response. Sent on every start_session(); old backends ignore it.
  if (!this->ws_connected_.load(std::memory_order_acquire) ||
      this->ws_handle_ == nullptr) {
    ESP_LOGW(TAG, "wake marker not sent — WS disconnected");
    return false;
  }

  const char msg[] = "{\"type\":\"wake\"}";
  auto handle = static_cast<esp_websocket_client_handle_t>(this->ws_handle_);
  const int sent = esp_websocket_client_send_text(
      handle, msg, sizeof(msg) - 1, pdMS_TO_TICKS(kWsControlSendTimeoutMs));
  if (sent != static_cast<int>(sizeof(msg) - 1)) {
    ESP_LOGW(TAG, "wake marker send failed (sent=%d expected=%u)", sent,
             (unsigned) (sizeof(msg) - 1));
    return false;
  }

  ESP_LOGI(TAG, "wake — sent {\"type\":\"wake\"} (dangling-VAD guard)");
  return true;
}

void VaClient::fire_phase_led_(const std::string &phase) {
  // Drive the yaml on_phase automation (LED ring + voice_assistant_phase global)
  // from a device-side timer, not a server message. Marshalled via defer() so it
  // runs on the main loop even if called from another task.
  std::string phase_copy = phase;
  this->defer([this, phase_copy]() {
    for (auto *t : this->phase_triggers_) {
      t->trigger(phase_copy);
    }
  });
}

void VaClient::commit_followup_mic() {
  // Called from yaml's on_followup_opened automation once the chime has
  // finished playing AND the i2s tail has cleared (wait_until + delay).
  // If anything pre-empted us between trigger fire and here (a fresh
  // wake word, a Stop, send_interrupt, or a new turn starting) the
  // armed flag was cleared — silently no-op so we don't reopen the mic
  // out of nowhere.
  if (!this->followup_armed_) {
    ESP_LOGD(TAG, "commit_followup_mic: not armed, ignoring");
    return;
  }
  this->followup_armed_ = false;
  if (this->control_channel_dirty_.load(std::memory_order_acquire) ||
      !this->ws_connected_.load(std::memory_order_acquire) ||
      this->ws_handle_ == nullptr) {
    ESP_LOGW(TAG, "commit_followup_mic: control channel unavailable, ignoring");
    return;
  }
  ESP_LOGI(TAG, "follow-up mic armed by yaml (window %u ms)",
           (unsigned) kRequestFollowUpMs);
  // Discard the pre-roll: the request_follow_up chime just played and leaked
  // into the ring; don't replay it to OpenAI. (Same "Au!" guard as the wake
  // path; consumed by the mic task on the next frame.)
  this->clear_mic_tx_();
  this->preroll_replay_pending_.store(false, std::memory_order_release);
  this->preroll_discard_pending_.store(true, std::memory_order_release);
  this->streaming_.store(true, std::memory_order_release);
  this->set_timeout("va_followup", kRequestFollowUpMs, [this]() {
    if (this->streaming_.load(std::memory_order_acquire)) {
      ESP_LOGI(TAG, "follow-up window expired — mic streaming off");
      this->streaming_.store(false, std::memory_order_release);
      this->clear_mic_tx_();
      this->send_mic_flush_();  // drop any uncommitted partial utterance
    }
  });
}

bool VaClient::send_interrupt() {
  // Best-effort cancel to the backend — ONLY if the socket is alive. The local
  // cleanup below must ALWAYS run: returning early on a dead socket (the old
  // behaviour) left streaming_ on, the PSRAM ring full and the follow-up timers
  // armed after a "stop" on a dead link, so the mic would resume streaming the
  // room the instant we reconnected.
  bool attempted = false;
  bool delivered = false;
  if (this->ws_connected_.load(std::memory_order_acquire) &&
      this->ws_handle_ != nullptr) {
    attempted = true;
    const char msg[] = "{\"type\":\"interrupt\"}";
    auto handle = static_cast<esp_websocket_client_handle_t>(this->ws_handle_);
    const int sent = esp_websocket_client_send_text(
        handle, msg, sizeof(msg) - 1, pdMS_TO_TICKS(kWsControlSendTimeoutMs));
    delivered = sent == static_cast<int>(sizeof(msg) - 1);
    if (!delivered) {
      ESP_LOGW(TAG, "interrupt send failed (sent=%d expected=%u)", sent,
               (unsigned) (sizeof(msg) - 1));
    }
  } else {
    ESP_LOGW(TAG, "send_interrupt: WS not connected — local cleanup only");
  }
  // Flush our PSRAM playback queue — what's already been pushed into the
  // resampler/mixer/leaf will still drain (~600 ms residual), but everything
  // we have yet to hand off is dropped. The yaml side stops the resampler
  // explicitly. Reset deferred state too so we don't accidentally hold an
  // "idle" emit waiting for the (now-empty) queue. The ring reset has to
  // happen under the mux: the WS task could be mid-write and seeing
  // head=tail=fill=0 partway through would let it write into a "freshly
  // empty" buffer the user just barge-cancelled.
  portENTER_CRITICAL(&this->ring_mux_);
  this->audio_head_ = 0;
  this->audio_tail_ = 0;
  this->audio_fill_ = 0;
  portEXIT_CRITICAL(&this->ring_mux_);
  // Drop further incoming TTS until the backend confirms the turn boundary —
  // it keeps streaming the rest of the (already-generated) reply otherwise.
  this->suppress_incoming_audio_ = true;
  // Ring was just flushed; re-arm the jitter buffer fresh for the next reply.
  this->playback_priming_ = false;
  // Abandon any in-progress cold-start silence-prime; the next reply will detect
  // cold and re-prime cleanly.
  this->chain_prime_remaining_ = 0;
  // Close the mic gate. An interrupt during the OPEN follow-up window would
  // otherwise leave streaming_ true while the va_followup close-timer gets
  // cancelled just below — mic open + streaming to OpenAI indefinitely, so any
  // later room speech becomes an unprompted turn. Callers that start a fresh
  // turn (start_session) re-open it themselves right after.
  this->streaming_.store(false, std::memory_order_release);
  this->clear_mic_tx_();
  this->followup_pending_ = false;
  this->waiting_for_speaker_stop_ = false;
  this->request_follow_up_pending_ = false;
  this->followup_armed_ = false;
  this->idle_emit_pending_ = false;
  this->cancel_timeout("va_no_speech");
  this->cancel_timeout("va_followup");
  this->cancel_timeout("va_tts_tail");
  // The phase=idle the server is about to send shouldn't open a follow-up
  // mic window — the user said "stop", not "wait for me to keep talking".
  this->suppress_followup_ = true;
  // Mic gate is now closed: no new turn can begin until a wake. Ignore any
  // `thinking` the backend emits in the meantime — it's the server VAD's
  // end-of-turn for the utterance we just cancelled, not a real new turn.
  // Cleared in start_session() (the next wake). See set_phase_.
  this->post_stop_guard_ = true;
  this->cancel_timeout("va_followup_open");
  if (attempted && !delivered)
    this->fail_control_channel_("interrupt failure");
  ESP_LOGI(TAG, "send_interrupt — WS msg %s, queue flushed",
           delivered ? "sent" : "not sent");
  return delivered;
}

}  // namespace va_client
}  // namespace esphome
