// Derived from ESPHome 2026.8.2 at commit 8b5c06add684cdbef0613e1c295c0f4e6f9dc5c2.
// Copyright (c) 2019 ESPHome.
// Modified 2026-09-06 by the Pippa wake-word project to execute external-state CRNN models.
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#ifdef USE_ESP32

#include "preprocessor_settings.h"
#include "model_data.h"

#include "esphome/core/preferences.h"

#include <array>
#include <atomic>
#include <memory>
#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

namespace esphome::micro_wake_word {

static const uint8_t MIN_SLICES_BEFORE_DETECTION = 100;
static const uint32_t STREAMING_MODEL_VARIABLE_ARENA_SIZE = 1024;
static const size_t EXTERNAL_STREAMING_STATE_SIZE = 112;
static const size_t STREAMING_OP_COUNT = 24;

struct DetectionEvent {
  std::string *wake_word;
  bool detected;
  bool partially_detection;  // Set if the most recent probability exceed the threshold, but the sliding window average
                             // hasn't yet
  uint8_t max_probability;
  uint8_t average_probability;
  bool blocked_by_vad = false;
};

#ifdef PIPPA_CRNN_SELF_TEST
struct CrnnSelfTestStepResult {
  uint16_t record;
  uint8_t expected_raw;
  uint8_t actual_raw;
  uint8_t expected_score;
  uint8_t actual_score;
  uint8_t state_exact;
  uint8_t state_within_one;
  uint8_t state_max_abs_diff;
  bool invoke_ok;
  bool feedback_ok;
  bool passed;
};
#endif

#ifdef PIPPA_CRNN_METRICS
struct CrnnRuntimeMetricsSnapshot {
  uint32_t samples;
  uint64_t total_samples;
  uint32_t runtime_budget_us;
  uint32_t p50_bucket_us;
  uint32_t p95_bucket_us;
  uint32_t p99_bucket_us;
  uint32_t runtime_max_us;
  uint32_t runtime_budget_overruns;
  uint32_t cold_samples;
  uint32_t cold_last_us;
  uint32_t cold_max_us;
  uint32_t cold_runtime_budget_overruns;
  uint32_t cadence_intervals;
  uint32_t cadence_threshold_us;
  uint32_t cadence_max_gap_us;
  uint32_t cadence_gap_over_threshold;
  uint32_t dropped_reports;
  uint8_t percentile_censored_mask;
};

struct CrnnScoreMetricsSnapshot {
  uint32_t samples;
  uint8_t raw_max;
  uint8_t sliding_average_max;
  uint8_t input_byte_max;
  uint8_t input_mean_min;
  uint8_t input_mean_max;
  uint32_t input_delta_l1_max;
};
#endif

class StreamingModel {
 public:
  // Runtime models are heap owned and destroyed while the device is running, so freeing the arenas cannot
  // depend on the owner calling unload_model() first. unload_model() is not virtual and is safe to repeat.
  virtual ~StreamingModel() { this->unload_model(); }

  virtual void log_model_config() = 0;
  virtual DetectionEvent determine_detected() = 0;

  // Performs inference on the given features.
  //  - If the model is enabled but not loaded, it will load it
  //  - If the model is disabled but loaded, it will unload it
  // Returns true if sucessful or false if there is an error
  bool perform_streaming_inference(const int8_t features[PREPROCESSOR_FEATURE_SIZE], uint32_t feature_step_us);

#ifdef PIPPA_CRNN_SELF_TEST
  /// Runs the pinned measurement candidate's known closed-loop trajectory once per boot.
  /// The caller must own the inference task and must not have started the microphone source.
  bool run_crnn_boot_self_test();

  /// Emits retained self-test results from the main loop after API logging is available.
  void report_crnn_boot_self_test();
#endif

#ifdef PIPPA_CRNN_METRICS
  /// Copies a complete 10,000-invocation CRNN metrics window from the inference task to the main loop.
  bool take_crnn_metrics_snapshot(CrnnRuntimeMetricsSnapshot *snapshot);

  /// Copies a short CRNN score window from the inference task to the main loop.
  bool take_crnn_score_snapshot(CrnnScoreMetricsSnapshot *snapshot);
#endif

  /// @brief Sets all recent_streaming_probabilities to 0 and resets the ignore window count
  void reset_probabilities();

  /// @brief Requests that any externally-fed recurrent state be reset before the next inference.
  /// This is safe to call from the main loop while inference runs on its worker task.
  void reset_streaming_state() { this->streaming_state_reset_requested_.store(true); }

  /// @brief Destroys the TFLite interpreter and frees the tensor and variable arenas' memory
  void unload_model();

  /// @brief Enable the model. The next performing_streaming_inference call will load it.
  virtual void enable() {
    this->reset_streaming_state();
    this->enabled_ = true;
  }

  /// @brief Disable the model. The next performing_streaming_inference call will unload it.
  virtual void disable() {
    this->reset_streaming_state();
    this->enabled_ = false;
  }

  /// @brief Return true if the model is enabled.
  bool is_enabled() const { return this->enabled_; }

  /// @brief Return true if the model has usable data. A model without it can never be loaded or run.
  bool has_model_data() const { return this->model_start_ != nullptr; }

  bool get_unprocessed_probability_status() const { return this->unprocessed_probability_status_; }

  // Quantized probability cutoffs mapping 0.0 - 1.0 to 0 - 255
  uint8_t get_default_probability_cutoff() const { return this->default_probability_cutoff_; }
  uint8_t get_probability_cutoff() const { return this->probability_cutoff_.load(std::memory_order_relaxed); }
  void set_probability_cutoff(uint8_t probability_cutoff) {
    this->probability_cutoff_.store(probability_cutoff, std::memory_order_relaxed);
  }

 protected:
  /// @brief Allocates tensor and variable arenas and sets up the model interpreter
  /// @return True if successful, false otherwise
  bool load_model_();
  /// @brief Probes the actual required tensor arena size by trial allocation.
  /// Tries the manifest size first, then 2x if that fails.
  /// @return The required arena size rounded up to 16-byte alignment, or 0 on failure.
  size_t probe_arena_size_();
  /// @brief Returns true if successfully registered the streaming model's TensorFlow operations
  bool register_streaming_ops_(tflite::MicroMutableOpResolver<STREAMING_OP_COUNT> &op_resolver);

  /// @brief Finds feature, probability, and optional external-state tensors by role/shape rather than position.
  bool discover_streaming_tensors_();
  /// @brief Applies the quantized representation of real zero to the external state and clears partial feature input.
  void apply_streaming_state_reset_();
  /// @brief Requantizes the state output into the state input after a successful invocation.
  bool feed_back_streaming_state_();
  /// @brief Reads either the stock uint8 probability or the CRNN int8 probability as a 0..255 value.
  uint8_t read_probability_() const;

  tflite::MicroMutableOpResolver<STREAMING_OP_COUNT> streaming_op_resolver_;

  bool loaded_{false};
  bool enabled_{true};
  bool tensor_arena_size_probed_{false};
  bool unprocessed_probability_status_{false};
  std::atomic<bool> streaming_state_reset_requested_{true};
  uint8_t current_stride_step_{0};
  int16_t ignore_windows_{-MIN_SLICES_BEFORE_DETECTION};

  uint8_t default_probability_cutoff_;
  std::atomic<uint8_t> probability_cutoff_{0};
  size_t sliding_window_size_;

  size_t last_n_index_{0};
  size_t tensor_arena_size_;
  std::vector<uint8_t> recent_streaming_probabilities_;

  const uint8_t *model_start_{nullptr};
  uint8_t *tensor_arena_{nullptr};
  uint8_t *var_arena_{nullptr};
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;
  tflite::MicroResourceVariables *mrv_{nullptr};
  tflite::MicroAllocator *ma_{nullptr};
  TfLiteTensor *feature_input_{nullptr};
  TfLiteTensor *probability_output_{nullptr};
  TfLiteTensor *state_input_{nullptr};
  TfLiteTensor *state_output_{nullptr};

#ifdef PIPPA_CRNN_SELF_TEST
  static constexpr size_t CRNN_SELF_TEST_MAX_STEPS = 20;
  bool crnn_boot_self_test_done_{false};
  bool crnn_boot_self_test_passed_{false};
  std::array<CrnnSelfTestStepResult, CRNN_SELF_TEST_MAX_STEPS> crnn_boot_self_test_results_{};
  size_t crnn_boot_self_test_steps_run_{0};
  size_t crnn_boot_self_test_report_index_{0};
  uint32_t crnn_boot_self_test_next_report_ms_{0};
  const char *crnn_boot_self_test_failure_reason_{"not_run"};
  std::atomic<bool> crnn_boot_self_test_report_ready_{false};
#endif

#ifdef PIPPA_CRNN_METRICS
  static constexpr size_t CRNN_METRICS_BUCKET_COUNT = 128;
  static constexpr uint32_t CRNN_METRICS_BUCKET_US = 500;
  static constexpr uint32_t CRNN_METRICS_REPORT_SAMPLES = 10000;
  static constexpr uint32_t CRNN_SCORE_REPORT_SAMPLES = 167;

  void record_crnn_cadence_(int64_t invoke_started_us, uint32_t expected_us);
  void record_crnn_runtime_(uint32_t elapsed_us, uint32_t expected_us);
  void record_crnn_score_(uint8_t probability);
  uint32_t crnn_percentile_bucket_(uint32_t numerator, uint8_t censored_bit, uint8_t *censored_mask) const;

  std::array<uint16_t, CRNN_METRICS_BUCKET_COUNT> crnn_runtime_histogram_{};
  uint32_t crnn_runtime_samples_{0};
  uint64_t crnn_runtime_total_samples_{0};
  uint32_t crnn_runtime_max_us_{0};
  uint32_t crnn_runtime_budget_overruns_{0};
  uint32_t crnn_cold_samples_{0};
  uint32_t crnn_cold_last_us_{0};
  uint32_t crnn_cold_max_us_{0};
  uint32_t crnn_cold_runtime_budget_overruns_{0};
  int64_t crnn_previous_invoke_start_us_{0};
  uint32_t crnn_cadence_intervals_{0};
  uint32_t crnn_cadence_max_gap_us_{0};
  uint32_t crnn_cadence_gap_over_threshold_{0};
  uint32_t crnn_dropped_reports_{0};
  bool crnn_cold_invoke_pending_{false};
  bool crnn_arena_logged_{false};
  CrnnRuntimeMetricsSnapshot crnn_metrics_snapshot_{};
  std::atomic<bool> crnn_metrics_snapshot_pending_{false};
  uint32_t crnn_score_samples_{0};
  uint8_t crnn_score_raw_max_{0};
  uint8_t crnn_score_sliding_average_max_{0};
  uint8_t crnn_input_byte_max_{0};
  uint8_t crnn_input_mean_min_{UINT8_MAX};
  uint8_t crnn_input_mean_max_{0};
  uint32_t crnn_input_delta_l1_max_{0};
  std::array<int8_t, PREPROCESSOR_FEATURE_SIZE * 3> crnn_previous_feature_input_{};
  bool crnn_previous_feature_input_valid_{false};
  CrnnScoreMetricsSnapshot crnn_score_snapshot_{};
  std::atomic<bool> crnn_score_snapshot_pending_{false};
#endif
};

class WakeWordModel final : public StreamingModel {
 public:
  /// @brief Constructs a wake word model object with compile-time model data
  /// @param id (std::string) identifier for this model
  /// @param model_start (const uint8_t *) pointer to the start of the model's TFLite FlatBuffer
  /// @param default_probability_cutoff (uint8_t) probability cutoff for acceping the wake word has been said
  /// @param sliding_window_average_size (size_t) the length of the sliding window computing the mean rolling
  ///                                    probability
  /// @param wake_word (std::string) Friendly name of the wake word
  /// @param tensor_arena_size (size_t) Size in bytes for allocating the tensor arena
  /// @param default_enabled (bool) If true, it will be enabled by default on first boot
  /// @param internal_only (bool) If true, the model will not be exposed to HomeAssistant as an available model
  WakeWordModel(const std::string &id, const uint8_t *model_start, uint8_t default_probability_cutoff,
                size_t sliding_window_average_size, const std::string &wake_word, size_t tensor_arena_size,
                bool default_enabled, bool internal_only);

  /// @brief Constructs a wake word model object with a runtime-downloaded model
  /// @param id (std::string) identifier for this model
  /// @param model_data (std::shared_ptr<ModelData>) owning handle to the downloaded model buffer; must be valid
  /// @param default_probability_cutoff (uint8_t) probability cutoff for acceping the wake word has been said
  /// @param sliding_window_average_size (size_t) the length of the sliding window computing the mean rolling
  ///                                    probability
  /// @param wake_word (std::string) Friendly name of the wake word
  /// @param trained_languages (std::vector<std::string>) Languages the model was trained on
  /// @param tensor_arena_size (size_t) Size in bytes for allocating the tensor arena
  WakeWordModel(const std::string &id, std::shared_ptr<ModelData> model_data, uint8_t default_probability_cutoff,
                size_t sliding_window_average_size, const std::string &wake_word,
                std::vector<std::string> trained_languages, size_t tensor_arena_size);

  // model_data_ is a member of this class, so it is destroyed before ~StreamingModel() runs. Unload here, while
  // the buffer is still alive, so the interpreter is never torn down over freed model data.
  ~WakeWordModel() override { this->unload_model(); }

  void log_model_config() override;

  /// @brief Checks for the wake word by comparing the mean probability in the sliding window with the probability
  /// cutoff
  /// @return True if wake word is detected, false otherwise
  DetectionEvent determine_detected() override;

  const std::string &get_id() const { return this->id_; }
  const std::string &get_wake_word() const { return this->wake_word_; }

  void add_trained_language(const std::string &language) { this->trained_languages_.push_back(language); }
  const std::vector<std::string> &get_trained_languages() const { return this->trained_languages_; }

  /// @brief Enable the model and save to flash. The next performing_streaming_inference call will load it.
  void enable() override;

  /// @brief Disable the model and save to flash. The next performing_streaming_inference call will unload it.
  void disable() override;

  bool get_internal_only() { return this->internal_only_; }

 protected:
  // Kept for runtime-downloaded models so the model buffer stays alive for the model's lifetime.
  // Null for compiled-in models (their data lives in flash).
  std::shared_ptr<ModelData> model_data_;

  std::string id_;
  std::string wake_word_;
  std::vector<std::string> trained_languages_;

  bool internal_only_;

  ESPPreferenceObject pref_;
};

class VADModel final : public StreamingModel {
 public:
  VADModel(const uint8_t *model_start, uint8_t default_probability_cutoff, size_t sliding_window_size,
           size_t tensor_arena_size);

  void log_model_config() override;

  /// @brief Checks for voice activity by comparing the max probability in the sliding window with the probability
  /// cutoff
  /// @return True if voice activity is detected, false otherwise
  DetectionEvent determine_detected() override;
};

}  // namespace esphome::micro_wake_word

#endif
