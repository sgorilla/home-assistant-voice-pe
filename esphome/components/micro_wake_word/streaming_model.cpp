// Derived from ESPHome 2026.8.2 at commit 8b5c06add684cdbef0613e1c295c0f4e6f9dc5c2.
// Copyright (c) 2019 ESPHome.
// Modified 2026-09-06 by the Pippa wake-word project to execute external-state CRNN models.
// SPDX-License-Identifier: GPL-3.0-only

#include "streaming_model.h"

#ifdef PIPPA_CRNN_SELF_TEST
#include "pippa_crnn_self_test_vectors.h"
#endif

#ifdef USE_ESP32

#include "esphome/core/helpers.h"
#include "esphome/core/hal.h"
#include "esphome/core/log.h"

#ifdef PIPPA_CRNN_METRICS
#include <esp_timer.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

static const char *const TAG = "micro_wake_word";

namespace esphome::micro_wake_word {

#ifdef PIPPA_CRNN_SELF_TEST
static constexpr uint32_t CRNN_SELF_TEST_REPORT_DELAY_MS = 20000;
#endif

namespace {

bool tensor_name_contains(const char *name, const char *needle) {
  return name != nullptr && std::strstr(name, needle) != nullptr;
}

const char *flatbuffer_tensor_name(const tflite::Model *model, bool input, size_t position) {
  if (model->subgraphs() == nullptr || model->subgraphs()->size() != 1) {
    return nullptr;
  }
  const auto *subgraph = model->subgraphs()->Get(0);
  const auto *indices = input ? subgraph->inputs() : subgraph->outputs();
  if (indices == nullptr || position >= indices->size()) {
    return nullptr;
  }
  const auto *tensor = subgraph->tensors()->Get(indices->Get(position));
  return tensor->name() == nullptr ? nullptr : tensor->name()->c_str();
}

bool tensor_shape_is(const TfLiteTensor *tensor, int dimensions, int dim0, int dim1) {
  return tensor->dims != nullptr && tensor->dims->size == dimensions && tensor->dims->data[0] == dim0 &&
         tensor->dims->data[1] == dim1;
}

bool is_feature_input(const TfLiteTensor *tensor) {
  return tensor->type == kTfLiteInt8 && tensor->dims != nullptr && tensor->dims->size == 3 &&
         tensor->dims->data[0] == 1 && tensor->dims->data[1] > 0 &&
         tensor->dims->data[2] == PREPROCESSOR_FEATURE_SIZE;
}

bool is_external_state(const TfLiteTensor *tensor) {
  return tensor->type == kTfLiteInt8 &&
         tensor_shape_is(tensor, 2, 1, static_cast<int>(EXTERNAL_STREAMING_STATE_SIZE));
}

bool is_probability_output(const TfLiteTensor *tensor) {
  return (tensor->type == kTfLiteUInt8 || tensor->type == kTfLiteInt8) && tensor_shape_is(tensor, 2, 1, 1);
}

}  // namespace

void WakeWordModel::log_model_config() {
  ESP_LOGCONFIG(TAG,
                "    - Wake Word: %s\n"
                "      Probability cutoff: %.2f\n"
                "      Sliding window size: %d",
                this->wake_word_.c_str(), this->get_probability_cutoff() / 255.0f, this->sliding_window_size_);
}

void VADModel::log_model_config() {
  ESP_LOGCONFIG(TAG,
                "    - VAD Model\n"
                "      Probability cutoff: %.2f\n"
                "      Sliding window size: %d",
                this->get_probability_cutoff() / 255.0f, this->sliding_window_size_);
}

bool StreamingModel::load_model_() {
  if (this->model_start_ == nullptr) {
    ESP_LOGE(TAG, "Streaming model has no data to load");
    return false;
  }

  RAMAllocator<uint8_t> arena_allocator;

  if (this->var_arena_ == nullptr) {
    this->var_arena_ = arena_allocator.allocate(STREAMING_MODEL_VARIABLE_ARENA_SIZE);
    if (this->var_arena_ == nullptr) {
      ESP_LOGE(TAG, "Could not allocate the streaming model's variable tensor arena.");
      return false;
    }
    this->ma_ = tflite::MicroAllocator::Create(this->var_arena_, STREAMING_MODEL_VARIABLE_ARENA_SIZE);
    this->mrv_ = tflite::MicroResourceVariables::Create(this->ma_, 20);
  }

  const tflite::Model *model = tflite::GetModel(this->model_start_);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Streaming model's schema is not supported");
    return false;
  }

  // Probe for the actual required tensor arena size if not yet determined
  if (!this->tensor_arena_size_probed_) {
    size_t probed_size = this->probe_arena_size_();
    if (probed_size > 0) {
      ESP_LOGD(TAG, "Probed tensor arena size: %zu bytes", probed_size);
      this->tensor_arena_size_ = probed_size;
    } else {
      ESP_LOGW(TAG, "Arena size probe failed, using manifest size: %zu bytes", this->tensor_arena_size_);
    }
    this->tensor_arena_size_probed_ = true;
  }

  if (this->tensor_arena_ == nullptr) {
    this->tensor_arena_ = arena_allocator.allocate(this->tensor_arena_size_);
    if (this->tensor_arena_ == nullptr) {
      ESP_LOGE(TAG, "Could not allocate the streaming model's tensor arena.");
      return false;
    }
  }

  if (this->interpreter_ == nullptr) {
    this->interpreter_ =
        make_unique<tflite::MicroInterpreter>(tflite::GetModel(this->model_start_), this->streaming_op_resolver_,
                                              this->tensor_arena_, this->tensor_arena_size_, this->mrv_);
    if (this->interpreter_->AllocateTensors() != kTfLiteOk) {
      ESP_LOGE(TAG, "Failed to allocate tensors for the streaming model");
      return false;
    }

    if (!this->discover_streaming_tensors_()) {
      return false;
    }

#ifdef PIPPA_CRNN_METRICS
    if (this->state_input_ != nullptr) {
      if (!this->crnn_arena_logged_) {
        ESP_LOGI(TAG, "CRNN_ARENA tensor_used=%zu tensor_allocated=%zu variable_used=%zu variable_allocated=%zu",
                 this->interpreter_->arena_used_bytes(), this->tensor_arena_size_, this->ma_->used_bytes(),
                 static_cast<size_t>(STREAMING_MODEL_VARIABLE_ARENA_SIZE));
        this->crnn_arena_logged_ = true;
      }
      this->crnn_cold_invoke_pending_ = true;
    }
#endif

    // Clear the initial request before applying it so a concurrent request made during setup remains pending.
    this->streaming_state_reset_requested_.store(false);
    this->apply_streaming_state_reset_();
  }

  this->loaded_ = true;
  this->reset_probabilities();
  return true;
}

size_t StreamingModel::probe_arena_size_() {
  RAMAllocator<uint8_t> arena_allocator;

  // Try with the manifest size first, then escalates to 1.5, then 2x if it fails. Different platforms and different
  // versions of the esp-nn library require different amounts of memory, so the manifest size may not always be correct,
  // and probing allows us to find the actual required size for the current build and platform. Aligns test sizes to 16
  // bytes.
  size_t attempt_sizes[] = {(this->tensor_arena_size_ + 15) & ~15, (this->tensor_arena_size_ * 3 / 2 + 15) & ~15,
                            (this->tensor_arena_size_ * 2 + 15) & ~15};

  for (size_t attempt_size : attempt_sizes) {
    uint8_t *probe_arena = arena_allocator.allocate(attempt_size);
    if (probe_arena == nullptr) {
      continue;
    }

    // Verify the model works at all with this arena size
    auto probe_interpreter = make_unique<tflite::MicroInterpreter>(
        tflite::GetModel(this->model_start_), this->streaming_op_resolver_, probe_arena, attempt_size, this->mrv_);

    if (probe_interpreter->AllocateTensors() != kTfLiteOk) {
      probe_interpreter.reset();
      arena_allocator.deallocate(probe_arena, attempt_size);
      this->ma_ = tflite::MicroAllocator::Create(this->var_arena_, STREAMING_MODEL_VARIABLE_ARENA_SIZE);
      this->mrv_ = tflite::MicroResourceVariables::Create(this->ma_, 20);
      continue;
    }

    // Try to shrink the arena. Start with arena_used_bytes() + 16 (rounded to 16-byte alignment).
    // If that works, use it. Otherwise, try midpoints between that and the full size until one succeeds.
    size_t lower = (probe_interpreter->arena_used_bytes() + 16 + 15) & ~15;
    probe_interpreter.reset();
    this->ma_ = tflite::MicroAllocator::Create(this->var_arena_, STREAMING_MODEL_VARIABLE_ARENA_SIZE);
    this->mrv_ = tflite::MicroResourceVariables::Create(this->ma_, 20);

    size_t upper = attempt_size;

    while (lower < upper) {
      auto test_interpreter = make_unique<tflite::MicroInterpreter>(
          tflite::GetModel(this->model_start_), this->streaming_op_resolver_, probe_arena, lower, this->mrv_);

      bool ok = test_interpreter->AllocateTensors() == kTfLiteOk;

      test_interpreter.reset();
      this->ma_ = tflite::MicroAllocator::Create(this->var_arena_, STREAMING_MODEL_VARIABLE_ARENA_SIZE);
      this->mrv_ = tflite::MicroResourceVariables::Create(this->ma_, 20);

      if (ok) {
        // Found a working size smaller than the full arena
        upper = lower + 16;  // Pad by 16 bytes to be safe for future allocations
        break;
      }

      // Try the midpoint between current attempt and full size
      lower = ((lower + upper) / 2 + 15) & ~15;
    }

    arena_allocator.deallocate(probe_arena, attempt_size);
    return upper;
  }

  return 0;
}

void StreamingModel::unload_model() {
  this->interpreter_.reset();
  this->feature_input_ = nullptr;
  this->probability_output_ = nullptr;
  this->state_input_ = nullptr;
  this->state_output_ = nullptr;
  this->streaming_state_reset_requested_.store(true);

  RAMAllocator<uint8_t> arena_allocator;

  if (this->tensor_arena_ != nullptr) {
    arena_allocator.deallocate(this->tensor_arena_, this->tensor_arena_size_);
    this->tensor_arena_ = nullptr;
  }

  if (this->var_arena_ != nullptr) {
    arena_allocator.deallocate(this->var_arena_, STREAMING_MODEL_VARIABLE_ARENA_SIZE);
    this->var_arena_ = nullptr;
  }

  this->loaded_ = false;
}

bool StreamingModel::perform_streaming_inference(const int8_t features[PREPROCESSOR_FEATURE_SIZE],
                                                 uint32_t feature_step_us) {
  if (this->model_start_ == nullptr) {
    // No usable model data, and that cannot change for this object. Skip the model instead of reporting a
    // failure, because a false return here stops the inference task for every other model too.
    this->enabled_ = false;
    return true;
  }

  if (this->enabled_ && !this->loaded_) {
    // Model is enabled but isn't loaded
    if (!this->load_model_()) {
      return false;
    }
  }

  if (!this->enabled_ && this->loaded_) {
    // Model is disabled but still loaded
    this->unload_model();
    return true;
  }

  if (this->loaded_) {
    if (this->streaming_state_reset_requested_.exchange(false)) {
      this->apply_streaming_state_reset_();
    }

    uint8_t stride = this->feature_input_->dims->data[1];
    this->current_stride_step_ = this->current_stride_step_ % stride;

    std::memmove(
        tflite::GetTensorData<int8_t>(this->feature_input_) + PREPROCESSOR_FEATURE_SIZE * this->current_stride_step_,
        features, PREPROCESSOR_FEATURE_SIZE);
    ++this->current_stride_step_;

    if (this->current_stride_step_ >= stride) {
#ifdef PIPPA_CRNN_METRICS
      const int64_t crnn_invoke_started_us = this->state_input_ == nullptr ? 0 : esp_timer_get_time();
      if (crnn_invoke_started_us != 0) {
        this->record_crnn_cadence_(crnn_invoke_started_us, static_cast<uint32_t>(stride) * feature_step_us);
      }
#endif
      TfLiteStatus invoke_status = this->interpreter_->Invoke();
      if (invoke_status != kTfLiteOk) {
        ESP_LOGW(TAG, "Streaming interpreter invoke failed");
        return false;
      }

      if (!this->feed_back_streaming_state_()) {
        return false;
      }

#ifdef PIPPA_CRNN_METRICS
      if (crnn_invoke_started_us != 0) {
        const uint32_t elapsed_us = static_cast<uint32_t>(esp_timer_get_time() - crnn_invoke_started_us);
        this->record_crnn_runtime_(elapsed_us, static_cast<uint32_t>(stride) * feature_step_us);
      }
#endif

      ++this->last_n_index_;
      if (this->last_n_index_ == this->sliding_window_size_)
        this->last_n_index_ = 0;
      const uint8_t probability = this->read_probability_();
      this->recent_streaming_probabilities_[this->last_n_index_] = probability;
#ifdef PIPPA_CRNN_METRICS
      if (this->state_input_ != nullptr) {
        this->record_crnn_score_(probability);
      }
#endif
      this->unprocessed_probability_status_ = true;
    }
    if (this->recent_streaming_probabilities_[this->last_n_index_] < this->get_probability_cutoff()) {
      // Only increment ignore windows if less than the probability cutoff; this forces the model to "cool-off" from a
      // previous detection and calling ``reset_probabilities`` so it avoids duplicate detections
      this->ignore_windows_ = std::min(this->ignore_windows_ + 1, 0);
    }
  }
  return true;
}

#ifdef PIPPA_CRNN_SELF_TEST
bool StreamingModel::run_crnn_boot_self_test() {
  static_assert(crnn_self_test::INVOCATION_COUNT <= CRNN_SELF_TEST_MAX_STEPS,
                "CRNN self-test result storage is too small");
  static_assert(sizeof(crnn_self_test::EXPECTED_RAW_PROBABILITY) == crnn_self_test::INVOCATION_COUNT,
                "CRNN self-test probability fixture length mismatch");
  static_assert(sizeof(crnn_self_test::FINAL_EXPECTED_STATE) == crnn_self_test::STATE_BYTES,
                "CRNN self-test final-state fixture length mismatch");
  if (this->crnn_boot_self_test_done_) {
    return this->crnn_boot_self_test_passed_;
  }
  this->crnn_boot_self_test_done_ = true;
  this->crnn_boot_self_test_passed_ = false;
  this->crnn_boot_self_test_steps_run_ = 0;
  this->crnn_boot_self_test_report_index_ = 0;
  this->crnn_boot_self_test_failure_reason_ = "none";

  if (!this->load_model_()) {
    this->crnn_boot_self_test_failure_reason_ = "load_failed";
    this->unload_model();
    this->crnn_boot_self_test_next_report_ms_ = millis() + CRNN_SELF_TEST_REPORT_DELAY_MS;
    this->crnn_boot_self_test_report_ready_.store(true, std::memory_order_release);
    return false;
  }

  const bool abi_ok = this->state_input_ != nullptr && this->state_output_ != nullptr &&
                      this->feature_input_->bytes == crnn_self_test::FEATURE_BYTES &&
                      this->state_input_->bytes == crnn_self_test::STATE_BYTES &&
                      this->state_output_->bytes == crnn_self_test::STATE_BYTES &&
                      this->probability_output_->type == kTfLiteInt8 && this->probability_output_->bytes == 1 &&
                      this->probability_output_->params.scale == (1.0f / 256.0f) &&
                      this->probability_output_->params.zero_point == INT8_MIN;
  if (!abi_ok) {
    this->crnn_boot_self_test_failure_reason_ = "abi_mismatch";
    this->unload_model();
    this->crnn_boot_self_test_next_report_ms_ = millis() + CRNN_SELF_TEST_REPORT_DELAY_MS;
    this->crnn_boot_self_test_report_ready_.store(true, std::memory_order_release);
    return false;
  }

  const uint8_t *first_record = crnn_self_test::CRNN_SELF_TEST_INPUT_RECORDS;
  std::memcpy(tflite::GetTensorData<int8_t>(this->state_input_), first_record + crnn_self_test::FEATURE_BYTES,
              crnn_self_test::STATE_BYTES);

  bool passed = true;
  for (size_t invocation = 0; invocation < crnn_self_test::INVOCATION_COUNT; ++invocation) {
    CrnnSelfTestStepResult &result = this->crnn_boot_self_test_results_[this->crnn_boot_self_test_steps_run_++];
    result.record = crnn_self_test::FIRST_RECORD + invocation;
    result.expected_raw = crnn_self_test::EXPECTED_RAW_PROBABILITY[invocation];
    result.expected_score = static_cast<uint8_t>(static_cast<unsigned>(result.expected_raw) + 128U);
    result.invoke_ok = false;
    result.feedback_ok = false;
    result.passed = false;

    const uint8_t *record = crnn_self_test::CRNN_SELF_TEST_INPUT_RECORDS +
                            invocation * crnn_self_test::INPUT_RECORD_BYTES;
    std::memcpy(tflite::GetTensorData<int8_t>(this->feature_input_), record, crnn_self_test::FEATURE_BYTES);

    if (this->interpreter_->Invoke() != kTfLiteOk) {
      this->crnn_boot_self_test_failure_reason_ = "invoke_failed";
      passed = false;
      break;
    }
    result.invoke_ok = true;

    result.actual_raw = static_cast<uint8_t>(this->probability_output_->data.int8[0]);
    result.actual_score = this->read_probability_();

    if (!this->feed_back_streaming_state_()) {
      this->crnn_boot_self_test_failure_reason_ = "feedback_failed";
      passed = false;
      break;
    }
    result.feedback_ok = true;

    const uint8_t *expected_state;
    if (invocation + 1 < crnn_self_test::INPUT_RECORD_COUNT) {
      const uint8_t *next_record = record + crnn_self_test::INPUT_RECORD_BYTES;
      expected_state = next_record + crnn_self_test::FEATURE_BYTES;
    } else {
      expected_state = crnn_self_test::FINAL_EXPECTED_STATE;
    }
    const int8_t *actual_state = tflite::GetTensorData<int8_t>(this->state_input_);
    result.state_exact = 0;
    result.state_within_one = 0;
    result.state_max_abs_diff = 0;
    for (size_t state_index = 0; state_index < crnn_self_test::STATE_BYTES; ++state_index) {
      int8_t expected_signed;
      std::memcpy(&expected_signed, &expected_state[state_index], sizeof(expected_signed));
      const bool byte_exact = static_cast<uint8_t>(actual_state[state_index]) == expected_state[state_index];
      const uint8_t absolute_difference = static_cast<uint8_t>(std::abs(
          static_cast<int>(actual_state[state_index]) - static_cast<int>(expected_signed)));
      if (byte_exact) {
        ++result.state_exact;
      }
      if (absolute_difference <= 1) {
        ++result.state_within_one;
      }
      result.state_max_abs_diff = std::max(result.state_max_abs_diff, absolute_difference);
    }

    result.passed = result.actual_raw == result.expected_raw && result.state_exact == crnn_self_test::STATE_BYTES;
    passed = passed && result.passed;
  }

  // Recreate the interpreter before live audio so the self-test cannot leave recurrent
  // or interpreter state in the measurement run. CPU caches may remain warm briefly.
  this->unload_model();
  this->crnn_boot_self_test_passed_ = passed;
  if (!passed && std::strcmp(this->crnn_boot_self_test_failure_reason_, "none") == 0) {
    this->crnn_boot_self_test_failure_reason_ = "byte_mismatch";
  }
  this->crnn_boot_self_test_next_report_ms_ = millis() + CRNN_SELF_TEST_REPORT_DELAY_MS;
  this->crnn_boot_self_test_report_ready_.store(true, std::memory_order_release);
  return passed;
}

void StreamingModel::report_crnn_boot_self_test() {
  if (!this->crnn_boot_self_test_report_ready_.load(std::memory_order_acquire)) {
    return;
  }
  const uint32_t now = millis();
  if (now < this->crnn_boot_self_test_next_report_ms_) {
    return;
  }

  if (this->crnn_boot_self_test_report_index_ < this->crnn_boot_self_test_steps_run_) {
    const CrnnSelfTestStepResult &result =
        this->crnn_boot_self_test_results_[this->crnn_boot_self_test_report_index_++];
    ESP_LOGI(TAG,
             "CRNN_SELFTEST record=%u invoke_ok=%u feedback_ok=%u expected_raw=%u actual_raw=%u "
             "expected_score=%u actual_score=%u state_exact=%u/%zu state_within1=%u/%zu "
             "state_max_abs_diff=%u passed=%u",
             static_cast<unsigned>(result.record), static_cast<unsigned>(result.invoke_ok),
             static_cast<unsigned>(result.feedback_ok), static_cast<unsigned>(result.expected_raw),
             static_cast<unsigned>(result.actual_raw), static_cast<unsigned>(result.expected_score),
             static_cast<unsigned>(result.actual_score), static_cast<unsigned>(result.state_exact),
             crnn_self_test::STATE_BYTES, static_cast<unsigned>(result.state_within_one),
             crnn_self_test::STATE_BYTES, static_cast<unsigned>(result.state_max_abs_diff),
             static_cast<unsigned>(result.passed));
    this->crnn_boot_self_test_next_report_ms_ = now + 200;
    return;
  }

  ESP_LOGI(TAG, "CRNN_SELFTEST_RESULT passed=%u records=%zu first=%zu last=%zu reason=%s",
           static_cast<unsigned>(this->crnn_boot_self_test_passed_), this->crnn_boot_self_test_steps_run_,
           crnn_self_test::FIRST_RECORD,
           crnn_self_test::FIRST_RECORD +
               (this->crnn_boot_self_test_steps_run_ == 0 ? 0 : this->crnn_boot_self_test_steps_run_ - 1),
           this->crnn_boot_self_test_failure_reason_);
  this->crnn_boot_self_test_report_ready_.store(false, std::memory_order_release);
}
#endif

#ifdef PIPPA_CRNN_METRICS
void StreamingModel::record_crnn_score_(uint8_t probability) {
  this->crnn_score_raw_max_ = std::max(this->crnn_score_raw_max_, probability);

  const int8_t *feature_data = tflite::GetTensorData<int8_t>(this->feature_input_);
  uint32_t input_sum = 0;
  uint8_t input_byte_max = 0;
  for (size_t i = 0; i < this->feature_input_->bytes; ++i) {
    const uint8_t rebased = static_cast<uint8_t>(static_cast<int16_t>(feature_data[i]) - INT8_MIN);
    input_sum += rebased;
    input_byte_max = std::max(input_byte_max, rebased);
  }
  const uint8_t input_mean = static_cast<uint8_t>(input_sum / this->feature_input_->bytes);
  this->crnn_input_byte_max_ = std::max(this->crnn_input_byte_max_, input_byte_max);
  this->crnn_input_mean_min_ = std::min(this->crnn_input_mean_min_, input_mean);
  this->crnn_input_mean_max_ = std::max(this->crnn_input_mean_max_, input_mean);

  if (this->feature_input_->bytes == this->crnn_previous_feature_input_.size()) {
    if (this->crnn_previous_feature_input_valid_) {
      uint32_t delta_l1 = 0;
      for (size_t i = 0; i < this->feature_input_->bytes; ++i) {
        delta_l1 += static_cast<uint32_t>(
            std::abs(static_cast<int>(feature_data[i]) - static_cast<int>(this->crnn_previous_feature_input_[i])));
      }
      this->crnn_input_delta_l1_max_ = std::max(this->crnn_input_delta_l1_max_, delta_l1);
    }
    std::memcpy(this->crnn_previous_feature_input_.data(), feature_data, this->feature_input_->bytes);
    this->crnn_previous_feature_input_valid_ = true;
  } else {
    this->crnn_previous_feature_input_valid_ = false;
  }

  uint32_t sum = 0;
  for (const auto recent_probability : this->recent_streaming_probabilities_) {
    sum += recent_probability;
  }
  const uint8_t sliding_average = static_cast<uint8_t>(sum / this->sliding_window_size_);
  this->crnn_score_sliding_average_max_ = std::max(this->crnn_score_sliding_average_max_, sliding_average);
  ++this->crnn_score_samples_;

  if (this->crnn_score_samples_ < CRNN_SCORE_REPORT_SAMPLES) {
    return;
  }

  if (!this->crnn_score_snapshot_pending_.load(std::memory_order_acquire)) {
    this->crnn_score_snapshot_.samples = this->crnn_score_samples_;
    this->crnn_score_snapshot_.raw_max = this->crnn_score_raw_max_;
    this->crnn_score_snapshot_.sliding_average_max = this->crnn_score_sliding_average_max_;
    this->crnn_score_snapshot_.input_byte_max = this->crnn_input_byte_max_;
    this->crnn_score_snapshot_.input_mean_min = this->crnn_input_mean_min_;
    this->crnn_score_snapshot_.input_mean_max = this->crnn_input_mean_max_;
    this->crnn_score_snapshot_.input_delta_l1_max = this->crnn_input_delta_l1_max_;
    this->crnn_score_snapshot_pending_.store(true, std::memory_order_release);
  }

  this->crnn_score_samples_ = 0;
  this->crnn_score_raw_max_ = 0;
  this->crnn_score_sliding_average_max_ = 0;
  this->crnn_input_byte_max_ = 0;
  this->crnn_input_mean_min_ = UINT8_MAX;
  this->crnn_input_mean_max_ = 0;
  this->crnn_input_delta_l1_max_ = 0;
}

uint32_t StreamingModel::crnn_percentile_bucket_(uint32_t numerator, uint8_t censored_bit,
                                                 uint8_t *censored_mask) const {
  const uint32_t target = (this->crnn_runtime_samples_ * numerator + 99U) / 100U;
  uint32_t cumulative = 0;
  for (size_t i = 0; i < this->crnn_runtime_histogram_.size(); ++i) {
    cumulative += this->crnn_runtime_histogram_[i];
    if (cumulative >= target) {
      if (i == this->crnn_runtime_histogram_.size() - 1) {
        *censored_mask |= censored_bit;
        return (this->crnn_runtime_histogram_.size() - 1) * CRNN_METRICS_BUCKET_US;
      }
      return (i + 1) * CRNN_METRICS_BUCKET_US;
    }
  }
  return 0;
}

void StreamingModel::record_crnn_cadence_(int64_t invoke_started_us, uint32_t expected_us) {
  if (this->crnn_previous_invoke_start_us_ != 0) {
    const uint32_t gap_us = static_cast<uint32_t>(invoke_started_us - this->crnn_previous_invoke_start_us_);
    ++this->crnn_cadence_intervals_;
    this->crnn_cadence_max_gap_us_ = std::max(this->crnn_cadence_max_gap_us_, gap_us);
    if (gap_us > expected_us) {
      ++this->crnn_cadence_gap_over_threshold_;
    }
  }
  this->crnn_previous_invoke_start_us_ = invoke_started_us;
}

void StreamingModel::record_crnn_runtime_(uint32_t elapsed_us, uint32_t expected_us) {
  if (this->crnn_cold_invoke_pending_) {
    this->crnn_cold_invoke_pending_ = false;
    ++this->crnn_cold_samples_;
    this->crnn_cold_last_us_ = elapsed_us;
    this->crnn_cold_max_us_ = std::max(this->crnn_cold_max_us_, elapsed_us);
    if (elapsed_us > expected_us) {
      ++this->crnn_cold_runtime_budget_overruns_;
    }
    return;
  }

  const size_t bucket =
      std::min<size_t>(elapsed_us / CRNN_METRICS_BUCKET_US, this->crnn_runtime_histogram_.size() - 1);
  ++this->crnn_runtime_histogram_[bucket];
  ++this->crnn_runtime_samples_;
  ++this->crnn_runtime_total_samples_;
  this->crnn_runtime_max_us_ = std::max(this->crnn_runtime_max_us_, elapsed_us);
  if (elapsed_us > expected_us) {
    ++this->crnn_runtime_budget_overruns_;
  }

  if (this->crnn_runtime_samples_ < CRNN_METRICS_REPORT_SAMPLES) {
    return;
  }

  if (!this->crnn_metrics_snapshot_pending_.load(std::memory_order_acquire)) {
    uint8_t censored_mask = 0;
    this->crnn_metrics_snapshot_.samples = this->crnn_runtime_samples_;
    this->crnn_metrics_snapshot_.total_samples = this->crnn_runtime_total_samples_;
    this->crnn_metrics_snapshot_.runtime_budget_us = expected_us;
    this->crnn_metrics_snapshot_.p50_bucket_us = this->crnn_percentile_bucket_(50, 1 << 0, &censored_mask);
    this->crnn_metrics_snapshot_.p95_bucket_us = this->crnn_percentile_bucket_(95, 1 << 1, &censored_mask);
    this->crnn_metrics_snapshot_.p99_bucket_us = this->crnn_percentile_bucket_(99, 1 << 2, &censored_mask);
    this->crnn_metrics_snapshot_.runtime_max_us = this->crnn_runtime_max_us_;
    this->crnn_metrics_snapshot_.runtime_budget_overruns = this->crnn_runtime_budget_overruns_;
    this->crnn_metrics_snapshot_.cold_samples = this->crnn_cold_samples_;
    this->crnn_metrics_snapshot_.cold_last_us = this->crnn_cold_last_us_;
    this->crnn_metrics_snapshot_.cold_max_us = this->crnn_cold_max_us_;
    this->crnn_metrics_snapshot_.cold_runtime_budget_overruns = this->crnn_cold_runtime_budget_overruns_;
    this->crnn_metrics_snapshot_.cadence_intervals = this->crnn_cadence_intervals_;
    this->crnn_metrics_snapshot_.cadence_threshold_us = expected_us;
    this->crnn_metrics_snapshot_.cadence_max_gap_us = this->crnn_cadence_max_gap_us_;
    this->crnn_metrics_snapshot_.cadence_gap_over_threshold = this->crnn_cadence_gap_over_threshold_;
    this->crnn_metrics_snapshot_.dropped_reports = this->crnn_dropped_reports_;
    this->crnn_metrics_snapshot_.percentile_censored_mask = censored_mask;
    this->crnn_metrics_snapshot_pending_.store(true, std::memory_order_release);
  } else {
    ++this->crnn_dropped_reports_;
  }

  this->crnn_runtime_histogram_.fill(0);
  this->crnn_runtime_samples_ = 0;
  this->crnn_runtime_max_us_ = 0;
  this->crnn_runtime_budget_overruns_ = 0;
  this->crnn_cold_samples_ = 0;
  this->crnn_cold_last_us_ = 0;
  this->crnn_cold_max_us_ = 0;
  this->crnn_cold_runtime_budget_overruns_ = 0;
  this->crnn_cadence_intervals_ = 0;
  this->crnn_cadence_max_gap_us_ = 0;
  this->crnn_cadence_gap_over_threshold_ = 0;
}

bool StreamingModel::take_crnn_metrics_snapshot(CrnnRuntimeMetricsSnapshot *snapshot) {
  if (!this->crnn_metrics_snapshot_pending_.load(std::memory_order_acquire)) {
    return false;
  }
  *snapshot = this->crnn_metrics_snapshot_;
  this->crnn_metrics_snapshot_pending_.store(false, std::memory_order_release);
  return true;
}

bool StreamingModel::take_crnn_score_snapshot(CrnnScoreMetricsSnapshot *snapshot) {
  if (!this->crnn_score_snapshot_pending_.load(std::memory_order_acquire)) {
    return false;
  }
  *snapshot = this->crnn_score_snapshot_;
  this->crnn_score_snapshot_pending_.store(false, std::memory_order_release);
  return true;
}
#endif

bool StreamingModel::discover_streaming_tensors_() {
  this->feature_input_ = nullptr;
  this->probability_output_ = nullptr;
  this->state_input_ = nullptr;
  this->state_output_ = nullptr;

  const tflite::Model *model = tflite::GetModel(this->model_start_);
  const size_t input_count = this->interpreter_->inputs_size();
  const size_t output_count = this->interpreter_->outputs_size();
  if (!((input_count == 1 && output_count == 1) || (input_count == 2 && output_count == 2))) {
    ESP_LOGE(TAG, "Streaming model must have exactly 1 input/output (stock) or 2 inputs/outputs (external state)");
    return false;
  }

  for (size_t i = 0; i < input_count; ++i) {
    TfLiteTensor *tensor = this->interpreter_->input(i);
    const char *name = flatbuffer_tensor_name(model, true, i);
    if (is_feature_input(tensor)) {
      if (this->feature_input_ != nullptr) {
        ESP_LOGE(TAG, "Streaming model has more than one feature-shaped input");
        return false;
      }
      this->feature_input_ = tensor;
      if (!tensor_name_contains(name, "feature")) {
        ESP_LOGV(TAG, "Feature input identified by shape; FlatBuffer tensor name is '%s'",
                 name == nullptr ? "<unnamed>" : name);
      }
    } else if (is_external_state(tensor)) {
      if (this->state_input_ != nullptr) {
        ESP_LOGE(TAG, "Streaming model has more than one 1x%zu state input", EXTERNAL_STREAMING_STATE_SIZE);
        return false;
      }
      this->state_input_ = tensor;
      if (!tensor_name_contains(name, "state")) {
        ESP_LOGW(TAG, "External state input identified by shape; FlatBuffer tensor name is '%s'",
                 name == nullptr ? "<unnamed>" : name);
      }
    }
  }

  for (size_t i = 0; i < output_count; ++i) {
    TfLiteTensor *tensor = this->interpreter_->output(i);
    if (is_probability_output(tensor)) {
      if (this->probability_output_ != nullptr) {
        ESP_LOGE(TAG, "Streaming model has more than one 1x1 probability-shaped output");
        return false;
      }
      this->probability_output_ = tensor;
    } else if (is_external_state(tensor)) {
      if (this->state_output_ != nullptr) {
        ESP_LOGE(TAG, "Streaming model has more than one 1x%zu state output", EXTERNAL_STREAMING_STATE_SIZE);
        return false;
      }
      this->state_output_ = tensor;
    }
  }

  if (this->feature_input_ == nullptr || this->probability_output_ == nullptr) {
    ESP_LOGE(TAG, "Streaming model is missing its feature input or probability output");
    return false;
  }

  if (this->probability_output_->type == kTfLiteInt8 && this->probability_output_->params.scale <= 0.0f) {
    ESP_LOGE(TAG, "Int8 probability output requires a positive per-tensor quantization scale");
    return false;
  }

  // A stock model has neither state tensor. A CRNN must expose both halves of the feedback pair.
  if ((this->state_input_ == nullptr) != (this->state_output_ == nullptr)) {
    ESP_LOGE(TAG, "Streaming model must provide both a state input and state output");
    return false;
  }

  const bool has_external_state = this->state_input_ != nullptr;
  if ((has_external_state && (input_count != 2 || output_count != 2)) ||
      (!has_external_state && (input_count != 1 || output_count != 1))) {
    ESP_LOGE(TAG, "Streaming model tensor cardinality does not match its state ABI");
    return false;
  }

  if (this->state_input_ != nullptr) {
    constexpr float canonical_feature_scale = 26.0f / 255.0f;
    constexpr float feature_scale_tolerance = 1.0e-6f;
    if (std::fabs(this->feature_input_->params.scale - canonical_feature_scale) > feature_scale_tolerance ||
        this->feature_input_->params.zero_point != INT8_MIN) {
      ESP_LOGE(TAG, "External-state model feature quantization does not match the Voice PE frontend ABI");
      return false;
    }
    if (this->state_input_->params.scale <= 0.0f || this->state_output_->params.scale <= 0.0f) {
      ESP_LOGE(TAG, "External state tensors require positive per-tensor quantization scales");
      return false;
    }
    ESP_LOGCONFIG(TAG, "Streaming model uses external recurrent state (1x%zu)", EXTERNAL_STREAMING_STATE_SIZE);
  }

  return true;
}

void StreamingModel::apply_streaming_state_reset_() {
  // Never combine feature frames from opposite sides of any reset boundary. This applies to both the stock K=0 path
  // and external-state models; only the state-tensor work below is conditional.
  const int8_t feature_zero = static_cast<int8_t>(clamp<int32_t>(this->feature_input_->params.zero_point, INT8_MIN,
                                                                 INT8_MAX));
  std::memset(tflite::GetTensorData<int8_t>(this->feature_input_), feature_zero, this->feature_input_->bytes);
  this->current_stride_step_ = 0;

#ifdef PIPPA_CRNN_METRICS
  // A model-context reset is a real stream boundary, so the next invoke has no valid predecessor interval.
  this->crnn_previous_invoke_start_us_ = 0;
  this->crnn_previous_feature_input_valid_ = false;
#endif

  if (this->state_input_ == nullptr) {
    return;
  }

  const int8_t quantized_zero = static_cast<int8_t>(clamp<int32_t>(this->state_input_->params.zero_point, INT8_MIN,
                                                                   INT8_MAX));
  std::memset(tflite::GetTensorData<int8_t>(this->state_input_), quantized_zero, this->state_input_->bytes);
}

bool StreamingModel::feed_back_streaming_state_() {
  if (this->state_input_ == nullptr) {
    return true;
  }

  if (this->state_input_->bytes != this->state_output_->bytes) {
    ESP_LOGE(TAG, "External state input/output byte sizes differ");
    return false;
  }

  const int8_t *source = tflite::GetTensorData<int8_t>(this->state_output_);
  int8_t *destination = tflite::GetTensorData<int8_t>(this->state_input_);
  const float input_scale = this->state_input_->params.scale;
  const float output_scale = this->state_output_->params.scale;
  const int32_t input_zero_point = this->state_input_->params.zero_point;
  const int32_t output_zero_point = this->state_output_->params.zero_point;

  if (input_scale == output_scale && input_zero_point == output_zero_point) {
    std::memmove(destination, source, this->state_input_->bytes);
    return true;
  }

  for (size_t i = 0; i < this->state_input_->bytes; ++i) {
    const float real_value = (static_cast<int32_t>(source[i]) - output_zero_point) * output_scale;
    const int32_t requantized = static_cast<int32_t>(std::lround(real_value / input_scale)) + input_zero_point;
    destination[i] = static_cast<int8_t>(clamp<int32_t>(requantized, INT8_MIN, INT8_MAX));
  }
  return true;
}

uint8_t StreamingModel::read_probability_() const {
  if (this->probability_output_->type == kTfLiteUInt8) {
    return this->probability_output_->data.uint8[0];
  }

  const int32_t quantized = this->probability_output_->data.int8[0];
  if (this->probability_output_->params.scale == (1.0f / 256.0f) &&
      this->probability_output_->params.zero_point == INT8_MIN) {
    // The exported sigmoid's canonical int8 ABI is byte-for-byte equivalent to uint8 after rebiasing.
    return static_cast<uint8_t>(quantized - INT8_MIN);
  }
  const float probability =
      (quantized - this->probability_output_->params.zero_point) * this->probability_output_->params.scale;
  return static_cast<uint8_t>(clamp<int32_t>(static_cast<int32_t>(std::lround(probability * 255.0f)), 0, 255));
}

void StreamingModel::reset_probabilities() {
  for (auto &prob : this->recent_streaming_probabilities_) {
    prob = 0;
  }
  this->ignore_windows_ = -MIN_SLICES_BEFORE_DETECTION;
}

WakeWordModel::WakeWordModel(const std::string &id, const uint8_t *model_start, uint8_t default_probability_cutoff,
                             size_t sliding_window_average_size, const std::string &wake_word, size_t tensor_arena_size,
                             bool default_enabled, bool internal_only) {
  this->id_ = id;
  this->model_start_ = model_start;
  this->default_probability_cutoff_ = default_probability_cutoff;
  this->probability_cutoff_.store(default_probability_cutoff, std::memory_order_relaxed);
  this->sliding_window_size_ = sliding_window_average_size;
  this->recent_streaming_probabilities_.resize(sliding_window_average_size, 0);
  this->wake_word_ = wake_word;
  this->tensor_arena_size_ = tensor_arena_size;
  this->register_streaming_ops_(this->streaming_op_resolver_);
  this->current_stride_step_ = 0;
  this->internal_only_ = internal_only;

  this->pref_ = global_preferences->make_preference<bool>(fnv1_hash(id));
  bool enabled;
  if (this->pref_.load(&enabled)) {
    // Use the enabled state loaded from flash
    this->enabled_ = enabled;
  } else {
    // If no state saved, then use the default
    this->enabled_ = default_enabled;
  }
};

WakeWordModel::WakeWordModel(const std::string &id, std::shared_ptr<ModelData> model_data,
                             uint8_t default_probability_cutoff, size_t sliding_window_average_size,
                             const std::string &wake_word, std::vector<std::string> trained_languages,
                             size_t tensor_arena_size) {
  this->id_ = id;
  this->model_data_ = std::move(model_data);
  // Callers are expected to pass a validated buffer, so this is normally the stable model pointer. Tolerate a
  // null or unvalidated handle rather than dereferencing it blindly: model_start_ stays null and the model is
  // never loaded.
  this->model_start_ = this->model_data_ ? this->model_data_->get_model_pointer() : nullptr;
  if (this->model_start_ == nullptr) {
    ESP_LOGE(TAG, "Model '%s' has no valid data and will not be loaded", id.c_str());
  }
  this->default_probability_cutoff_ = default_probability_cutoff;
  this->probability_cutoff_.store(default_probability_cutoff, std::memory_order_relaxed);
  this->sliding_window_size_ = sliding_window_average_size;
  this->recent_streaming_probabilities_.resize(sliding_window_average_size, 0);
  this->wake_word_ = wake_word;
  this->trained_languages_ = std::move(trained_languages);
  this->tensor_arena_size_ = tensor_arena_size;
  this->register_streaming_ops_(this->streaming_op_resolver_);
  this->current_stride_step_ = 0;
  this->internal_only_ = false;  // Runtime models are always exposed to Home Assistant

  this->pref_ = global_preferences->make_preference<bool>(fnv1_hash(id));
  bool enabled;
  if (this->pref_.load(&enabled)) {
    // Use the enabled state loaded from flash
    this->enabled_ = enabled;
  } else {
    // No saved state: stay disabled. The activation flow calls enable() explicitly after adding.
    this->enabled_ = false;
  }
};

void WakeWordModel::enable() {
  this->reset_streaming_state();
  this->enabled_ = true;
  if (!this->internal_only_) {
    this->pref_.save(&this->enabled_);
  }
}

void WakeWordModel::disable() {
  this->reset_streaming_state();
  this->enabled_ = false;
  if (!this->internal_only_) {
    this->pref_.save(&this->enabled_);
  }
}

DetectionEvent WakeWordModel::determine_detected() {
  DetectionEvent detection_event;
  detection_event.wake_word = &this->wake_word_;
  detection_event.max_probability = 0;
  detection_event.average_probability = 0;

  if ((this->ignore_windows_ < 0) || !this->enabled_) {
    detection_event.detected = false;
    return detection_event;
  }

  uint32_t sum = 0;
  for (auto &prob : this->recent_streaming_probabilities_) {
    detection_event.max_probability = std::max(detection_event.max_probability, prob);
    sum += prob;
  }

  detection_event.average_probability = sum / this->sliding_window_size_;
  detection_event.detected = sum > this->get_probability_cutoff() * this->sliding_window_size_;

  this->unprocessed_probability_status_ = false;
  return detection_event;
}

VADModel::VADModel(const uint8_t *model_start, uint8_t default_probability_cutoff, size_t sliding_window_size,
                   size_t tensor_arena_size) {
  this->model_start_ = model_start;
  this->default_probability_cutoff_ = default_probability_cutoff;
  this->probability_cutoff_.store(default_probability_cutoff, std::memory_order_relaxed);
  this->sliding_window_size_ = sliding_window_size;
  this->recent_streaming_probabilities_.resize(sliding_window_size, 0);
  this->tensor_arena_size_ = tensor_arena_size;
  this->register_streaming_ops_(this->streaming_op_resolver_);
}

DetectionEvent VADModel::determine_detected() {
  DetectionEvent detection_event;
  detection_event.max_probability = 0;
  detection_event.average_probability = 0;

  if (!this->enabled_) {
    // We disabled the VAD model for some reason... so we shouldn't block wake words from being detected
    detection_event.detected = true;
    return detection_event;
  }

  uint32_t sum = 0;
  for (auto &prob : this->recent_streaming_probabilities_) {
    detection_event.max_probability = std::max(detection_event.max_probability, prob);
    sum += prob;
  }

  detection_event.average_probability = sum / this->sliding_window_size_;
  detection_event.detected = sum > (this->get_probability_cutoff() * this->sliding_window_size_);

  return detection_event;
}

bool StreamingModel::register_streaming_ops_(
    tflite::MicroMutableOpResolver<STREAMING_OP_COUNT> &op_resolver) {
  if (op_resolver.AddCallOnce() != kTfLiteOk)
    return false;
  if (op_resolver.AddVarHandle() != kTfLiteOk)
    return false;
  if (op_resolver.AddReshape() != kTfLiteOk)
    return false;
  if (op_resolver.AddReadVariable() != kTfLiteOk)
    return false;
  if (op_resolver.AddStridedSlice() != kTfLiteOk)
    return false;
  if (op_resolver.AddConcatenation() != kTfLiteOk)
    return false;
  if (op_resolver.AddAssignVariable() != kTfLiteOk)
    return false;
  if (op_resolver.AddConv2D() != kTfLiteOk)
    return false;
  if (op_resolver.AddMul() != kTfLiteOk)
    return false;
  if (op_resolver.AddAdd() != kTfLiteOk)
    return false;
  if (op_resolver.AddMean() != kTfLiteOk)
    return false;
  if (op_resolver.AddFullyConnected() != kTfLiteOk)
    return false;
  if (op_resolver.AddLogistic() != kTfLiteOk)
    return false;
  if (op_resolver.AddQuantize() != kTfLiteOk)
    return false;
  if (op_resolver.AddDepthwiseConv2D() != kTfLiteOk)
    return false;
  if (op_resolver.AddAveragePool2D() != kTfLiteOk)
    return false;
  if (op_resolver.AddMaxPool2D() != kTfLiteOk)
    return false;
  if (op_resolver.AddPad() != kTfLiteOk)
    return false;
  if (op_resolver.AddPack() != kTfLiteOk)
    return false;
  if (op_resolver.AddSplitV() != kTfLiteOk)
    return false;
  // External-state CRNN additions. These are the only operators in the exported graph that the stock resolver lacks.
  if (op_resolver.AddGather() != kTfLiteOk)
    return false;
  if (op_resolver.AddSplit() != kTfLiteOk)
    return false;
  if (op_resolver.AddSub() != kTfLiteOk)
    return false;
  if (op_resolver.AddTanh() != kTfLiteOk)
    return false;

  return true;
}

}  // namespace esphome::micro_wake_word

#endif
