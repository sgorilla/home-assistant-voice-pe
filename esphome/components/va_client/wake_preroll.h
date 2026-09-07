#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace esphome {
namespace va_client {

// Return the exact tail of rolling history that belongs to a wake handoff:
// the detector's causal look-back plus every sample captured after the wake
// boundary. uint32 subtraction is intentional; an age-bounded marker makes
// both the millisecond and sample counters safe across normal wraparound.
inline size_t wake_preroll_replay_samples(
    bool marker_armed, uint32_t now_ms, uint32_t marked_ms,
    uint32_t current_sample_sequence, uint32_t boundary_sample_sequence,
    size_t history_count, size_t history_capacity,
    size_t detector_lookback_samples, uint32_t maximum_marker_age_ms) {
  if (!marker_armed || history_capacity == 0 ||
      static_cast<uint32_t>(now_ms - marked_ms) > maximum_marker_age_ms) {
    return 0;
  }

  const size_t post_boundary_samples = static_cast<uint32_t>(
      current_sample_sequence - boundary_sample_sequence);
  size_t wanted = history_capacity;
  if (post_boundary_samples < history_capacity) {
    wanted = std::min(history_capacity,
                      post_boundary_samples + detector_lookback_samples);
  }
  return std::min(history_count, wanted);
}

}  // namespace va_client
}  // namespace esphome
