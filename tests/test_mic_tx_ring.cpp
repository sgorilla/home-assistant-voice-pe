#include "../esphome/components/va_client/mic_tx_ring.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

using esphome::va_client::MicTxRing;

static void expect_bytes(MicTxRing &ring, const uint8_t *expected, size_t length) {
  std::array<uint8_t, 32> actual{};
  assert(length <= actual.size());
  assert(ring.peek(actual.data(), actual.size()) == length);
  for (size_t i = 0; i < length; i++)
    assert(actual[i] == expected[i]);
}

static void test_wraparound_preserves_order() {
  std::array<uint8_t, 8> storage{};
  MicTxRing ring;
  ring.set_storage(storage.data(), storage.size());

  const uint8_t first[] = {0, 1, 2, 3, 4, 5};
  assert(ring.push_all(first, sizeof(first)));
  ring.consume(4);

  const uint8_t wrapped[] = {6, 7, 8, 9};
  assert(ring.push_all(wrapped, sizeof(wrapped)));
  const uint8_t expected[] = {4, 5, 6, 7, 8, 9};
  expect_bytes(ring, expected, sizeof(expected));
}

static void test_full_write_is_all_or_nothing() {
  std::array<uint8_t, 4> storage{};
  MicTxRing ring;
  ring.set_storage(storage.data(), storage.size());

  const uint8_t full[] = {10, 11, 12, 13};
  const uint8_t extra[] = {99};
  assert(ring.push_all(full, sizeof(full)));
  assert(!ring.push_all(extra, sizeof(extra)));
  expect_bytes(ring, full, sizeof(full));
}

static void test_short_send_consumes_only_confirmed_bytes() {
  std::array<uint8_t, 12> storage{};
  MicTxRing ring;
  ring.set_storage(storage.data(), storage.size());

  const uint8_t audio[] = {20, 21, 22, 23, 24, 25};
  assert(ring.push_all(audio, sizeof(audio)));
  ring.consume(2);  // Simulate a short WebSocket write.
  const uint8_t expected[] = {22, 23, 24, 25};
  expect_bytes(ring, expected, sizeof(expected));
}

static void test_preroll_then_live_exactly_once() {
  std::array<uint8_t, 16> storage{};
  MicTxRing ring;
  ring.set_storage(storage.data(), storage.size());

  // A wrapped pre-roll is appended as oldest-tail, wrapped-head, then live.
  const uint8_t preroll_tail[] = {30, 31, 32};
  const uint8_t preroll_head[] = {33, 34};
  const uint8_t live[] = {35, 36, 37};
  assert(ring.push_all(preroll_tail, sizeof(preroll_tail)));
  assert(ring.push_all(preroll_head, sizeof(preroll_head)));
  assert(ring.push_all(live, sizeof(live)));
  const uint8_t expected[] = {30, 31, 32, 33, 34, 35, 36, 37};
  expect_bytes(ring, expected, sizeof(expected));

  ring.consume(sizeof(expected));
  assert(ring.size() == 0);
}

static void test_clear_removes_stale_session_audio() {
  std::array<uint8_t, 8> storage{};
  MicTxRing ring;
  ring.set_storage(storage.data(), storage.size());

  const uint8_t stale[] = {40, 41, 42};
  const uint8_t fresh[] = {50, 51};
  assert(ring.push_all(stale, sizeof(stale)));
  ring.clear();
  assert(ring.size() == 0);
  assert(ring.push_all(fresh, sizeof(fresh)));
  expect_bytes(ring, fresh, sizeof(fresh));
}

static void test_zero_capacity_is_safe() {
  MicTxRing ring;
  ring.set_storage(nullptr, 0);

  const uint8_t byte[] = {60};
  std::array<uint8_t, 1> output{};
  assert(ring.size() == 0);
  assert(ring.free() == 0);
  assert(!ring.push_all(byte, sizeof(byte)));
  assert(ring.peek(output.data(), output.size()) == 0);
  ring.consume(1);
  assert(ring.size() == 0);
}

static void test_randomized_operations_match_reference_queue() {
  std::mt19937 random(0x50495050);  // deterministic: "PIPP"

  for (size_t capacity = 1; capacity <= 64; capacity++) {
    std::vector<uint8_t> storage(capacity);
    MicTxRing ring;
    ring.set_storage(storage.data(), storage.size());
    std::deque<uint8_t> reference;

    for (size_t step = 0; step < 20000; step++) {
      const uint32_t operation = random() % 6;
      if (operation <= 1) {
        const size_t length = random() % 17;
        std::vector<uint8_t> input(length);
        for (auto &byte : input)
          byte = static_cast<uint8_t>(random());

        const bool should_fit = length <= capacity - reference.size();
        assert(ring.push_all(input.data(), input.size()) == should_fit);
        if (should_fit) {
          for (uint8_t byte : input)
            reference.push_back(byte);
        }
      } else if (operation == 2) {
        const size_t requested = random() % 80;
        std::array<uint8_t, 80> actual{};
        const size_t copied = ring.peek(actual.data(), requested);
        assert(copied == std::min(requested, reference.size()));
        for (size_t i = 0; i < copied; i++)
          assert(actual[i] == reference[i]);
      } else if (operation == 3) {
        const size_t requested = random() % 80;
        const size_t consumed = std::min(requested, reference.size());
        ring.consume(requested);
        for (size_t i = 0; i < consumed; i++)
          reference.pop_front();
      } else if (operation == 4) {
        ring.clear();
        reference.clear();
      }

      assert(ring.size() == reference.size());
      assert(ring.free() == capacity - reference.size());
      std::array<uint8_t, 64> actual{};
      assert(ring.peek(actual.data(), actual.size()) == reference.size());
      for (size_t i = 0; i < reference.size(); i++)
        assert(actual[i] == reference[i]);
    }
  }
}

int main() {
  test_wraparound_preserves_order();
  test_full_write_is_all_or_nothing();
  test_short_send_consumes_only_confirmed_bytes();
  test_preroll_then_live_exactly_once();
  test_clear_removes_stale_session_audio();
  test_zero_capacity_is_safe();
  test_randomized_operations_match_reference_queue();
  return 0;
}
