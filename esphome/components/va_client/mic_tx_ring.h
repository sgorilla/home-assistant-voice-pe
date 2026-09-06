#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace esphome {
namespace va_client {

// Allocation-free byte ring used between the high-priority I2S microphone
// producer and the lower-priority WebSocket sender. Synchronization is owned by
// VaClient so this small class can be exercised by host-side tests.
class MicTxRing {
 public:
  void set_storage(uint8_t *storage, size_t capacity) {
    this->storage_ = storage;
    this->capacity_ = capacity;
    this->clear();
  }

  void clear() {
    this->head_ = 0;
    this->tail_ = 0;
    this->size_ = 0;
  }

  size_t size() const { return this->size_; }
  size_t capacity() const { return this->capacity_; }
  size_t free() const { return this->capacity_ - this->size_; }

  // Enqueue the whole span or nothing. Dropping a complete PCM frame is easier
  // to diagnose than silently accepting a partial sample/frame.
  bool push_all(const uint8_t *data, size_t length) {
    if (length == 0)
      return true;
    if (this->storage_ == nullptr || data == nullptr || length > this->free())
      return false;

    const size_t first = std::min(length, this->capacity_ - this->tail_);
    std::memcpy(this->storage_ + this->tail_, data, first);
    if (first < length)
      std::memcpy(this->storage_, data + first, length - first);
    this->tail_ = (this->tail_ + length) % this->capacity_;
    this->size_ += length;
    return true;
  }

  // Copy without consuming. The caller removes only the byte count confirmed
  // by esp_websocket_client_send_bin(), so short writes preserve ordering.
  size_t peek(uint8_t *destination, size_t length) const {
    if (destination == nullptr || this->storage_ == nullptr || length == 0)
      return 0;
    length = std::min(length, this->size_);
    const size_t first = std::min(length, this->capacity_ - this->head_);
    std::memcpy(destination, this->storage_ + this->head_, first);
    if (first < length)
      std::memcpy(destination + first, this->storage_, length - first);
    return length;
  }

  void consume(size_t length) {
    length = std::min(length, this->size_);
    if (length == 0)
      return;
    this->head_ = (this->head_ + length) % this->capacity_;
    this->size_ -= length;
    if (this->size_ == 0) {
      // Canonical empty state simplifies diagnostics and wraparound tests.
      this->head_ = 0;
      this->tail_ = 0;
    }
  }

 private:
  uint8_t *storage_{nullptr};
  size_t capacity_{0};
  size_t head_{0};
  size_t tail_{0};
  size_t size_{0};
};

}  // namespace va_client
}  // namespace esphome
