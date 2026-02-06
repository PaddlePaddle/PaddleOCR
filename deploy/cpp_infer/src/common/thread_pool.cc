
// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "thread_pool.h"

namespace PaddlePool {

constexpr size_t ThreadPool::WAIT_SECONDS;

ThreadPool::ThreadPool() : ThreadPool(Thread::hardware_concurrency()) {}

ThreadPool::ThreadPool(size_t maxThreads)
    : quit_(false), currentThreads_(0), idleThreads_(0),
      maxThreads_(maxThreads) {}

ThreadPool::~ThreadPool() {
  {
    MutexGuard guard(mutex_);
    quit_ = true;
  }
  cv_.notify_all();

  for (auto &elem : threads_) {
    assert(elem.second.joinable());
    elem.second.join();
  }
}

size_t ThreadPool::threadsNum() const {
  MutexGuard guard(mutex_);
  return currentThreads_;
}

void ThreadPool::worker() {
  while (true) {
    Task task;
    {
      UniqueLock uniqueLock(mutex_);
      ++idleThreads_;
      auto hasTimedout =
          !cv_.wait_for(uniqueLock, std::chrono::seconds(WAIT_SECONDS),
                        [this]() { return quit_ || !tasks_.empty(); });
      --idleThreads_;
      if (tasks_.empty()) {
        if (quit_) {
          --currentThreads_;
          return;
        }
        if (hasTimedout) {
          --currentThreads_;
          joinFinishedThreads();
          finishedThreadIDs_.emplace(std::this_thread::get_id());
          return;
        }
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

void ThreadPool::joinFinishedThreads() {
  while (!finishedThreadIDs_.empty()) {
    auto id = std::move(finishedThreadIDs_.front());
    finishedThreadIDs_.pop();
    auto iter = threads_.find(id);

    assert(iter != threads_.end());
    assert(iter->second.joinable());

    iter->second.join();
    threads_.erase(iter);
  }
}

} // namespace PaddlePool
