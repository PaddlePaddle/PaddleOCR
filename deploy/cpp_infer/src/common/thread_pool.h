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
#pragma once

#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

namespace PaddlePool {

class ThreadPool {
public:
  using MutexGuard = std::lock_guard<std::mutex>;
  using UniqueLock = std::unique_lock<std::mutex>;
  using Thread = std::thread;
  using ThreadID = std::thread::id;
  using Task = std::function<void()>;

  ThreadPool();
  explicit ThreadPool(size_t maxThreads);

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  ~ThreadPool();

  template <typename Func, typename... Ts>
  auto submit(Func &&func, Ts &&...params)
      -> std::future<typename std::result_of<Func(Ts...)>::type>;

  size_t threadsNum() const;

private:
  static constexpr size_t WAIT_SECONDS = 2;
  void worker();
  void joinFinishedThreads();

  bool quit_;
  size_t currentThreads_;
  size_t idleThreads_;
  size_t maxThreads_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<Task> tasks_;
  std::queue<ThreadID> finishedThreadIDs_;
  std::unordered_map<ThreadID, Thread> threads_;
};

} // namespace PaddlePool

namespace PaddlePool {

template <typename Func, typename... Ts>
auto ThreadPool::submit(Func &&func, Ts &&...params)
    -> std::future<typename std::result_of<Func(Ts...)>::type> {
  auto execute =
      std::bind(std::forward<Func>(func), std::forward<Ts>(params)...);

  using ReturnType = typename std::result_of<Func(Ts...)>::type;
  using PackagedTask = std::packaged_task<ReturnType()>;

  auto task = std::make_shared<PackagedTask>(std::move(execute));
  auto result = task->get_future();

  MutexGuard guard(mutex_);
  assert(!quit_);

  tasks_.emplace([task]() { (*task)(); });
  if (idleThreads_ > 0) {
    cv_.notify_one();
  } else if (currentThreads_ < maxThreads_) {
    Thread t(&ThreadPool::worker, this);
    assert(threads_.find(t.get_id()) == threads_.end());
    threads_[t.get_id()] = std::move(t);
    ++currentThreads_;
  }

  return result;
}

} // namespace PaddlePool
