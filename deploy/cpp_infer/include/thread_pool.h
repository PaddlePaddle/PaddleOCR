#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

namespace std {
#define  MAX_THREAD_NUM 256
class ThreadPool{
private:
  using Task = std::function<void()>;
  std::vector<std::thread> pool;
//task queue
  std::queue<Task> tasks;

  std::mutex m_lock;

  std::condition_variable cv_task;

  std::atomic<bool> stoped;

  std::atomic<int>  idlThrNum;

//Constructor
public:
  inline ThreadPool(unsigned short size = 4):stoped{false}{
    //number of initialization threads
    idlThrNum = size < 1 ? 1 : size;
    for (size = 0; size < idlThrNum; ++size){
      pool.emplace_back([this]{
        while(!this->stoped){
          //worker thread function
          std::function<void()> task;{
            std::unique_lock<std::mutex> lock{ this->m_lock };
            this->cv_task.wait(lock,[this]{return this->stoped.load() || !this->tasks.empty();});
            if (this->stoped && this->tasks.empty())
              return;
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          idlThrNum--;
          task();
          idlThrNum++;
        }
      });
    }
  }
//destructor
  inline ~ThreadPool()
  {
    stoped.store(true);
    //wakes all threads to execute
    cv_task.notify_all();
    for (std::thread& thread : pool){
      if(thread.joinable())
        //wait for the finished task
        thread.join();
    }
  }
public:
  //submit a task;call Get() to get the return value, and wait for the task to execute
  template<class F, class... Args>
  auto commit(F&& f, Args&&... args) ->std::future<decltype(f(args...))>
  {
    if (stoped.load())
      throw std::runtime_error("commit on ThreadPool is stopped.");
    using RetType = decltype(f(args...));
    auto task = std::make_shared<std::packaged_task<RetType()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::future<RetType> future = task->get_future();{
    //add task to queue
      std::lock_guard<std::mutex> lock{ m_lock };
      tasks.emplace([task](){
        (*task)();
      });
     }
     cv_task.notify_one();
     return future;
  }

  int idlCount() { 
    return idlThrNum; 
  }

  int tasksSize(){
    return tasks.size();
  }
};
}

#endif
