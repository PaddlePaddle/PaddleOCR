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

namespace std
{
#define  MAX_THREAD_NUM 256
//线程池,可以提交变参函数或拉姆达表达式的匿名函数执行,可以获取执行返回值
//不支持类成员函数, 支持类静态成员函数或全局函数,Opteron()函数等
class threadpool
{
private:
	using Task = std::function<void()>;//是类型别名，简化了 typedef 的用法。function<void()> 可以认为是一个函数类型，接受任意原型是 void() 的函数，或是函数对象，或是匿名函数。void() 意思是不带参数，没有返回值。

	// 线程池
	std::vector<std::thread> pool;
	// 任务队列
	std::queue<Task> tasks;
	// 同步
	std::mutex m_lock;
	// 条件阻塞
	std::condition_variable cv_task;
	// 是否关闭提交
	std::atomic<bool> stoped; //本身就是原子类型，load()和store()是原子操作，不需要加mutex
	//空闲线程数量
	std::atomic<int>  idlThrNum;

public:
	//构造函数
	inline threadpool(unsigned short size = 4):stoped{false}
	{
		idlThrNum = size < 1 ? 1 : size;
		for (size = 0; size < idlThrNum; ++size)
		{   //初始化线程数量
/*pool.emplace_back([this]{...}) 是构造了一个线程对象，执行函数是拉姆达匿名函数
 匿名函数： [this]{...} 不多说。[] 是捕捉器，this 是引用域外的变量 this指针， 内部使用死循环, 由cv_task.wait(lock,[this]{...}) 来阻塞线程*/		
			pool.emplace_back([this]
			{ // 工作线程函数
				while(!this->stoped)
				{
					std::function<void()> task;
					{   // 获取一个待执行的 task
						std::unique_lock<std::mutex> lock{ this->m_lock };
			// unique_lock 相比 lock_guard 的好处是：可以随时 unlock() 和 lock()
						this->cv_task.wait(lock,[this]{return this->stoped.load() || !this->tasks.empty();}); // wait 直到有 task
						if (this->stoped && this->tasks.empty())
							return;
						task = std::move(this->tasks.front()); // 取一个 task
						this->tasks.pop();
					}
					idlThrNum--;
					task();
					idlThrNum++;
				}//while
			});//pool.emplace_back
		}//for
	}//threadpool构造函数
	
	//析构函数
	inline ~threadpool()
	{
		stoped.store(true);
		cv_task.notify_all(); // 唤醒所有线程执行
		for (std::thread& thread : pool) 
		{
			//thread.detach(); // 让线程“自生自灭”
			if(thread.joinable())
				thread.join(); // 等待任务结束， 前提：线程一定会执行完
		}
	}

public:
	// 提交一个任务
	// 调用.get()获取返回值会等待任务执行完,获取返回值
	// 有两种方法可以实现调用类成员，
	// 一种是使用   bind： .commit(std::bind(&Dog::sayHello, &dog));
	// 一种是用 mem_fn： .commit(std::mem_fn(&Dog::sayHello), &dog)
	//commit,  不是固定参数的, 无参数数量限制!  这得益于可变参数模板.
	template<class F, class... Args>
	auto commit(F&& f, Args&&... args) ->std::future<decltype(f(args...))>
	{
		if (stoped.load())    // stop == true ??
			throw std::runtime_error("commit on ThreadPool is stopped.");
//delctype(expr) 用来推断 expr 的类型，和 auto 是类似的，相当于类型占位符，占据一个类型的位置
		using RetType = decltype(f(args...)); // typename std::result_of<F(Args...)>::type, 函数 f 的返回值类型
//packaged_task 就是任务函数的封装类，通过 get_future 获取 future ， 然后通过 future 可以获取函数的返回值(future.get())；packaged_task 本身可以像函数一样调用 () ；
		auto task = std::make_shared<std::packaged_task<RetType()> >(
//bind 函数，接受函数 f 和部分参数，返回currying后的匿名函数，譬如 bind(add, 4) 可以实现类似 add4 的函数！
//forward() 函数，类似于 move() 函数，后者是将参数右值化,前者是，不改变最初传入的类型的引用类型
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)       
		);    // wtf !
		std::future<RetType> future = task->get_future();
		{    // 添加任务到队列
//lock_guard 是 mutex 的 stack 封装类，构造的时候 lock()，析构的时候 unlock()
			std::lock_guard<std::mutex> lock{ m_lock };//对当前块的语句加锁  lock_guard 是 mutex 的 stack 封装类，构造的时候 lock()，析构的时候 unlock()
			tasks.emplace([task](){ // push(Task{...})
				(*task)();
			});
		}
		cv_task.notify_one(); // 唤醒一个线程执行
		return future;
	}
	
	//空闲线程数量
	int idlCount() 
	{ 
		return idlThrNum; 
	}
	int tasksSize()
	{
		return tasks.size();
	}
};

}

#endif
