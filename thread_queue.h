
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

template <typename Type, typename Queue = std::queue<Type>>
class ThreadQueue : Queue, std::mutex, std::condition_variable {
    typename Queue::size_type capacity;
    bool done = false;
    std::vector<std::thread> threads;

   public:
    template <typename Function>
    ThreadQueue(Function function, int concurrency, int num_items)
        : capacity{concurrency * num_items} {
        for (int count{0}; count < concurrency; count += 1)
            threads.emplace_back(static_cast<void (ThreadQueue::*)(Function)>(
                                     &ThreadQueue::consume), this, function);
    }

    ThreadQueue(ThreadQueue &&) = default;
    ThreadQueue &operator=(ThreadQueue &&) = delete;

    ~ThreadQueue() {
        {
            std::lock_guard<std::mutex> guard(*this);
            done = true;
            notify_all();
        }
        for (auto &&thread : threads) {
            thread.join();
        }
    }

    void operator()(Type &&value) {
        std::unique_lock<std::mutex> lock(*this);
        while (Queue::size() == capacity) {
            wait(lock);
        }
        Queue::push(std::forward<Type>(value));
        notify_one();
    }

   private:
    template <typename Function>
    void consume(Function process) {
        std::unique_lock<std::mutex> lock(*this);
        while (true) {
            if (not Queue::empty()) {
                Type item{std::move(Queue::front())};
                Queue::pop();
                notify_one();
                lock.unlock();
                process(item);
                lock.lock();
            } else if (done) {
                break;
            } else {
                wait(lock);
            }
        }
    }
};