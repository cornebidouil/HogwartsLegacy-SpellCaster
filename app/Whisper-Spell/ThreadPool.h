#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>

class ThreadPool {
public:
    enum class Priority {
        HIGH,    // Spell transcription - processed first
        NORMAL,  // Regular tasks
        LOW      // Background tasks (crowdsourcing, etc.)
    };

private:
    struct PrioritizedTask {
        std::function<void()> task;
        Priority priority;
        uint64_t enqueueTime;  // For FIFO within same priority

        // Higher priority tasks should be processed first
        bool operator<(const PrioritizedTask& other) const {
            if (priority != other.priority) {
                return priority > other.priority;  // Reversed for min-heap (HIGH < NORMAL < LOW)
            }
            return enqueueTime > other.enqueueTime;  // Earlier tasks first
        }
    };

    std::vector<std::thread> workers;
    std::priority_queue<PrioritizedTask> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        if(stop && tasks.empty()) return;
                        task = std::move(const_cast<std::function<void()>&>(tasks.top().task));
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f, Priority priority = Priority::NORMAL) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(PrioritizedTask{
                std::forward<F>(f),
                priority,
                static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())
            });
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) {
            worker.join();
        }
    }
};
