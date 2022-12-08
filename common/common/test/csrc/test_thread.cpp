#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

std::mutex mutex;
std::condition_variable condvar;

std::queue<int> msgQueue;

void producer(int start, int end)
{
    for (int x = start; x < end; x++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        {        
            std::lock_guard<std::mutex> guard(mutex);
            msgQueue.push(x);
        }
        printf("Produce message %d\n", x);
        condvar.notify_all();
    }
}

void consumer(int demand)
{
    while (true) {
        std::unique_lock<std::mutex> ulock(mutex);
        condvar.wait(ulock, []{ return msgQueue.size() > 0;});
        printf("Consume message %d\n", msgQueue.front());
        msgQueue.pop();
        --demand;
        if (!demand) break;
    }
}


int main()
{
    std::thread producer1(producer, 0, 10);
    std::thread producer2(producer, 10, 20);
    std::thread producer3(producer, 20, 30);
    std::thread consumer1(consumer, 20);
    std::thread consumer2(consumer, 10);

    producer1.join();
    producer2.join();
    producer3.join();
    consumer1.join();
    consumer2.join();
} 