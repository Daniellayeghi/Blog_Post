#ifndef TEST_CPP_TIME_BENCH_H
#define TEST_CPP_TIME_BENCH_H

#include <chrono>
#include <iostream>

struct TimeBench
{
    std::chrono::time_point<std::chrono::steady_clock> strat, end;
    std::chrono::duration<double> duration;

    TimeBench()
    {
        strat = std::chrono::steady_clock::now();
    }
public:
    ~TimeBench()
    {
        end = std::chrono::steady_clock::now();
        duration = end - strat;
        std::cout << "Computation took: " << duration.count() * 1000.0 << " ms" << "\n";
    }
};


#endif //TEST_CPP_TIME_BENCH_H
