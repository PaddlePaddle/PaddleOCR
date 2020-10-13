//
//  timer.h
//  face_demo
//
//  Created by Li,Xiaoyang(SYS) on 2019/8/20.
//  Copyright © 2019年 Li,Xiaoyang(SYS). All rights reserved.
//

#ifndef timer_h
#define timer_h
#include <chrono>
#include <list>
class Timer final {
    
public:
    Timer() {}
    
    ~Timer() {}
    
    void clear() {
        ms_time.clear();
    }
    
    void start() {
        tstart = std::chrono::system_clock::now();
    }
    
    void end() {
        tend = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart);
        float elapse_ms = 1000.f * float(ts.count()) * std::chrono::microseconds::period::num / \
        std::chrono::microseconds::period::den;
        ms_time.push_back(elapse_ms);
    }
    
    float get_average_ms() {
        if (ms_time.size() == 0) {
            return 0.f;
        }
        float sum = 0.f;
        for (auto i : ms_time){
            sum += i;
        }
        return sum / ms_time.size();
    }
    
    float get_sum_ms(){
        if (ms_time.size() == 0) {
            return 0.f;
        }
        float sum = 0.f;
        for (auto i : ms_time){
            sum += i;
        }
        return sum;
    }
    
    // return tile (0-99) time.
    float get_tile_time(float tile) {
        
        if (tile <0 || tile > 100) {
            return -1.f;
        }
        int total_items = (int)ms_time.size();
        if (total_items <= 0) {
            return -2.f;
        }
        ms_time.sort();
        int pos = (int)(tile * total_items / 100);
        auto it = ms_time.begin();
        for (int i = 0; i < pos; ++i) {
            ++it;
        }
        return *it;
    }
    
    const std::list<float> get_time_stat() {
        return ms_time;
    }
    
private:
    std::chrono::time_point<std::chrono::system_clock> tstart;
    std::chrono::time_point<std::chrono::system_clock> tend;
    std::list<float> ms_time;
};

#endif /* timer_h */
