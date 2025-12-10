#include "tasksys.h"
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <iostream>
using namespace std;


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    thread_num = num_threads;
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    mutex mtx;
    int task_cnt = 0;
    auto fthread = [this, runnable, num_total_tasks, &mtx, &task_cnt]() {
        int i=0;
        while(1)
        {
            mtx.lock();
            i = task_cnt++;
            mtx.unlock();
            if(i >= num_total_tasks)
                return;
            runnable->runTask(i, num_total_tasks);
        }
    };
    vector<thread> t;
    for(int i=1;i<thread_num;i++)
        t.push_back(thread(fthread));
    fthread();
    for(auto &th : t)
        th.join();
    t.clear();
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    thread_num = num_threads;
    task_total = task_left = 0;
    stop = false;
    for(int i=0;i<num_threads;i++)
        t.push_back(thread(&TaskSystemParallelThreadPoolSpinning::fthread, this));
        // t.push_back(thread(fthread));
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    stop = true;
    for(auto &th : t)
        th.join();
    t.clear();
}

void TaskSystemParallelThreadPoolSpinning::fthread() {
    while(!stop) {
        mtx.lock();
        if(q.empty()) { mtx.unlock(); continue; }
        int i = q.front();
        q.pop();
        mtx.unlock();
            cur_task->runTask(i, task_total);
            task_left--;
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    task_total = num_total_tasks;
    task_left = num_total_tasks;
    cur_task = runnable;
    mtx.lock();
    for(int i=0;i<num_total_tasks;i++)
        q.push(i);
    mtx.unlock();
    while(task_left>0);
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    thread_num = num_threads;
    stop = false;
    for(int i=0;i<thread_num;i++)
        t.push_back(thread(&TaskSystemParallelThreadPoolSleeping::fthread, this));
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    stop = true;
    cv.notify_all();
    for(int i=0;i<thread_num;i++)
        t[i].join();
}

void TaskSystemParallelThreadPoolSleeping::fthread() {
    while (!stop) {
        int u=0;
        while(1) {
            unique_lock<mutex> lck(mtx);
            cv.wait(lck, [this] { return true; });
            if(stop) return;
            if(q.empty()) continue;
            u = q.front();
            q.pop();
            break;
        }
        cur_task->runTask(u, task_total);
        unique_lock<mutex> lck(mtx);
        task_left--;
        if(task_left>0) cv.notify_all();
        else cv.notify_one();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    cur_task = runnable;
    task_left = num_total_tasks;
    task_total = num_total_tasks;
    mtx.lock();
    for (int i = 0; i < num_total_tasks; i++)
        q.push(i);
    mtx.unlock();
    cv.notify_all();
    while(task_left>0) {
        unique_lock<mutex> lck(mtx);
        cv.wait(lck, [this] { return true; });
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
