#pragma once

#include <string>
#include <sstream>
#include <unordered_set>
#include <optional>
#include <variant>
#include <map>
#include <functional>

#include "aux_structs.h"

using namespace std;

struct TaskBase
{
    TaskBase(TaskId, MachineType);
    virtual string repr() = 0;

    TaskId id;
    TaskStatus status;
    MachineType machine_type;
};

struct JobBegin : public TaskBase
{
    JobBegin(TaskId, MachineType);
    string repr() override;

    unordered_set<TaskId> successors;
    size_t sent_count, arrived_count;
};

struct JobEnd : public TaskBase
{
    JobEnd(TaskId, MachineType);
    string repr() override;

    unordered_set<TaskId> predecessors;
    size_t recive_cound;
};

struct Task : public TaskBase
{
    Task(TaskId, MachineType, double);
    string repr() override;

    double process_time;
    double finish_timestamp;
    TaskId predecessor;
    TaskId successor;
    optional<MachineId> processing_machine;
};

struct Machine
{
    Machine(MachineId, MachineType);
    string repr();

    MachineId id;
    MachineType type;
    MachineStatus status;
    optional<TaskId> working_task;
};

struct AGV
{
    AGV(AGVId, double, MachineId);
    string repr();

    AGVId id;
    AGVStatus status;
    double speed;
    MachineId position, target;
    optional<TaskId> transport_target;
    double finish_timestamp;

    optional<TaskId> loaded_item;
};

struct Action
{
    Action(ActionType, AGVId, MachineId, optional<TaskId>);
    string repr();

    ActionType type;
    AGVId act_AGV;
    MachineId target_machine;
    optional<TaskId> target_task;
};

struct JobFeatures;

class Job
{
public:
    using TaskNodePtr = variant<shared_ptr<JobBegin>, shared_ptr<JobEnd>, shared_ptr<Task>>;

    using ProcessingTaskQueue = copyable_priority_queue<TaskId, function<bool(TaskId, TaskId)>>;
    using MovingAGVQueue = copyable_priority_queue<AGVId, function<bool(AGVId, AGVId)>>;

    static const TaskId begin_task_id = 0, end_task_id = 9999;
    static const MachineId dummy_machine_id = 0;
    static const MachineType dummy_machine_type = 0;

    static const size_t task_feature_size = 10;
    static const size_t machine_feature_size = 5;
    static const size_t AGV_feature_size = 5;

    Job();

    Job(const Job &);

    TaskId add_task(MachineType, double, optional<TaskId>, optional<TaskId>);
    void remove_task(TaskId);
    TaskNodePtr get_task(TaskId);
    bool contains(TaskId);

    MachineId add_machine(MachineType);
    shared_ptr<Machine> get_machine(MachineId);

    AGVId add_AGV(double, MachineId);
    shared_ptr<AGV> get_AGV(AGVId);

    double get_timestamp();
    void set_timestamp(double);

    string repr();

    void set_distance(MachineId, MachineId, double);
    void set_distance(map<MachineId, map<MachineId, double>> &);
    void set_rand_distance(double, double);
    double get_travel_time(MachineId, MachineId, AGVId);

    static shared_ptr<Job> rand_generate(vector<size_t>, size_t, size_t, size_t, double, double, double, double, double);

    void init_task_status();

    shared_ptr<Job> copy();

    RepeatedTuple<double, task_feature_size> get_task_feature(shared_ptr<Task>);
    RepeatedTuple<double, machine_feature_size> get_machine_feature(shared_ptr<Machine>);
    RepeatedTuple<double, AGV_feature_size> get_AGV_feature(shared_ptr<AGV>);
    shared_ptr<JobFeatures> features();

    bool finished();

    vector<Action> get_available_actions();

    bool task_time_compare(TaskId, TaskId);
    bool AGV_time_compare(AGVId, AGVId);

    void act_move(AGVId, MachineId);
    void act_pick(AGVId, MachineId);
    void act_transport(AGVId, TaskId, MachineId);
    shared_ptr<Job> act(Action);

    void wait_task();
    void wait_AGV();
    void _wait(double);
    shared_ptr<Job> wait(double);

protected:
    TaskId next_task_id;
    MachineId next_machine_id;
    AGVId next_AGV_id;

    double timestamp;

    map<TaskId, TaskNodePtr> tasks;
    map<MachineId, shared_ptr<Machine>> machines;
    map<AGVId, shared_ptr<AGV>> AGVs;

    map<MachineId, map<MachineId, double>> distances;

    ProcessingTaskQueue processing_tasks;
    MovingAGVQueue moving_AGVs;

    optional<vector<Action>> available_actions;
};

struct JobFeatures
{
    vector<RepeatedTuple<double, Job::task_feature_size>> task_features;
    vector<RepeatedTuple<optional<size_t>, 2>> task_relations;
    vector<RepeatedTuple<double, Job::machine_feature_size>> machine_features;
    vector<vector<double>> processable_machine_mask;
    vector<RepeatedTuple<double, Job::AGV_feature_size>> AGV_features;
};