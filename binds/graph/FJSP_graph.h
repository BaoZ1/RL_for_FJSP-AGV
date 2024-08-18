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

struct OperationBase
{
    OperationBase(OperationId, MachineType);
    virtual string repr() = 0;

    OperationId id;
    OperationStatus status;
    MachineType machine_type;
};

struct GraphBegin : public OperationBase
{
    GraphBegin(OperationId, MachineType);
    string repr() override;

    unordered_set<OperationId> successors;
    size_t sent_count, arrived_count;
};

struct GraphEnd : public OperationBase
{
    GraphEnd(OperationId, MachineType);
    string repr() override;

    unordered_set<OperationId> predecessors;
    size_t recive_cound;
};

struct Operation : public OperationBase
{
    Operation(OperationId, MachineType, float);
    string repr() override;

    float process_time;
    float finish_timestamp;
    OperationId predecessor;
    OperationId successor;
    optional<MachineId> processing_machine;
};

struct Machine
{
    Machine(MachineId, MachineType);
    string repr();

    MachineId id;
    MachineType type;
    MachineStatus status;
    optional<OperationId> working_operation;
};

struct AGV
{
    AGV(AGVId, float, MachineId);
    string repr();

    AGVId id;
    AGVStatus status;
    float speed;
    MachineId position, target;
    optional<OperationId> transport_target;
    float finish_timestamp;

    optional<OperationId> loaded_item;
};

struct Action
{
    Action(ActionType, AGVId, MachineId, optional<OperationId>);
    string repr();

    ActionType type;
    AGVId act_AGV;
    MachineId target_machine;
    optional<OperationId> target_operation;
};

struct GraphFeatures;

class Graph
{
public:
    using OperationNodePtr = variant<shared_ptr<GraphBegin>, shared_ptr<GraphEnd>, shared_ptr<Operation>>;

    using ProcessingOperationQueue = copyable_priority_queue<OperationId, function<bool(OperationId, OperationId)>>;
    using MovingAGVQueue = copyable_priority_queue<AGVId, function<bool(AGVId, AGVId)>>;

    static const OperationId begin_operation_id = 0, end_operation_id = 9999;
    static const MachineId dummy_machine_id = 0;
    static const MachineType dummy_machine_type = 0;

    static const size_t operation_feature_size = 10;
    static const size_t machine_feature_size = 5;
    static const size_t AGV_feature_size = 5;

    Graph();

    Graph(const Graph &);

    OperationId add_operation(MachineType, float, optional<OperationId>, optional<OperationId>);
    void remove_operation(OperationId);
    OperationNodePtr get_operation(OperationId);
    bool contains(OperationId);

    MachineId add_machine(MachineType);
    shared_ptr<Machine> get_machine(MachineId);

    AGVId add_AGV(float, MachineId);
    shared_ptr<AGV> get_AGV(AGVId);

    float get_timestamp();
    void set_timestamp(float);

    string repr();

    void set_distance(MachineId, MachineId, float);
    void set_distance(map<MachineId, map<MachineId, float>> &);
    void set_rand_distance(float, float);
    float get_travel_time(MachineId, MachineId, AGVId);

    static shared_ptr<Graph> rand_generate(vector<size_t>, size_t, size_t, size_t, float, float, float, float, float);

    void init_operation_status();

    shared_ptr<Graph> copy();

    RepeatedTuple<float, operation_feature_size> get_operation_feature(shared_ptr<Operation>);
    RepeatedTuple<float, machine_feature_size> get_machine_feature(shared_ptr<Machine>);
    RepeatedTuple<float, AGV_feature_size> get_AGV_feature(shared_ptr<AGV>);
    shared_ptr<GraphFeatures> features();

    bool finished();

    vector<Action> get_available_actions();

    bool operation_time_compare(OperationId, OperationId);
    bool AGV_time_compare(AGVId, AGVId);

    void act_move(AGVId, MachineId);
    void act_pick(AGVId, MachineId);
    void act_transport(AGVId, OperationId, MachineId);
    shared_ptr<Graph> act(Action);

    void wait_operation();
    void wait_AGV();
    void _wait(float);
    shared_ptr<Graph> wait(float);

protected:
    OperationId next_operation_id;
    MachineId next_machine_id;
    AGVId next_AGV_id;

    float timestamp;

    map<OperationId, OperationNodePtr> operations;
    map<MachineId, shared_ptr<Machine>> machines;
    map<AGVId, shared_ptr<AGV>> AGVs;

    map<MachineId, map<MachineId, float>> distances;

    ProcessingOperationQueue processing_operations;
    MovingAGVQueue moving_AGVs;

    optional<vector<Action>> available_actions;
};

struct GraphFeatures
{
    vector<RepeatedTuple<float, Graph::operation_feature_size>> operation_features;
    vector<RepeatedTuple<int, 2>> operation_relations;
    vector<size_t> job_index;
    vector<MachineType> machine_type;
    vector<RepeatedTuple<float, Graph::machine_feature_size>> machine_features;
    vector<vector<float>> processable_machine_mask;
    vector<RepeatedTuple<float, Graph::AGV_feature_size>> AGV_features;
};