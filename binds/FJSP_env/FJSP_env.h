#pragma once

#include <string>
#include <sstream>
#include <set>
#include <optional>
#include <variant>
#include <map>
#include <functional>
#include <pybind11/pybind11.h>

#include "aux_structs.h"

using namespace std;
namespace py = pybind11;

struct Product
{
    OperationId from, to;
    auto operator<=>(const Product &other) const = default;
    string repr();
};

struct Operation
{
    Operation(OperationId, MachineType, float);
    string repr();

    OperationId id;
    OperationStatus status;
    MachineType machine_type;
    float process_time;

    optional<MachineId> processing_machine;
    float finish_timestamp;

    set<OperationId> predecessors, arrived_preds;
    set<OperationId> successors, sent_succs;
};

struct Machine
{
    Machine(MachineId, MachineType);
    string repr();

    MachineId id;
    MachineType type;
    MachineStatus status;
    optional<OperationId> working_operation, waiting_operation;
    set<Product> materials, products;
};

struct AGV
{
    AGV(AGVId, float, MachineId);
    string repr();

    AGVId id;
    AGVStatus status;
    float speed;
    MachineId position, target_machine;
    optional<Product> loaded_item, target_item;
    float finish_timestamp;
};

struct Action
{
    Action(ActionType, AGVId, MachineId);
    Action(ActionType, AGVId, MachineId, Product);
    string repr();

    ActionType type;
    AGVId act_AGV;
    MachineId target_machine;
    optional<Product> target_product;
};

struct GraphFeature;

class Graph
{
public:
    using ProcessingOperationQueue = copyable_priority_queue<OperationId, function<bool(OperationId, OperationId)>>;
    using MovingAGVQueue = copyable_priority_queue<AGVId, function<bool(AGVId, AGVId)>>;

    static const OperationId begin_operation_id = 0, end_operation_id = 9999;
    static const MachineId dummy_machine_id = 0;
    static const MachineType dummy_machine_type = 0;

    static const size_t operation_feature_size = 8;
    static const size_t machine_feature_size = 7;
    static const size_t AGV_feature_size = 5;

    Graph();

    Graph(const Graph &);

    py::dict get_state();
    static shared_ptr<Graph> from_state(py::dict);

    OperationId add_operation(MachineType, float);
    void add_relation(OperationId, OperationId);
    void remove_relation(OperationId, OperationId);
    OperationId insert_operation(MachineType, float, optional<OperationId>, optional<OperationId>);
    void remove_operation(OperationId);
    shared_ptr<Operation> get_operation(OperationId);
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

    static shared_ptr<Graph> rand_generate(size_t, size_t, size_t, size_t, float, float, float, float, float);

    void init();

    shared_ptr<Graph> copy();

    RepeatedTuple<float, operation_feature_size> get_operation_feature(shared_ptr<Operation>);
    RepeatedTuple<float, machine_feature_size> get_machine_feature(shared_ptr<Machine>);
    RepeatedTuple<float, AGV_feature_size> get_AGV_feature(shared_ptr<AGV>);
    shared_ptr<GraphFeature> features();

    bool finished();
    double finish_time_lower_bound();

    vector<Action> get_available_actions();

    bool operation_time_compare(OperationId, OperationId);
    bool AGV_time_compare(AGVId, AGVId);

    void act_move(AGVId, MachineId);
    void act_pick(AGVId, MachineId, Product);
    void act_transport(AGVId, MachineId);
    shared_ptr<Graph> act(Action);

    void wait_operation();
    void wait_AGV();
    void _wait(float);
    shared_ptr<Graph> wait(float);

protected:
    OperationId next_operation_id;
    MachineId next_machine_id;
    AGVId next_AGV_id;

    bool inited;
    float timestamp;

    map<OperationId, shared_ptr<Operation>> operations;
    map<MachineId, shared_ptr<Machine>> machines;
    map<AGVId, shared_ptr<AGV>> AGVs;

    map<MachineId, map<MachineId, float>> distances;

    ProcessingOperationQueue processing_operations;
    MovingAGVQueue moving_AGVs;

    optional<vector<Action>> available_actions;
};

struct GraphFeature
{
    vector<RepeatedTuple<float, Graph::operation_feature_size>> operation_features;
    vector<tuple<size_t, size_t>> predecessor_idx; // predecessor_idx, successor_idx;
    vector<tuple<size_t, size_t>> successor_idx;   // successor_idx, predecessor_idx
    vector<RepeatedTuple<float, Graph::machine_feature_size>> machine_features;
    vector<tuple<size_t, size_t>> processable_idx; // machine_idx, operation_idx
    vector<tuple<size_t, size_t, float>> processing;  // machine_idx, operation_idx, rest_time
    vector<tuple<size_t, size_t, size_t, size_t>> waiting;     // machine_idx, operation_idx, total, current
    vector<RepeatedTuple<float, Graph::AGV_feature_size>> AGV_features;
    vector<tuple<size_t, size_t>> AGV_position;       // AGV_idx, machine_idx
    vector<tuple<size_t, size_t, float>> AGV_target;  // AGV_idx, machine_idx, rest_time
    vector<tuple<size_t, size_t, size_t>> AGV_loaded; // AGV_idx, from_operation_idx, to_operation_idx
};