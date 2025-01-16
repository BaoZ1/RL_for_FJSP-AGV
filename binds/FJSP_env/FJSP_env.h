#ifndef FJSP_ENV_H
#define FJSP_ENV_H

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
    string repr() const;

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
    Machine(MachineId, MachineType, Position);
    string repr() const;

    MachineId id;
    MachineType type;
    Position pos;
    MachineStatus status;
    optional<OperationId> working_operation;
    vector<OperationId> waiting_operations;
    set<Product> materials, products;
};

struct AGV
{
    AGV(AGVId, float, MachineId);
    string repr() const;

    AGVId id;
    AGVStatus status;
    float speed;
    MachineId position, target_machine;
    optional<Product> loaded_item, target_item;
    float finish_timestamp;
};

struct Action
{
    Action(ActionType);
    Action(ActionType, AGVId, MachineId);
    Action(ActionType, AGVId, MachineId, Product);
    string repr() const;

    ActionType type;
    optional<AGVId> act_AGV;
    optional<MachineId> target_machine;
    optional<Product> target_product;
};

struct GenerateParam
{
    size_t operation_count;
    size_t machine_count;
    size_t AGV_count;
    size_t machine_type_count;
    float min_transport_time;
    float max_transport_time;
    float min_max_speed_ratio;
    float min_process_time;
    float max_process_time;
    bool simple_mode;
};

struct GraphFeature;
struct IdIdxMapper;

class Graph
{
public:
    using ProcessingOperationQueue = copyable_priority_queue<OperationId, function<bool(OperationId, OperationId)>>;
    using MovingAGVQueue = copyable_priority_queue<AGVId, function<bool(AGVId, AGVId)>>;

    static const OperationId begin_operation_id = 0, end_operation_id = 9999;
    static const MachineId dummy_machine_id = 0;
    static const MachineType dummy_machine_type = 0;

    static const size_t global_feature_size = 3;
    static const size_t operation_feature_size = 9;
    static const size_t machine_feature_size = 7;
    static const size_t AGV_feature_size = 5;

    Graph();

    Graph(const Graph &);

    py::dict get_state() const;
    static shared_ptr<Graph> from_state(py::dict);

    vector<OperationId> get_operations_id() const;
    vector<MachineId> get_machines_id() const;
    vector<AGVId> get_AGVs_id() const;

    OperationId add_operation(MachineType, float);
    void add_relation(OperationId, OperationId);
    void remove_relation(OperationId, OperationId);
    OperationId insert_operation(MachineType, float, optional<OperationId>, optional<OperationId>);
    void remove_operation(OperationId);
    shared_ptr<Operation> get_operation(OperationId) const;
    bool contains(OperationId) const;

    MachineId add_machine(MachineType, Position);
    void remove_machine(MachineId);
    shared_ptr<Machine> get_machine(MachineId) const;

    AGVId add_AGV(float, MachineId);
    void remove_AGV(AGVId);
    shared_ptr<AGV> get_AGV(AGVId) const;

    float get_timestamp() const;
    void set_timestamp(float);

    string repr() const;

    void add_path(MachineId, MachineId);
    void remove_path(MachineId, MachineId);
    void calc_distance();
    map<MachineId, map<MachineId, tuple<vector<MachineId>, float>>> get_paths();
    float get_travel_time(MachineId, MachineId, AGVId) const;

    static shared_ptr<Graph> rand_generate(GenerateParam);

    shared_ptr<Graph> copy() const;

    shared_ptr<Graph> reset() const;
    shared_ptr<Graph> init() const;

    RepeatedTuple<float, operation_feature_size> get_operation_feature(shared_ptr<Operation>) const;
    RepeatedTuple<float, machine_feature_size> get_machine_feature(shared_ptr<Machine>) const;
    RepeatedTuple<float, AGV_feature_size> get_AGV_feature(shared_ptr<AGV>) const;
    tuple<shared_ptr<GraphFeature>, shared_ptr<IdIdxMapper>> features() const;

    tuple<size_t, size_t> progress() const;
    bool finished() const;

    float finish_time_lower_bound() const;

    vector<Action> get_available_actions();

    bool operation_time_compare(OperationId, OperationId);
    bool AGV_time_compare(AGVId, AGVId);

    void act_move(AGVId, MachineId);
    void act_pick(AGVId, MachineId, Product);
    void act_transport(AGVId, MachineId);
    void act_wait();
    shared_ptr<Graph> act(Action) const;

    void wait_operation();
    void wait_AGV();

protected:
    OperationId next_operation_id;
    MachineId next_machine_id;
    AGVId next_AGV_id;

    bool inited;
    float timestamp;

    map<OperationId, shared_ptr<Operation>> operations;
    map<MachineId, shared_ptr<Machine>> machines;
    map<AGVId, shared_ptr<AGV>> AGVs;

    set<tuple<MachineId, MachineId>> direct_paths;
    map<MachineId, map<MachineId, tuple<vector<MachineId>, float>>> paths;

    ProcessingOperationQueue processing_operations;
    MovingAGVQueue moving_AGVs;
};

struct GraphFeature
{
    tuple<float, float, float> global_feature; // timestamp, total_task, finish_rate
    vector<RepeatedTuple<float, Graph::operation_feature_size>> operation_features;
    vector<tuple<size_t, size_t>> predecessor_idx; // predecessor_idx, successor_idx;
    vector<tuple<size_t, size_t>> successor_idx;   // successor_idx, predecessor_idx
    vector<RepeatedTuple<float, Graph::machine_feature_size>> machine_features;
    vector<tuple<size_t, size_t>> processable_idx;         // machine_idx, operation_idx
    vector<tuple<size_t, size_t, float>> processing;       // machine_idx, operation_idx, rest_time
    vector<tuple<size_t, size_t, size_t, size_t>> waiting; // machine_idx, operation_idx, total, current
    vector<RepeatedTuple<float, Graph::AGV_feature_size>> AGV_features;
    vector<tuple<size_t, size_t>> AGV_position;       // AGV_idx, machine_idx
    vector<tuple<size_t, size_t, float>> AGV_target;  // AGV_idx, machine_idx, rest_time
    vector<tuple<size_t, size_t, size_t>> AGV_loaded; // AGV_idx, from_operation_idx, to_operation_idx
};

struct IdIdxMapper
{
    map<OperationId, size_t> operation;
    map<MachineId, size_t> machine;
    map<AGVId, size_t> AGV;
};

#endif