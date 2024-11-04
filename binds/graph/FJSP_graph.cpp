#include <variant>
#include <optional>
#include <memory>
#include <format>
#include <ranges>
#include <iostream>
#include <concepts>
#include <random>
#include <functional>
#include <limits>
#include <execution>

#include <map>
#include <tuple>
#include <set>
#include <sstream>
#include <string>
#include <queue>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FJSP_graph.h"
#include "utils.h"

using namespace std;
namespace py = pybind11;
namespace ph = placeholders;

string Product::repr()
{
    return format("<Product from: {} to: {}>", this->from, this->to);
}

Operation::Operation(OperationId id, MachineType mt, float pt) : id(id),
                                                                 status(OperationStatus::blocked),
                                                                 machine_type(mt),
                                                                 process_time(pt),
                                                                 finish_timestamp(0),
                                                                 processing_machine(nullopt)
{
}

string Operation::repr()
{
    stringstream ss;
    if (this->status != OperationStatus::processing)
    {
        ss << format("<operation (id: {}, type: {}, process_time: {:.2f}, status: {})\n",
                     this->id, this->machine_type, this->process_time, enum_string(this->status));
    }
    else
    {
        ss << format("<operation (id: {}, type: {}, process_time: {:.2f}, status: {}, finish_timestamp: {:.2f})\n",
                     this->id, this->machine_type, this->process_time, enum_string(this->status), this->finish_timestamp);
    }

    ss << add_indent(format("p: {}", id_set_string(this->predecessors)));
    ss << add_indent(format("s: {}", id_set_string(this->successors)));
    ss << "\n>";
    return ss.str();
}

Machine::Machine(MachineId id, MachineType tp) : id(id),
                                                 type(tp),
                                                 status(MachineStatus::idle),
                                                 working_operation(nullopt)
{
}

string Machine::repr()
{
    return format("<machine (id: {}, type: {}, status: {}, working: {}, waiting: {})>",
                  this->id, this->type, enum_string(this->status), o2s(this->working_operation), o2s(this->waiting_operation));
}

AGV::AGV(AGVId id, float speed, MachineId init_pos) : id(id),
                                                      speed(speed),
                                                      status(AGVStatus::idle),
                                                      position(init_pos),
                                                      target_machine(init_pos),
                                                      finish_timestamp(0),
                                                      loaded_item(nullopt),
                                                      target_item(nullopt)
{
}

string AGV::repr()
{
    switch (this->status)
    {
    case AGVStatus::idle:
        return format("<AGV (id: {}, speed: {:.2f}, status: {}, position: {}, loaded_item: {})>",
                      this->id, this->speed, enum_string(this->status), this->position, o2s(this->loaded_item));
    case AGVStatus::picking:
        return format("<AGV (id: {}, speed: {:.2f}, status: {}, from: {}, to: {}, finish_timestamp: {:.2f}, target_item: {})>",
                      this->id, this->speed, enum_string(this->status), this->position, this->target_machine, this->finish_timestamp, o2s(this->target_item));
    default:
        return format("<AGV (id: {}, speed: {:.2f}, status: {}, from: {}, to: {}, finish_timestamp: {:.2f}, loaded_item: {})>",
                      this->id, this->speed, enum_string(this->status), this->position, this->target_machine, this->finish_timestamp, o2s(this->loaded_item));
    }
}

Action::Action(ActionType type, AGVId AGV, MachineId machine) : type(type), act_AGV(AGV), target_machine(machine), target_product(nullopt)
{
    assert(type != ActionType::pick);
}

Action::Action(ActionType type, AGVId AGV, MachineId machine, Product product) : type(type), act_AGV(AGV), target_machine(machine), target_product(product)
{
    assert(type == ActionType::pick);
}

string Action::repr()
{
    if (this->type == ActionType::pick)
    {
        return format("<Action (type: {}, AGV_id: {}, target_machine: {}, target_product: {})>",
                      enum_string(this->type), this->act_AGV, this->target_machine, o2s(this->target_product));
    }
    else
    {
        return format("<Action (type: {}, AGV_id: {}, target_machine: {})>",
                      enum_string(this->type), this->act_AGV, this->target_machine);
    }
}

Graph::Graph() : inited(false),
                 timestamp(0),
                 processing_operations(ProcessingOperationQueue(bind(&Graph::operation_time_compare, this, ph::_1, ph::_2))),
                 moving_AGVs(MovingAGVQueue(bind(&Graph::AGV_time_compare, this, ph::_1, ph::_2))),
                 available_actions(nullopt)
{
    this->operations[this->begin_operation_id] = make_shared<Operation>(this->begin_operation_id, this->dummy_machine_type, 0.0f);
    this->operations[this->end_operation_id] = make_shared<Operation>(this->end_operation_id, this->dummy_machine_type, 0.0f);

    this->machines[this->dummy_machine_id] = make_shared<Machine>(this->dummy_machine_id, this->dummy_machine_type);

    this->next_operation_id = 1;
    this->next_machine_id = 1;
    this->next_AGV_id = 0;
}

Graph::Graph(const Graph &other) : processing_operations(ProcessingOperationQueue(bind(&Graph::operation_time_compare, this, ph::_1, ph::_2))),
                                   moving_AGVs(MovingAGVQueue(bind(&Graph::AGV_time_compare, this, ph::_1, ph::_2))),
                                   available_actions(nullopt)
{
    for (auto [id, ptr] : other.operations)
    {
        this->operations[id] = make_shared<Operation>(*ptr);
    }
    this->next_operation_id = other.next_operation_id;

    for (auto [id, ptr] : other.machines)
    {
        this->machines[id] = make_shared<Machine>(*ptr);
    }
    this->next_machine_id = other.next_machine_id;

    for (auto [id, ptr] : other.AGVs)
    {
        this->AGVs[id] = make_shared<AGV>(*ptr);
    }
    this->next_AGV_id = other.next_AGV_id;

    this->distances = other.distances;

    this->inited = other.inited;
    this->timestamp = other.timestamp;

    this->processing_operations.set_data(other.processing_operations.get_data());
    this->moving_AGVs.set_data(other.moving_AGVs.get_data());
}

OperationId Graph::add_operation(MachineType type, float process_time)
{
    auto node = make_shared<Operation>(this->next_operation_id++, type, process_time);
    this->operations[node->id] = node;
    node->predecessors = {this->begin_operation_id};
    this->operations[this->begin_operation_id]->successors.emplace(node->id);
    node->successors = {this->end_operation_id};
    this->operations[this->end_operation_id]->predecessors.emplace(node->id);
    return node->id;
}

void Graph::add_relation(OperationId from, OperationId to)
{
    assert(this->contains(from));
    assert(this->contains(to));
    auto p_node = this->operations[from];
    auto s_node = this->operations[to];
    if (p_node->successors.size() == 1 && p_node->successors.contains(this->end_operation_id))
    {
        p_node->successors.clear();
        this->operations[this->end_operation_id]->predecessors.erase(p_node->id);
    }
    if (s_node->predecessors.size() == 1 && s_node->predecessors.contains(this->begin_operation_id))
    {
        s_node->predecessors.clear();
        this->operations[this->begin_operation_id]->successors.erase(s_node->id);
    }
    p_node->successors.emplace(s_node->id);
    s_node->predecessors.emplace(p_node->id);
}

void Graph::remove_relation(OperationId from, OperationId to)
{
    assert(this->contains(from));
    assert(this->contains(to));
    auto p_node = this->operations[from];
    auto s_node = this->operations[to];
    p_node->successors.erase(s_node->id);
    s_node->predecessors.erase(p_node->id);
    if (p_node->successors.empty())
    {
        p_node->successors.emplace(this->end_operation_id);
        this->operations[this->end_operation_id]->predecessors.emplace(p_node->id);
    }
    if (s_node->predecessors.empty())
    {
        s_node->predecessors.emplace(this->begin_operation_id);
        this->operations[this->begin_operation_id]->successors.emplace(s_node->id);
    }
}

OperationId Graph::insert_operation(
    MachineType type,
    float process_time,
    optional<OperationId> predecessor,
    optional<OperationId> successor)
{
    assert(!(predecessor.has_value() && successor.has_value()));
    auto node = make_shared<Operation>(this->next_operation_id++, type, process_time);
    this->operations[node->id] = node;
    if (predecessor.has_value())
    {
        node->predecessors = {predecessor.value()};
        auto p_node = this->operations[predecessor.value()];
        node->successors = p_node->successors;
        p_node->successors = {node->id};
        for (auto s_id : node->successors)
        {
            auto s_node = this->operations[s_id];
            s_node->predecessors.erase(p_node->id);
            s_node->predecessors.emplace(node->id);
        }
    }
    else if (successor.has_value())
    {
        node->successors = {successor.value()};
        auto s_node = this->operations[successor.value()];
        node->predecessors = s_node->predecessors;
        s_node->predecessors = {node->id};
        for (auto p_id : node->predecessors)
        {
            auto p_node = this->operations[p_id];
            p_node->successors.erase(s_node->id);
            p_node->successors.emplace(node->id);
        }
    }
    else
    {
        node->predecessors = {this->begin_operation_id};
        node->successors = {this->end_operation_id};
        this->operations[this->begin_operation_id]->successors.emplace(node->id);
        this->operations[this->end_operation_id]->predecessors.emplace(node->id);
    }

    return node->id;
}

void Graph::remove_operation(OperationId id)
{
    assert(id != this->begin_operation_id && id != this->end_operation_id);

    auto node = this->operations[id];
    auto ps = node->predecessors | views::transform([this](OperationId p_id)
                                                    { return this->operations[p_id]; });
    auto ss = node->successors | views::transform([this](OperationId s_id)
                                                  { return this->operations[s_id]; });
    for (auto p_node : ps)
    {
        p_node->successors.erase(id);
    }
    for (auto s_node : ss)
    {
        s_node->predecessors.erase(id);
    }
    for (auto p_node : ps)
    {
        bool is_begin = p_node->id == this->begin_operation_id;
        for (auto s_node : ss)
        {
            bool is_end = s_node->id == this->end_operation_id;
            if (!(is_begin && is_end))
            {
                p_node->successors.emplace(s_node->id);
                s_node->predecessors.emplace(p_node->id);
            }
        }
    }
}

bool Graph::contains(OperationId id)
{
    return this->operations.contains(id);
}

shared_ptr<Operation> Graph::get_operation(OperationId id)
{
    return this->operations[id];
}

MachineId Graph::add_machine(MachineType type)
{
    auto node = make_shared<Machine>(this->next_machine_id++, type);
    this->machines[node->id] = node;
    return node->id;
}

shared_ptr<Machine> Graph::get_machine(MachineId id)
{
    return this->machines[id];
}

AGVId Graph::add_AGV(float speed, MachineId init_pos)
{
    auto new_AGV = make_shared<AGV>(this->next_AGV_id++, speed, init_pos);
    this->AGVs[new_AGV->id] = new_AGV;

    return new_AGV->id;
}

shared_ptr<AGV> Graph::get_AGV(AGVId id)
{
    return this->AGVs[id];
}

float Graph::get_timestamp()
{
    return this->timestamp;
}

void Graph::set_timestamp(float value)
{
    this->timestamp = value;
}

string Graph::repr()
{
    stringstream ss;
    ss << format("<job (timestamp: {:.2f})", this->timestamp);
    ss << format("\n operation: {}\n", this->operations.size());
    for (auto &&[id, operation] : this->operations)
    {
        ss << add_indent(operation->repr()) << '\n';
    }
    ss << format("\n machine: {}\n", this->machines.size());
    for (auto &&[_, machine] : this->machines)
    {
        ss << add_indent(machine->repr()) << '\n';
    }
    ss << format("\n AGV: {}\n", this->AGVs.size());
    for (auto &&[_, agv] : this->AGVs)
    {
        ss << add_indent(agv->repr()) << '\n';
    }
    ss << format("\n distance: \n");
    ss << add_indent("     ");
    for (auto &&[id, _] : this->machines)
    {
        ss << format("{:^6}", id);
    }
    for (auto &&[from, _] : this->machines)
    {
        ss << format("\n{:^6}", from);
        for (auto &&[to, _] : this->machines)
        {
            ss << format("{:^6.2f}", this->distances[from][to]);
        }
    }
    ss << "\n>";
    return ss.str();
}

void Graph::set_distance(MachineId from, MachineId to, float distance)
{
    this->distances.try_emplace(from, map<MachineId, float>{}).first->second[to] = distance;
}

void Graph::set_distance(map<MachineId, map<MachineId, float>> &other)
{
    this->distances = other;
}

void Graph::set_rand_distance(float min_distance, float max_distance)
{
    mt19937 engine(random_device{}());
    uniform_real_distribution distance_dist(min_distance, max_distance);
    for (auto [from_id, _] : this->machines)
    {
        for (auto [to_id, _] : this->machines)
        {
            this->set_distance(from_id, to_id, from_id == to_id ? 0 : distance_dist(engine));
        }
    }
}

float Graph::get_travel_time(MachineId from, MachineId to, AGVId agv)
{
    return this->distances[from][to] / this->AGVs[agv]->speed;
}

shared_ptr<Graph> Graph::rand_generate(
    size_t operation_count,
    size_t machine_count,
    size_t AGV_count,
    size_t machine_type_count,
    float min_transport_time,
    float max_transport_time,
    float min_max_speed_ratio,
    float min_process_time,
    float max_process_time)
{
    if (machine_count < machine_type_count)
    {
        throw py::value_error();
    }
    auto ret = make_shared<Graph>();

    mt19937 engine(random_device{}());

    MachineType min_machine_type = Graph::dummy_machine_type + 1;
    MachineType max_machine_type = Graph::dummy_machine_type + machine_type_count;
    uniform_int_distribution<MachineType> machine_type_dist(min_machine_type, max_machine_type);
    vector<MachineId> machines;

    for (MachineType t = min_machine_type; t <= max_machine_type; t++)
    {
        machines.emplace_back(ret->add_machine(t));
    }
    for (size_t i = machine_type_count; i < machine_count; i++)
    {
        machines.emplace_back(ret->add_machine(machine_type_dist(engine)));
    }

    float max_speed = 1, min_speed = min_max_speed_ratio;
    float max_distance = max_transport_time * min_speed;
    float min_distance = min_transport_time * max_speed;
    uniform_real_distribution distance_dist(min_distance, max_distance);
    for (MachineId from_id : machines)
    {
        for (MachineId to_id : machines)
        {
            ret->set_distance(from_id, to_id, from_id == to_id ? 0 : distance_dist(engine));
        }
    }

    uniform_real_distribution AGV_speed_dist(min_speed, max_speed);
    uniform_int_distribution<MachineId> machine_idx_dist(0, machines.size() - 1);
    for (size_t i = 0; i < AGV_count; i++)
    {
        ret->add_AGV(AGV_speed_dist(engine), machines[machine_idx_dist(engine)]);
    }

    uniform_real_distribution<float> process_time_dist(min_process_time, max_process_time);
    discrete_distribution<size_t> prev_count_dist({3, 6, 2, 1});
    set<OperationId> exist_ids;
    for (size_t i = 0; i < operation_count; i++)
    {
        OperationId new_id = ret->add_operation(machine_type_dist(engine), process_time_dist(engine));
        for (auto p_id : random_unique(exist_ids, min(prev_count_dist(engine), exist_ids.size())))
        {
            ret->add_relation(p_id, new_id);
        }
        exist_ids.emplace(new_id);
    }

    return ret;
}

void Graph::init()
{
    auto begin_operation = this->operations[this->begin_operation_id];
    begin_operation->status = OperationStatus::finished;
    auto dummy_machine = this->machines[this->dummy_machine_id];
    for (OperationId id : begin_operation->successors)
    {
        this->operations[id]->status = OperationStatus::unscheduled;
        dummy_machine->products.emplace(Product{this->begin_operation_id, id});
    }
    this->operations[this->end_operation_id]->status = OperationStatus::blocked;
    this->inited = true;
}

shared_ptr<Graph> Graph::copy()
{
    return make_shared<Graph>(*this);
}

RepeatedTuple<float, Graph::operation_feature_size> Graph::get_operation_feature(shared_ptr<Operation> operation)
{
    float status_is_blocked = operation->status == OperationStatus::blocked ? 1.0 : 0.0;
    float status_is_waiting = operation->status == OperationStatus::waiting ? 1.0 : 0.0;
    float status_is_processing = operation->status == OperationStatus::processing ? 1.0 : 0.0;
    float status_is_finished = operation->status == OperationStatus::finished ? 1.0 : 0.0;

    float total_process_time = operation->process_time;
    float rest_process_time = status_is_processing ? (operation->finish_timestamp - this->timestamp) : 0.0;

    float total_material = operation->status <= OperationStatus::waiting ? operation->predecessors.size() : 0;
    float rest_material = operation->status <= OperationStatus::waiting ? (total_material - operation->arrived_preds.size()) : 0;

    float total_product = operation->status >= OperationStatus::processing ? operation->successors.size() : 0;
    float rest_pruduct = operation->status >= OperationStatus::processing ? total_product - operation->sent_succs.size() : 0;

    return {
        status_is_blocked,
        status_is_waiting,
        status_is_processing,
        status_is_finished,
        total_process_time,
        rest_process_time,
        total_product,
        rest_pruduct};
}

RepeatedTuple<float, Graph::machine_feature_size> Graph::get_machine_feature(shared_ptr<Machine> machine)
{
    float status_is_idle = machine->status == MachineStatus::idle;
    float status_is_waiting_material = machine->status == MachineStatus::waiting_material;
    float status_is_working = machine->status == MachineStatus::working;

    float rest_process_time = machine->status == MachineStatus::working
                                  ? (this->operations[machine->waiting_operation.value()]->finish_timestamp - this->timestamp)
                                  : 0;

    float same_type_count = ranges::count_if(this->machines | views::values, [this, machine](auto other)
                                             { return other->type == machine->type; });
    auto processable_operations = this->operations | views::values | views::filter([this, machine](auto op)
                                                                                   { return op->machine_type == machine->type; });
    float processable_operation_count = ranges::distance(processable_operations);
    float rest_processable_operation_count = ranges::count_if(processable_operations, [](auto op)
                                                              { return op->status < OperationStatus::processing; });

    return {
        status_is_idle,
        status_is_waiting_material,
        status_is_working,
        rest_process_time,
        same_type_count,
        processable_operation_count,
        rest_processable_operation_count};
}
RepeatedTuple<float, Graph::AGV_feature_size> Graph::get_AGV_feature(shared_ptr<AGV> AGV)
{
    float status_is_idle = AGV->status == AGVStatus::idle;
    float status_is_moving = AGV->status == AGVStatus::moving;
    float status_is_picking = AGV->status == AGVStatus::picking;
    float status_is_transporting = AGV->status == AGVStatus::transporting;

    float rest_act_time = AGV->status == AGVStatus::idle ? 0 : AGV->finish_timestamp - this->timestamp;

    return {status_is_idle,
            status_is_moving,
            status_is_picking,
            status_is_transporting,
            rest_act_time};
}

shared_ptr<GraphFeature> Graph::features()
{
    assert(this->inited);

    auto ret = make_shared<GraphFeature>();

    ret->operation_features.resize(this->operations.size());
    auto operations_with_idx = this->operations | views::values | views::enumerate;
    map<OperationId, size_t> operation_id_idx_mapper;
    for (auto [i, operation] : operations_with_idx)
    {
        ret->operation_features[i] = this->get_operation_feature(operation);
        operation_id_idx_mapper[operation->id] = static_cast<size_t>(i);
    }

    ret->machine_features.resize(this->machines.size());
    auto machine_with_idx = this->machines | views::values | views::enumerate;
    map<MachineId, size_t> machine_id_idx_mapper;
    map<MachineType, vector<float>> machine_type_idx_mask;
    for (auto [i, machine] : machine_with_idx)
    {
        ret->machine_features[i] = Graph::get_machine_feature(machine);
        machine_id_idx_mapper[machine->id] = static_cast<size_t>(i);

        if(machine->working_operation.has_value())
        {
            auto operation = this->operations[machine->working_operation.value()];
            ret->processing.emplace_back(i, operation_id_idx_mapper[operation->id], operation->finish_timestamp - this->timestamp);
        }
        if (machine->waiting_operation.has_value())
        {
            auto operation = this->operations[machine->waiting_operation.value()];
            ret->waiting.emplace_back(i, operation_id_idx_mapper[machine->waiting_operation.value()], operation->predecessors.size(), operation->arrived_preds.size());
        }
    }

    for (auto [i, operation] : operations_with_idx)
    {
        for (auto p_id : operation->predecessors)
        {
            ret->predecessor_idx.emplace_back(i, operation_id_idx_mapper[p_id]);
        }
        for (auto s_id : operation->successors)
        {
            ret->successor_idx.emplace_back(i, operation_id_idx_mapper[s_id]);
        }
        for (auto [m_id, m] : machine_with_idx)
        {
            if (m->type == operation->machine_type)
            {
                ret->processable_idx.emplace_back(m_id, i);
            }
        }
    }

    ret->AGV_features.resize(this->AGVs.size());
    auto AGV_with_idx = this->AGVs | views::values | views::enumerate;
    for (auto [i, AGV] : AGV_with_idx)
    {
        ret->AGV_features[i] = Graph::get_AGV_feature(AGV);
        if (AGV->status == AGVStatus::idle)
        {
            ret->AGV_position.emplace_back(i, machine_id_idx_mapper[AGV->position]);
        }
        else
        {
            auto t = AGV->finish_timestamp - this->timestamp;
            ret->AGV_target.emplace_back(i, machine_id_idx_mapper[AGV->target_machine], t);
        }
        if (AGV->loaded_item.has_value())
        {
            auto p = AGV->loaded_item.value();
            ret->AGV_loaded.emplace_back(i, operation_id_idx_mapper[p.from], operation_id_idx_mapper[p.to]);
        }
    }

    return ret;
}

bool Graph::finished()
{
    return this->operations[this->end_operation_id]->status == OperationStatus::finished;
}

double Graph::finish_time_lower_bound()
{
    map<MachineType, float> remain_process_time;
    float remain_transport_distance = 0;

    map<pair<MachineType, MachineType>, float> min_type_distance;
    map<MachineType, size_t> type_count;
    for (auto [fid, from] : this->machines)
    {
        for (auto [tid, to] : this->machines)
        {
            float distance = this->distances[fid][tid];
            auto [iter, new_add] = min_type_distance.try_emplace({from->type, to->type}, distance);
            if (!new_add)
            {
                iter->second = min(distance, iter->second);
            }
        }
        type_count.try_emplace(from->type, 0).first->second++;
    }
    for (auto operation : this->operations | views::values)
    {

        auto [iter, _] = remain_process_time.try_emplace(operation->machine_type, 0.0f);
        if (operation->status <= OperationStatus::waiting)
        {
            iter->second += operation->process_time;
        }
        else if (operation->status == OperationStatus::processing)
        {
            iter->second += operation->finish_timestamp - this->timestamp;
        }

        set<OperationId> unarrived_pred_ids;
        ranges::set_difference(operation->predecessors, operation->arrived_preds, inserter(unarrived_pred_ids, unarrived_pred_ids.end()));
        for (auto unarrived_pred : unarrived_pred_ids | views::transform([this](auto id)
                                                                         { return this->operations[id]; }))
        {
            if (!unarrived_pred->sent_succs.contains(operation->id))
            {
                remain_transport_distance += min_type_distance[{unarrived_pred->machine_type, operation->machine_type}];
            }
        }
    }

    for (auto AGV : this->AGVs | views::values)
    {
        if (AGV->status != AGVStatus::idle)
        {
            remain_transport_distance += (AGV->finish_timestamp - this->timestamp) * AGV->speed;
        }
    }

    float max_time = 0;
    for (auto [type, total_time] : remain_process_time)
    {
        max_time = max(max_time, total_time / type_count[type]);
    }

    float total_speed = 0;
    for (auto AGV : this->AGVs | views::values)
    {
        total_speed += AGV->speed;
    }
    max_time = max(max_time, remain_transport_distance / total_speed);

    return this->timestamp + max_time;
}

vector<Action> Graph::get_available_actions()
{
    assert(this->inited);

    if (this->available_actions.has_value())
    {
        return this->available_actions.value();
    }

    vector<Action> ret;

    vector<shared_ptr<Machine>> transportable_machines;
    vector<pair<shared_ptr<Machine>, Product>> pickable_products;
    for (auto [id, machine] : this->machines)
    {
        if (!machine->waiting_operation.has_value())
        {
            transportable_machines.emplace_back(machine);
        }
        if (!machine->products.empty())
        {
            for (auto product : machine->products)
            {
                if (ranges::none_of(this->AGVs | views::values, [product](auto AGV)
                                    { return AGV->target_item == product; }))
                {
                    pickable_products.emplace_back(machine, product);
                }
            }
        }
    }

    for (auto [AGV_id, AGV] : this->AGVs)
    {
        if (AGV->status != AGVStatus::idle)
        {
            continue;
        }
        if (AGV->loaded_item.has_value())
        {
            auto operation = this->operations[AGV->loaded_item->to];
            if (operation->processing_machine.has_value())
            {
                auto machine = this->machines[operation->processing_machine.value()];
                assert(operation->status == OperationStatus::blocked || operation->status == OperationStatus::waiting);
                assert(operation->machine_type == machine->type);
                ret.emplace_back(ActionType::transport, AGV_id, machine->id);
            }
            else
            {
                for (auto machine : transportable_machines)
                {
                    if (machine->type == operation->machine_type)
                    {
                        ret.emplace_back(ActionType::transport, AGV_id, machine->id);
                    }
                }
            }
        }
        else
        {
            for (auto [machine, product] : pickable_products)
            {
                ret.emplace_back(ActionType::pick, AGV_id, machine->id, product);
            }
        }

        for (auto machine_id : this->machines | views::keys)
        {
            if (machine_id == AGV->position)
            {
                continue;
            }
            ret.emplace_back(ActionType::move, AGV_id, machine_id);
        }
    }
    this->available_actions = ret;
    return ret;
}

bool Graph::operation_time_compare(OperationId a, OperationId b)
{
    return this->operations[a]->finish_timestamp > this->operations[b]->finish_timestamp;
}

bool Graph::AGV_time_compare(AGVId a, AGVId b)
{
    return this->AGVs[a]->finish_timestamp > this->AGVs[b]->finish_timestamp;
}

void Graph::act_move(AGVId id, MachineId target)
{
    auto AGV = this->AGVs[id];
    assert(AGV->status == AGVStatus::idle);
    AGV->target_machine = target;
    AGV->status = AGVStatus::moving;
    AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;
    this->moving_AGVs.push(id);
}

void Graph::act_pick(AGVId id, MachineId target, Product product)
{
    auto AGV = this->AGVs[id];
    assert(AGV->status == AGVStatus::idle && !AGV->loaded_item.has_value());

    auto target_machine = this->machines[target];
    assert(target_machine->products.contains(product));

    auto transport_operation = this->operations[product.to];

    AGV->target_machine = target;
    AGV->target_item = product;
    AGV->status = AGVStatus::picking;
    AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;
    this->moving_AGVs.push(id);
}

void Graph::act_transport(AGVId id, MachineId target)
{
    auto AGV = this->AGVs[id];
    assert(AGV->status == AGVStatus::idle && AGV->loaded_item.has_value());
    auto product = AGV->loaded_item.value();
    auto target_machine = this->machines[target];
    auto target_operation = this->operations[product.to];
    assert(target_operation->predecessors.contains(AGV->loaded_item->from));
    assert(target_operation->machine_type == target_machine->type);

    if (target_machine->waiting_operation.has_value())
    {
        if(target_machine->working_operation.has_value())
        {
            assert(target_machine->status == MachineStatus::working);
        }
        else
        {
            assert(target_machine->status == MachineStatus::waiting_material);
        }
        assert(target_machine->waiting_operation == target_operation->id);
        assert(target_operation->status <= OperationStatus::waiting);
        assert(target_operation->processing_machine == target_machine->id);
    }
    else
    {
        assert(target_operation->status <= OperationStatus::unscheduled);
        assert(!target_operation->processing_machine.has_value());
        if (target_machine->status == MachineStatus::idle)
        {
            target_machine->status = MachineStatus::waiting_material;
        }
        target_machine->waiting_operation = target_operation->id;
        if (target_operation->status == OperationStatus::unscheduled)
        {
            target_operation->status = OperationStatus::waiting;
        }
        target_operation->processing_machine = target_machine->id;
    }

    AGV->target_machine = target;
    AGV->status = AGVStatus::transporting;
    AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;

    this->moving_AGVs.push(id);
}

shared_ptr<Graph> Graph::act(Action action)
{
    assert(this->inited);

    auto ret = this->copy();
    switch (action.type)
    {
    case ActionType::move:
        ret->act_move(action.act_AGV, action.target_machine);
        break;

    case ActionType::pick:
        ret->act_pick(action.act_AGV, action.target_machine, action.target_product.value());
        break;

    case ActionType::transport:
        ret->act_transport(action.act_AGV, action.target_machine);
        break;
    default:
        assert(false);
    }
    return ret;
}

void Graph::wait_operation()
{
    auto operation = this->operations[this->processing_operations.top()];
    this->processing_operations.pop();
    assert(operation->status == OperationStatus::processing);
    assert(operation->processing_machine.has_value());
    this->timestamp = operation->finish_timestamp;
    operation->status = OperationStatus::finished;
    auto machine = this->machines[operation->processing_machine.value()];
    assert(machine->status == MachineStatus::working);
    assert(machine->working_operation == operation->id);
    machine->status = MachineStatus::idle;
    machine->working_operation = nullopt;
    for (auto s_id : operation->successors)
    {
        machine->products.emplace(Product{operation->id, s_id});
        auto successor = this->operations[s_id];
        assert(successor->status == OperationStatus::blocked);
        if (ranges::all_of(successor->predecessors, [this](auto p_id)
                           { return this->operations[p_id]->status == OperationStatus::finished; }))
        {
            if (successor->processing_machine.has_value())
            {
                successor->status = OperationStatus::waiting;
            }
            else
            {
                successor->status = OperationStatus::unscheduled;
            }
        }
    }

    if (machine->waiting_operation.has_value())
    {
        machine->status = MachineStatus::waiting_material;
        auto operation = this->operations[machine->waiting_operation.value()];
        assert(operation->status == OperationStatus::blocked || operation->status == OperationStatus::waiting);
        if (operation->status == OperationStatus::waiting && ranges::all_of(operation->predecessors, [machine](auto id)
                                                                            { return machine->materials.contains(Product{id, machine->waiting_operation.value()}); }))
        {
            for (auto p_id : operation->predecessors)
            {
                machine->materials.erase(Product{p_id, operation->id});
            }
            machine->working_operation = machine->waiting_operation;
            machine->waiting_operation = nullopt;
            machine->status = MachineStatus::working;
            operation->status = OperationStatus::processing;
            operation->finish_timestamp = this->timestamp + operation->process_time;
            this->processing_operations.push(operation->id);
        }
    }
}

void Graph::wait_AGV()
{
    auto AGV = this->AGVs[this->moving_AGVs.top()];
    this->moving_AGVs.pop();

    this->timestamp = AGV->finish_timestamp;

    switch (AGV->status)
    {
    case AGVStatus::moving:
    {
        AGV->status = AGVStatus::idle;
        AGV->position = AGV->target_machine;
        break;
    }
    case AGVStatus::picking:
    {
        assert(AGV->target_item.has_value());
        assert(!AGV->loaded_item.has_value());
        auto target_machine = this->machines[AGV->target_machine];
        assert(target_machine->products.contains(AGV->target_item.value()));
        auto operation = this->operations[AGV->target_item->from];
        assert(operation->status == OperationStatus::finished);
        assert(operation->successors.contains(AGV->target_item->to));
        assert(!operation->sent_succs.contains(AGV->target_item->to));

        target_machine->products.erase(AGV->target_item.value());
        AGV->loaded_item = AGV->target_item.value();
        AGV->target_item = nullopt;
        operation->sent_succs.emplace(AGV->loaded_item->to);

        AGV->status = AGVStatus::idle;
        AGV->position = AGV->target_machine;

        break;
    }

    case AGVStatus::transporting:
    {
        assert(AGV->loaded_item.has_value());
        auto machine = this->machines[AGV->target_machine];
        assert(machine->waiting_operation == AGV->loaded_item->to);
        assert(machine->status == MachineStatus::waiting_material || machine->status == MachineStatus::working);
        auto operation = this->operations[AGV->loaded_item->to];
        assert(operation->status == OperationStatus::blocked || operation->status == OperationStatus::waiting);
        assert(operation->machine_type == machine->type);
        assert(operation->predecessors.contains(AGV->loaded_item->from));
        assert(!operation->arrived_preds.contains(AGV->loaded_item->from));

        AGV->status = AGVStatus::idle;
        AGV->position = AGV->target_machine;
        machine->materials.emplace(AGV->loaded_item.value());
        operation->arrived_preds.emplace(AGV->loaded_item->from);
        AGV->loaded_item = nullopt;
        bool processable = operation->arrived_preds.size() == operation->predecessors.size();
        processable &= machine->status == MachineStatus::waiting_material;
        processable &= operation->status == OperationStatus::waiting;
        if (processable)
        {
            for (auto p_id : operation->arrived_preds)
            {
                assert(machine->materials.contains(Product{p_id, operation->id}));
                machine->materials.erase({p_id, operation->id});
            }
            machine->status = MachineStatus::working;
            machine->working_operation = machine->waiting_operation;
            machine->waiting_operation = nullopt;

            operation->status = OperationStatus::processing;
            operation->finish_timestamp = this->timestamp + operation->process_time;
            this->processing_operations.push(operation->id);
        }

        break;
    }

    default:
        throw logic_error(format("wrong AGV status: {}", enum_string(AGV->status)));
    }
}

void Graph::_wait(float delta_time)
{
    float nearest_operation_finish_time = DBL_MAX;
    float nearest_AGV_finish_time = DBL_MAX;
    if (!this->processing_operations.empty())
    {
        nearest_operation_finish_time = this->operations[this->processing_operations.top()]->finish_timestamp;
    }

    if (!this->moving_AGVs.empty())
    {
        nearest_AGV_finish_time = this->AGVs[this->moving_AGVs.top()]->finish_timestamp;
    }

    if (nearest_operation_finish_time <= this->timestamp + delta_time)
    {
        wait_operation();
    }
    else if (nearest_AGV_finish_time <= this->timestamp + delta_time)
    {
        wait_AGV();
    }
    else
    {
        this->timestamp += delta_time;
    }
}

shared_ptr<Graph> Graph::wait(float delta_time)
{
    auto ret = this->copy();
    ret->_wait(delta_time);
    return ret;
}
