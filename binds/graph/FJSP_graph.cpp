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
#include <unordered_set>
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

OperationBase::OperationBase(OperationId id, MachineType mt) : id(id),
                                                               machine_type(mt),
                                                               status(OperationStatus::blocked)
{
}

GraphBegin::GraphBegin(OperationId id, MachineType mt) : OperationBase(id, mt),
                                                         sent_count(0),
                                                         arrived_count(0)
{
}

string GraphBegin::repr()
{
    stringstream ss;
    ss << format("<begin (id: {}, total: {}, sent: {}, status: {})\n",
                 this->id, this->successors.size(), this->sent_count, enum_string(this->status));
    ss << add_indent(format("s: {}", id_set_string(this->successors)));
    ss << "\n>";
    return ss.str();
}

GraphEnd::GraphEnd(OperationId id, MachineType mt) : OperationBase(id, mt),
                                                     recive_cound(0)
{
}

string GraphEnd::repr()
{
    stringstream ss;
    ss << format("<end (id: {}, total: {}, recived: {}, status: {})\n",
                 this->id, this->predecessors.size(), this->recive_cound, enum_string(this->status));
    ss << add_indent(format("p: {}", id_set_string(this->predecessors)));
    ss << "\n>";
    return ss.str();
}

Operation::Operation(OperationId id, MachineType mt, float pt) : OperationBase(id, mt),
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

    ss << add_indent(format("p: {}", this->predecessor));
    ss << add_indent(format("s: {}", this->successor));
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
    return format("<machine (id: {}, type: {}, status: {})>", this->id, this->type, enum_string(this->status));
}

AGV::AGV(AGVId id, float speed, MachineId init_pos) : id(id),
                                                      speed(speed),
                                                      status(AGVStatus::idle),
                                                      position(init_pos),
                                                      target(0),
                                                      transport_target(nullopt),
                                                      finish_timestamp(0),
                                                      loaded_item(nullopt)
{
}

string AGV::repr()
{
    switch (this->status)
    {
    case AGVStatus::idle:
        return format("<AGV (speed: {:.2f}, status: {}, position: {}, loaded_item: {})>",
                      this->speed, enum_string(this->status), this->position, o2s(this->loaded_item));
    default:
        return format("<AGV (speed: {:.2f}, status: {}, from: {}, to: {}, finish_timestamp: {:.2f}, loaded_item: {})>",
                      this->speed, enum_string(this->status), this->position, this->target, this->finish_timestamp, o2s(this->loaded_item));
    }
}

Action::Action(ActionType type, AGVId agv, MachineId m, optional<OperationId> t = nullopt) : type(type),
                                                                                             act_AGV(agv),
                                                                                             target_machine(m),
                                                                                             target_operation(t)
{
}

string Action::repr()
{
    return format("<Action (type: {}, AGV_id: {}, target_machine: {}, target_operation: {})>",
                  enum_string(this->type), this->act_AGV, this->target_machine, o2s(this->target_operation));
}

Graph::Graph() : timestamp(0),
                 processing_operations(ProcessingOperationQueue(bind(&Graph::operation_time_compare, this, ph::_1, ph::_2))),
                 moving_AGVs(MovingAGVQueue(bind(&Graph::AGV_time_compare, this, ph::_1, ph::_2))),
                 available_actions(nullopt)
{
    this->operations[this->begin_operation_id] = make_shared<GraphBegin>(this->begin_operation_id, this->dummy_machine_type);
    this->operations[this->end_operation_id] = make_shared<GraphEnd>(this->end_operation_id, this->dummy_machine_type);

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
        visit([id, this]<typename T>(shared_ptr<T> p)
              { this->operations[id] = make_shared<T>(*p); }, ptr);
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

    this->timestamp = other.timestamp;

    this->processing_operations.set_data(other.processing_operations.get_data());
    this->moving_AGVs.set_data(other.moving_AGVs.get_data());
}

OperationId Graph::add_operation(
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
        node->predecessor = predecessor.value();
        auto p_node = get<shared_ptr<Operation>>(this->operations[predecessor.value()]);
        node->successor = p_node->successor;
        p_node->successor = node->id;
        if (node->successor == this->end_operation_id)
        {
            auto &&job_end_node = get<shared_ptr<GraphEnd>>(this->operations[this->end_operation_id]);
            job_end_node->predecessors.erase(p_node->id);
            job_end_node->predecessors.emplace(node->id);
        }
    }
    else if (successor.has_value())
    {
        node->successor = successor.value();
        auto s_node = get<shared_ptr<Operation>>(this->operations[successor.value()]);
        node->predecessor = s_node->predecessor;
        s_node->predecessor = node->id;
        if (node->predecessor == this->begin_operation_id)
        {
            auto &&job_begin_node = get<shared_ptr<GraphBegin>>(this->operations[this->begin_operation_id]);
            job_begin_node->successors.erase(s_node->id);
            job_begin_node->successors.emplace(node->id);
        }
    }
    else
    {
        node->predecessor = this->begin_operation_id;
        node->successor = this->end_operation_id;
        get<shared_ptr<GraphBegin>>(this->operations[this->begin_operation_id])->successors.emplace(node->id);
        get<shared_ptr<GraphEnd>>(this->operations[this->end_operation_id])->predecessors.emplace(node->id);
    }

    return node->id;
}

void Graph::remove_operation(OperationId id)
{
    assert(id != this->begin_operation_id && id != this->end_operation_id);

    auto node_ptr = get<shared_ptr<Operation>>(this->operations[id]);
    if (holds_alternative<shared_ptr<GraphBegin>>(this->operations[node_ptr->predecessor]))
    {
        get<shared_ptr<GraphBegin>>(this->operations[node_ptr->predecessor])->successors.erase(id);
    }
    else
    {
        get<shared_ptr<Operation>>(this->operations[node_ptr->predecessor])->successor = node_ptr->successor;
    }

    if (holds_alternative<shared_ptr<GraphEnd>>(this->operations[node_ptr->successor]))
    {
        get<shared_ptr<GraphEnd>>(this->operations[node_ptr->successor])->predecessors.erase(id);
    }
    else
    {
        get<shared_ptr<Operation>>(this->operations[node_ptr->successor])->predecessor = node_ptr->predecessor;
    }
}

bool Graph::contains(OperationId id)
{
    return this->operations.contains(id);
}

Graph::OperationNodePtr Graph::get_operation(OperationId id)
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
        ss << add_indent(to_operation_base_ptr(operation)->repr()) << '\n';
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
    vector<size_t> operation_counts,
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
    for (size_t i = 0; i < machine_count - machine_type_count; i++)
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
    discrete_distribution<size_t> prev_count_dist({3, 5, 3, 1});
    discrete_distribution<size_t> repeat_count_dist({5, 3, 1});
    for (size_t operation_count : operation_counts)
    {
        OperationId prev_id;
        for (size_t i = 0; i < operation_count; i++)
        {
            prev_id = ret->add_operation(machine_type_dist(engine), process_time_dist(engine), i == 0 ? nullopt : optional{prev_id}, nullopt);
        }
    }

    return ret;
}

void Graph::init_operation_status()
{
    auto begin_operation = get<shared_ptr<GraphBegin>>(this->operations[this->begin_operation_id]);
    begin_operation->status = OperationStatus::need_transport;
    this->machines[this->dummy_machine_id]->status = MachineStatus::holding_product;
    for (OperationId id : begin_operation->successors)
    {
        auto operation = get<shared_ptr<Operation>>(this->operations[id]);
        if (operation->status == OperationStatus::blocked)
        {
            operation->status = OperationStatus::waiting_machine;
        }
    }
    get<shared_ptr<GraphEnd>>(this->operations[this->end_operation_id])->status = OperationStatus::blocked;
}

shared_ptr<Graph> Graph::copy()
{
    return make_shared<Graph>(*this);
}

RepeatedTuple<float, Graph::operation_feature_size> Graph::get_operation_feature(shared_ptr<Operation> operation)
{
    float status_is_blocked = operation->status == OperationStatus::blocked ? 1.0 : 0.0;
    float status_is_waiting_machine = operation->status == OperationStatus::waiting_machine ? 1.0 : 0.0;
    float status_is_waiting_material = operation->status == OperationStatus::waiting_material ? 1.0 : 0.0;
    float status_is_processing = operation->status == OperationStatus::processing ? 1.0 : 0.0;
    float status_is_need_transport = operation->status == OperationStatus::need_transport ? 1.0 : 0.0;
    float status_is_waiting_transport = operation->status == OperationStatus::waiting_transport ? 1.0 : 0.0;
    float status_is_transporting = operation->status == OperationStatus::transporting ? 1.0 : 0.0;
    float status_is_finished = operation->status == OperationStatus::finished ? 1.0 : 0.0;

    float total_process_time = operation->process_time;
    float rest_process_time = status_is_processing ? (operation->finish_timestamp - this->timestamp) : 0.0;

    return {
        status_is_blocked,
        status_is_waiting_machine,
        status_is_waiting_material,
        status_is_processing,
        status_is_need_transport,
        status_is_waiting_transport,
        status_is_transporting,
        status_is_finished,
        total_process_time,
        rest_process_time,
    };
}

RepeatedTuple<float, Graph::machine_feature_size> Graph::get_machine_feature(shared_ptr<Machine> machine)
{
    // TODO
    return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
RepeatedTuple<float, Graph::AGV_feature_size> Graph::get_AGV_feature(shared_ptr<AGV> AGV)
{
    // TODO
    return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}

shared_ptr<GraphFeatures> Graph::features()
{
    auto ret = make_shared<GraphFeatures>();

    vector<shared_ptr<Operation>> normal_operations;
    for (auto [_, ptr] : this->operations)
    {
        if (holds_alternative<shared_ptr<Operation>>(ptr))
        {
            normal_operations.emplace_back(get<shared_ptr<Operation>>(ptr));
        }
    }

    ret->operation_features.resize(normal_operations.size());
    auto operations_with_idx = views::enumerate(normal_operations);
    map<OperationId, size_t> operation_id_idx_mapper;
    for (auto [i, operation] : operations_with_idx)
    {
        ret->operation_features[i] = this->get_operation_feature(operation);
        operation_id_idx_mapper[operation->id] = static_cast<size_t>(i);
    }

    ret->job_index.resize(normal_operations.size());
    auto begin_op = get<shared_ptr<GraphBegin>>(this->operations[this->begin_operation_id]);
    for (auto [idx, first_id] : views::enumerate(begin_op->successors))
    {
        auto id = first_id;
        while (id != this->end_operation_id)
        {
            ret->job_index[operation_id_idx_mapper[id]] = idx;
            id = get<shared_ptr<Operation>>(this->operations[id])->successor;
        }
    }

    ret->machine_type.resize(normal_operations.size());
    for (auto [idx, operation] : operations_with_idx)
    {
        ret->machine_type[idx] = operation->machine_type;
    }

    ret->machine_features.resize(this->machines.size());
    auto machine_with_idx = views::enumerate(this->machines | views::values);
    map<MachineId, size_t> machine_id_idx_mapper;
    map<MachineType, vector<float>> machine_type_idx_mask;
    for (auto [i, machine] : machine_with_idx)
    {
        ret->machine_features[i] = Graph::get_machine_feature(machine);
        machine_id_idx_mapper[machine->id] = static_cast<size_t>(i);
        auto [iter, _] = machine_type_idx_mask.try_emplace(machine->type, vector<float>(this->machines.size(), 0));
        iter->second[i] = 1;
    }

    ret->operation_relations.resize(normal_operations.size());
    ret->processable_machine_mask.resize(normal_operations.size());
    for (auto [i, operation] : operations_with_idx)
    {
        int p = operation->predecessor == this->begin_operation_id
                    ? -1
                    : operation_id_idx_mapper[operation->predecessor];
        int s = operation->successor == this->end_operation_id
                    ? -1
                    : operation_id_idx_mapper[operation->successor];
        ret->operation_relations[i] = {p, s};
        ret->processable_machine_mask[i] = machine_type_idx_mask[operation->machine_type];
    }

    ret->AGV_features.resize(this->AGVs.size());
    auto AGV_with_idx = views::enumerate(this->AGVs | views::values);
    for (auto [i, AGV] : AGV_with_idx)
    {
        ret->AGV_features[i] = Graph::get_AGV_feature(AGV);
    }

    return ret;
}

bool Graph::finished()
{
    return get<shared_ptr<GraphEnd>>(this->operations[this->end_operation_id])->status == OperationStatus::finished;
}

double Graph::finish_time_lower_bound()
{
    map<MachineType, float> remain_process_time;
    float remain_transport_distance = 0.0f;

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
    for (auto ptr : this->operations | views::values)
    {
        if (!holds_alternative<shared_ptr<Operation>>(ptr))
        {
            continue;
        }
        auto operation = get<shared_ptr<Operation>>(ptr);

        auto [iter, _] = remain_process_time.try_emplace(operation->machine_type, 0.0f);
        if (operation->status <= OperationStatus::waiting_material)
        {
            iter->second += operation->process_time;
        }
        else if (operation->status == OperationStatus::processing)
        {
            iter->second += operation->finish_timestamp - this->timestamp;
        }

        auto predecessor = to_operation_base_ptr(this->operations[operation->predecessor]);
        if (operation->status <= OperationStatus::waiting_machine)
        {
            remain_transport_distance += min_type_distance[{predecessor->machine_type, operation->machine_type}];
        }
        else if (operation->status == OperationStatus::waiting_material)
        {
            for (auto AGV : this->AGVs | views::values)
            {
                if (AGV->status == AGVStatus::transporting && AGV->loaded_item.value() == predecessor->id)
                {
                    remain_transport_distance += (AGV->finish_timestamp - this->timestamp) * AGV->speed;
                    break;
                }
            }
        }
    }

    float max_time = 0;
    for(auto [type, total_time] : remain_process_time)
    {
        max_time = max(max_time, total_time / type_count[type]);
    }

    float total_speed = 0;
    for(auto AGV : this->AGVs | views::values)
    {
        total_speed += AGV->speed;
    }
    max_time = max(max_time, remain_transport_distance / total_speed);

    return this->timestamp + max_time;
}

vector<Action> Graph::get_available_actions()
{
    if (this->available_actions.has_value())
    {
        return this->available_actions.value();
    }

    vector<Action> ret;

    vector<shared_ptr<Machine>> idle_machines;
    vector<MachineId> pickable_machines;
    for (auto [id, machine] : this->machines)
    {
        switch (machine->status)
        {
        case MachineStatus::idle:
            idle_machines.push_back(machine);
            break;
        case MachineStatus::holding_product:
            if (id == this->dummy_machine_id || to_operation_base_ptr(this->operations[machine->working_operation.value()])->status == OperationStatus::need_transport)
            {
                pickable_machines.push_back(id);
            }
            break;
        default:
            break;
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
            visit([&]<typename T>(shared_ptr<T> operation)
                  {
                    if constexpr (is_same_v<T, GraphBegin>)
                    {
                        for(OperationId sid : operation->successors)
                        {
                            auto s = get<shared_ptr<Operation>>(this->operations[sid]);
                            if(s->status == OperationStatus::waiting_machine)
                            {
                                for(auto machine : idle_machines)
                                {
                                    if(machine->type == s->machine_type)
                                    {
                                        ret.emplace_back(ActionType::transport, AGV_id, machine->id, sid);
                                    }
                                }
                            }
                        }
                    }
                    else if constexpr (is_same_v<T, Operation>)
                    {
                        if(operation->successor == this->end_operation_id)
                        {
                            ret.emplace_back(ActionType::transport, AGV_id, this->dummy_machine_id, operation->successor);
                        }
                        else
                        {
                            for(auto machine : idle_machines)
                            {
                                if(machine->type == to_operation_base_ptr(this->operations[operation->successor])->machine_type)
                                {
                                    ret.emplace_back(ActionType::transport, AGV_id, machine->id, operation->successor);
                                }
                            }
                        }
                    }
                    else
                    {
                        unreachable();
                    } }, this->operations[AGV->loaded_item.value()]);
        }
        else
        {
            for (auto machine_id : pickable_machines)
            {
                ret.emplace_back(ActionType::pick, AGV_id, machine_id);
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
    }
    this->available_actions = ret;
    return ret;
}

bool Graph::operation_time_compare(OperationId a, OperationId b)
{
    return get<shared_ptr<Operation>>(this->operations[a])->finish_timestamp > get<shared_ptr<Operation>>(this->operations[b])->finish_timestamp;
}

bool Graph::AGV_time_compare(AGVId a, AGVId b)
{
    return this->AGVs[a]->finish_timestamp > this->AGVs[b]->finish_timestamp;
}

void Graph::act_move(AGVId id, MachineId target)
{
    auto AGV = this->AGVs[id];
    assert(AGV->status == AGVStatus::idle);
    AGV->target = target;
    AGV->status = AGVStatus::moving;
    AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;
    this->moving_AGVs.push(id);
}

void Graph::act_pick(AGVId id, MachineId target)
{
    auto AGV = this->AGVs[id];
    assert(AGV->status == AGVStatus::idle && !AGV->loaded_item.has_value());

    if (target != this->dummy_machine_id)
    {
        auto target_machine = this->machines[target];
        assert(target_machine->status == MachineStatus::holding_product);
        assert(target_machine->working_operation.has_value());

        auto transport_operation = get<shared_ptr<Operation>>(this->operations[target_machine->working_operation.value()]);
        assert(transport_operation->status == OperationStatus::need_transport);

        AGV->target = target;
        AGV->status = AGVStatus::picking;
        AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;

        transport_operation->status = OperationStatus::waiting_transport;
    }
    else
    {
        AGV->target = target;
        AGV->status = AGVStatus::picking;
        AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;

        auto begin_operation = get<shared_ptr<GraphBegin>>(this->operations[this->begin_operation_id]);
        begin_operation->sent_count++;
        if (begin_operation->sent_count == begin_operation->successors.size())
        {
            begin_operation->status = OperationStatus::transporting;
            this->machines[this->dummy_machine_id]->status = MachineStatus::lack_of_material;
        }
    }
    this->moving_AGVs.push(id);
}

void Graph::act_transport(AGVId id, OperationId _target_operation, MachineId _target_machine)
{
    auto AGV = this->AGVs[id];
    assert(AGV->status == AGVStatus::idle && AGV->loaded_item.has_value());
    assert(!AGV->transport_target.has_value());

    bool from_begin = holds_alternative<shared_ptr<GraphBegin>>(this->operations[AGV->loaded_item.value()]);

    if (_target_machine != this->dummy_machine_id)
    {
        auto target_machine = this->machines[_target_machine];
        assert(target_machine->status == MachineStatus::idle);
        target_machine->status = MachineStatus::lack_of_material;

        auto target_operation = get<shared_ptr<Operation>>(this->operations[_target_operation]);
        assert(target_operation->status == OperationStatus::waiting_machine);
        assert(target_operation->predecessor == AGV->loaded_item.value());
        assert(target_operation->machine_type == target_machine->type);
        target_operation->status = OperationStatus::waiting_material;

        target_operation->processing_machine = target_machine->id;
        target_machine->working_operation = target_operation->id;
    }
    AGV->target = _target_machine;
    AGV->transport_target = _target_operation;
    AGV->status = AGVStatus::transporting;
    AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][_target_machine] / AGV->speed;

    this->moving_AGVs.push(id);
}

shared_ptr<Graph> Graph::act(Action action)
{
    auto ret = this->copy();
    switch (action.type)
    {
    case ActionType::move:
        ret->act_move(action.act_AGV, action.target_machine);
        break;

    case ActionType::pick:
        ret->act_pick(action.act_AGV, action.target_machine);
        break;

    case ActionType::transport:
        ret->act_transport(action.act_AGV, action.target_operation.value(), action.target_machine);
        break;

    default:
        unreachable();
    }
    return ret;
}

void Graph::wait_operation()
{
    auto operation = get<shared_ptr<Operation>>(this->operations[this->processing_operations.top()]);
    this->processing_operations.pop();
    assert(operation->status == OperationStatus::processing);
    this->timestamp = operation->finish_timestamp;
    operation->status = OperationStatus::need_transport;
    auto machine = this->machines[operation->processing_machine.value()];
    assert(machine->status == MachineStatus::working);
    machine->status = MachineStatus::holding_product;
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
        AGV->position = AGV->target;
        break;
    }
    case AGVStatus::picking:
    {

        if (AGV->target != this->dummy_machine_id)
        {
            auto target_machine = this->machines[AGV->target];
            assert(target_machine->status == MachineStatus::holding_product);
            assert(target_machine->working_operation.has_value());

            auto transport_operation = get<shared_ptr<Operation>>(this->operations[target_machine->working_operation.value()]);
            assert(transport_operation->status == OperationStatus::waiting_transport);

            if (transport_operation->successor != this->end_operation_id)
            {
                auto target_operation = get<shared_ptr<Operation>>(this->operations[transport_operation->successor]);
                target_operation->status = OperationStatus::waiting_machine;
            }

            target_machine->status = MachineStatus::idle;
            target_machine->working_operation = nullopt;

            transport_operation->status = OperationStatus::transporting;

            AGV->loaded_item = transport_operation->id;
        }
        else
        {
            AGV->loaded_item = this->begin_operation_id;
        }

        AGV->status = AGVStatus::idle;
        AGV->position = AGV->target;

        break;
    }

    case AGVStatus::transporting:
    {
        assert(AGV->loaded_item.has_value() && AGV->transport_target.has_value());
        if (AGV->loaded_item.value() != this->begin_operation_id)
        {
            auto transport_operation = get<shared_ptr<Operation>>(this->operations[AGV->loaded_item.value()]);
            assert(transport_operation->status == OperationStatus::transporting);
            transport_operation->status = OperationStatus::finished;
        }
        else
        {
            auto begin_operation = get<shared_ptr<GraphBegin>>(this->operations[this->begin_operation_id]);
            begin_operation->arrived_count++;
            if (begin_operation->arrived_count == begin_operation->successors.size())
            {
                begin_operation->status = OperationStatus::finished;
            }
        }

        if (AGV->target != this->dummy_machine_id)
        {
            auto target_machine = this->machines[AGV->target];
            assert(target_machine->status == MachineStatus::lack_of_material);
            target_machine->status = MachineStatus::working;

            auto target_operation = get<shared_ptr<Operation>>(this->operations[AGV->transport_target.value()]);
            assert(target_operation->status == OperationStatus::waiting_material);
            assert(target_operation->machine_type == target_machine->type);
            target_operation->status = OperationStatus::processing;
            target_operation->finish_timestamp = this->timestamp + target_operation->process_time;
            this->processing_operations.push(target_operation->id);
        }
        else
        {
            auto end_operation = get<shared_ptr<GraphEnd>>(this->operations[this->end_operation_id]);
            end_operation->recive_cound++;
            if (end_operation->recive_cound == end_operation->predecessors.size())
            {
                end_operation->status = OperationStatus::finished;
                this->machines[this->dummy_machine_id]->status = MachineStatus::idle;
            }
        }

        AGV->status = AGVStatus::idle;
        AGV->position = AGV->target;
        AGV->transport_target = nullopt;
        AGV->loaded_item = nullopt;

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
        nearest_operation_finish_time = get<shared_ptr<Operation>>(this->operations[this->processing_operations.top()])->finish_timestamp;
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