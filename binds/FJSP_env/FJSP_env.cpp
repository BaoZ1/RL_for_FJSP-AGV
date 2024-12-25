#include <concepts>
#include <execution>
#include <format>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <variant>

#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FJSP_env.h"
#include "utils.h"

using namespace std;
namespace py = pybind11;
namespace ph = placeholders;
using namespace py::literals;

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

string Operation::repr() const
{
    stringstream ss;
    if (this->status != OperationStatus::processing)
    {
        ss << format("<operation (id: {}, type: {}, process_time: {:.2f}, status: {})\n", this->id, this->machine_type, this->process_time, enum_string(this->status));
    }
    else
    {
        ss << format("<operation (id: {}, type: {}, process_time: {:.2f}, status: {}, finish_timestamp: {:.2f})\n", this->id, this->machine_type, this->process_time, enum_string(this->status), this->finish_timestamp);
    }

    ss << add_indent(format("p: {}", id_set_string(this->predecessors)));
    ss << add_indent(format("s: {}", id_set_string(this->successors)));
    ss << "\n>";
    return ss.str();
}

Machine::Machine(MachineId id, MachineType tp, Position pos) : id(id),
                                                               type(tp),
                                                               pos(pos),
                                                               status(MachineStatus::idle),
                                                               working_operation(nullopt)
{
}

string Machine::repr() const
{
    return format("<machine (id: {}, type: {}, status: {}, working: {}, waiting: {})>", this->id, this->type, enum_string(this->status), o2s(this->working_operation), v2s(this->waiting_operations));
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

string AGV::repr() const
{
    switch (this->status)
    {
    case AGVStatus::idle:
        return format("<AGV (id: {}, speed: {:.2f}, status: {}, position: {}, loaded_item: {})>", this->id, this->speed, enum_string(this->status), this->position, o2s(this->loaded_item));
    case AGVStatus::picking:
        return format("<AGV (id: {}, speed: {:.2f}, status: {}, from: {}, to: {}, finish_timestamp: {:.2f}, target_item: {})>", this->id, this->speed, enum_string(this->status), this->position, this->target_machine, this->finish_timestamp, o2s(this->target_item));
    default:
        return format("<AGV (id: {}, speed: {:.2f}, status: {}, from: {}, to: {}, finish_timestamp: {:.2f}, loaded_item: {})>", this->id, this->speed, enum_string(this->status), this->position, this->target_machine, this->finish_timestamp, o2s(this->loaded_item));
    }
}

Action::Action(ActionType type) : type(type), act_AGV(nullopt), target_machine(nullopt), target_product(nullopt)
{
    assert(type == ActionType::wait);
}

Action::Action(ActionType type, AGVId AGV, MachineId machine) : type(type), act_AGV(AGV), target_machine(machine), target_product(nullopt)
{
    assert(type == ActionType::move || type == ActionType::transport);
}

Action::Action(ActionType type, AGVId AGV, MachineId machine, Product product) : type(type), act_AGV(AGV), target_machine(machine), target_product(product)
{
    assert(type == ActionType::pick);
}

string Action::repr() const
{
    switch (this->type)
    {
    case ActionType::move:
    case ActionType::transport:
        return format("<Action (type: {}, AGV_id: {}, target_machine: {})>", enum_string(this->type), o2s(this->act_AGV), o2s(this->target_machine));
    case ActionType::pick:
        return format("<Action (type: {}, AGV_id: {}, target_machine: {}, target_product: {})>", enum_string(this->type), o2s(this->act_AGV), o2s(this->target_machine), o2s(this->target_product));
    case ActionType::wait:
        return format("<Action (type: {})>", enum_string(this->type));
    };
    assert(false);
    return format("<Action (type: ERROR)>");
}

Graph::Graph() : inited(false),
                 timestamp(0),
                 processing_operations(ProcessingOperationQueue(bind(&Graph::operation_time_compare, this, ph::_1, ph::_2))),
                 moving_AGVs(MovingAGVQueue(bind(&Graph::AGV_time_compare, this, ph::_1, ph::_2)))
{
    this->operations[this->begin_operation_id] = make_shared<Operation>(this->begin_operation_id, this->dummy_machine_type, 0.0f);
    this->operations[this->end_operation_id] = make_shared<Operation>(this->end_operation_id, this->dummy_machine_type, 0.0f);
    this->add_relation(this->begin_operation_id, this->end_operation_id);

    this->machines[this->dummy_machine_id] = make_shared<Machine>(this->dummy_machine_id, this->dummy_machine_type, Position{0, 0});
    auto &&dummy_path_row = this->paths.emplace(this->dummy_machine_id, map<MachineId, vector<MachineId>>()).first->second;
    dummy_path_row.emplace(this->dummy_machine_id, vector<MachineId>{this->dummy_machine_id});

    this->next_operation_id = 1;
    this->next_machine_id = 1;
    this->next_AGV_id = 0;
}

Graph::Graph(const Graph &other) : processing_operations(ProcessingOperationQueue(bind(&Graph::operation_time_compare, this, ph::_1, ph::_2))),
                                   moving_AGVs(MovingAGVQueue(bind(&Graph::AGV_time_compare, this, ph::_1, ph::_2)))
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

    this->direct_paths = other.direct_paths;
    this->paths = other.paths;
    this->distances = other.distances;

    this->inited = other.inited;
    this->timestamp = other.timestamp;

    this->processing_operations.set_data(other.processing_operations.get_data());
    this->moving_AGVs.set_data(other.moving_AGVs.get_data());
}

py::dict Graph::get_state() const
{
    vector<py::dict> operation_dict;
    for (auto operation : this->operations | views::values)
    {
        operation_dict.emplace_back(
            "id"_a = operation->id,
            "status"_a = static_cast<int>(operation->status),
            "machine_type"_a = operation->machine_type,
            "process_time"_a = operation->process_time,
            "processing_machine"_a = operation->processing_machine,
            "finish_timestamp"_a = operation->finish_timestamp,
            "predecessors"_a = vector<OperationId>(from_range, operation->predecessors),
            "arrived_preds"_a = vector<OperationId>(from_range, operation->arrived_preds),
            "successors"_a = vector<OperationId>(from_range, operation->successors),
            "sent_succs"_a = vector<OperationId>(from_range, operation->sent_succs)
        );
    }
    vector<py::dict> machine_dict;
    for (auto machine : this->machines | views::values)
    {
        auto materials = ranges::to<vector<py::dict>>(
            machine->materials | views::transform([](Product p)
                                                  { return py::dict("from"_a = p.from, "to"_a = p.to); })
        );
        auto products = ranges::to<vector<py::dict>>(
            machine->products | views::transform([](Product p)
                                                 { return py::dict("from"_a = p.from, "to"_a = p.to); })
        );
        machine_dict.emplace_back(
            "id"_a = machine->id,
            "type"_a = machine->type,
            "pos"_a = py::dict("x"_a = machine->pos.x, "y"_a = machine->pos.y),
            "status"_a = static_cast<int>(machine->status),
            "working_operation"_a = machine->working_operation,
            "waiting_operations"_a = machine->waiting_operations,
            "materials"_a = materials,
            "products"_a = products
        );
    }
    vector<py::dict> AGV_dict;
    for (auto AGV : this->AGVs | views::values)
    {
        auto loaded_item = AGV->loaded_item.transform(
            [](Product p)
            { return py::dict("from"_a = p.from, "to"_a = p.to); }
        );
        auto target_item = AGV->target_item.transform(
            [](Product p)
            { return py::dict("from"_a = p.from, "to"_a = p.to); }
        );

        AGV_dict.emplace_back(
            "id"_a = AGV->id,
            "status"_a = static_cast<int>(AGV->status),
            "speed"_a = AGV->speed,
            "position"_a = AGV->position,
            "target_machine"_a = AGV->target_machine,
            "loaded_item"_a = loaded_item,
            "target_item"_a = target_item,
            "finish_timestamp"_a = AGV->finish_timestamp
        );
    }
    return py::dict(
        "inited"_a = this->inited,
        "timestamp"_a = this->timestamp,
        "operations"_a = operation_dict,
        "machines"_a = machine_dict,
        "AGVs"_a = AGV_dict,
        "direct_paths"_a = this->direct_paths,
        // "paths"_a = this->paths,
        // "distances"_a = this->distances,
        "next_operation_id"_a = this->next_operation_id,
        "next_machine_id"_a = this->next_machine_id,
        "next_AGV_id"_a = this->next_AGV_id
    );
}

shared_ptr<Graph> Graph::from_state(py::dict d)
{
    auto res = make_shared<Graph>();
    res->operations.clear();
    res->machines.clear();
    res->AGVs.clear();

    res->inited = d["inited"].cast<bool>();
    res->timestamp = d["timestamp"].cast<float>();

    for (auto operation_dict : d["operations"].cast<vector<py::dict>>())
    {
        auto new_operation = make_shared<Operation>(
            operation_dict["id"].cast<OperationId>(),
            operation_dict["machine_type"].cast<MachineType>(),
            operation_dict["process_time"].cast<float>()
        );
        new_operation->status = static_cast<OperationStatus>(operation_dict["status"].cast<int>());
        new_operation->processing_machine = operation_dict["processing_machine"].cast<optional<MachineId>>();
        new_operation->finish_timestamp = operation_dict["finish_timestamp"].cast<float>();
        new_operation->predecessors = set(from_range, operation_dict["predecessors"].cast<vector<OperationId>>());
        new_operation->arrived_preds = set(from_range, operation_dict["arrived_preds"].cast<vector<OperationId>>());
        new_operation->successors = set(from_range, operation_dict["successors"].cast<vector<OperationId>>());
        new_operation->sent_succs = set(from_range, operation_dict["sent_succs"].cast<vector<OperationId>>());
        res->operations[new_operation->id] = new_operation;
        if (new_operation->status == OperationStatus::processing)
        {
            res->processing_operations.push(new_operation->id);
        }
    }

    for (auto machine_dict : d["machines"].cast<vector<py::dict>>())
    {
        auto pos_dict = machine_dict["pos"].cast<py::dict>();
        auto new_machine = make_shared<Machine>(
            machine_dict["id"].cast<MachineId>(),
            machine_dict["type"].cast<MachineType>(),
            Position{pos_dict["x"].cast<float>(), pos_dict["y"].cast<float>()}
        );
        new_machine->status = static_cast<MachineStatus>(machine_dict["status"].cast<int>());
        new_machine->working_operation = machine_dict["working_operation"].cast<optional<OperationId>>();
        new_machine->waiting_operations = machine_dict["waiting_operations"].cast<vector<OperationId>>();
        for (auto material_dict : machine_dict["materials"].cast<vector<py::dict>>())
        {
            new_machine->materials.emplace(
                material_dict["from"].cast<OperationId>(),
                material_dict["to"].cast<OperationId>()
            );
        }
        for (auto product_dict : machine_dict["products"].cast<vector<py::dict>>())
        {
            new_machine->products.emplace(
                product_dict["from"].cast<OperationId>(),
                product_dict["to"].cast<OperationId>()
            );
        }
        res->machines[new_machine->id] = new_machine;
    }

    for (auto AGV_dict : d["AGVs"].cast<vector<py::dict>>())
    {
        auto new_AGV = make_shared<AGV>(
            AGV_dict["id"].cast<AGVId>(),
            AGV_dict["speed"].cast<float>(),
            AGV_dict["position"].cast<MachineId>()
        );
        new_AGV->status = static_cast<AGVStatus>(AGV_dict["status"].cast<int>());
        new_AGV->target_machine = AGV_dict["target_machine"].cast<MachineId>();
        new_AGV->finish_timestamp = AGV_dict["finish_timestamp"].cast<float>();
        new_AGV->loaded_item = AGV_dict["loaded_item"].cast<optional<py::dict>>().transform(
            [](py::dict d)
            { return Product{d["from"].cast<OperationId>(), d["to"].cast<OperationId>()}; }
        );
        new_AGV->target_item = AGV_dict["target_item"].cast<optional<py::dict>>().transform(
            [](py::dict d)
            { return Product{d["from"].cast<OperationId>(), d["to"].cast<OperationId>()}; }
        );
        res->AGVs[new_AGV->id] = new_AGV;
        if (new_AGV->status != AGVStatus::idle)
        {
            res->moving_AGVs.push(new_AGV->id);
        }
    }

    res->direct_paths.clear();
    res->direct_paths = d["direct_paths"].cast<set<tuple<MachineId, MachineId>>>();

    // res->paths.clear();
    // for (auto [k1, v1] : d["paths"].cast<py::dict>())
    // {
    //     map<MachineId, vector<MachineId>> line;
    //     for (auto [k2, v2] : v1.cast<py::dict>())
    //     {
    //         line[k2.cast<int>()] = v2.cast<vector<MachineId>>();
    //     }
    //     res->paths[k1.cast<int>()] = line;
    // }

    // res->distances.clear();
    // for (auto [k1, v1] : d["distances"].cast<py::dict>())
    // {
    //     map<MachineId, float> line;
    //     for (auto [k2, v2] : v1.cast<py::dict>())
    //     {
    //         line[stoi(k2.cast<string>())] = v2.cast<float>();
    //     }
    //     res->distances[stoi(k1.cast<string>())] = line;
    // }

    res->next_operation_id = d["next_operation_id"].cast<OperationId>();
    res->next_machine_id = d["next_machine_id"].cast<MachineId>();
    res->next_AGV_id = d["next_AGV_id"].cast<AGVId>();

    return res;
}

vector<OperationId> Graph::get_operations_id() const
{
    return this->operations | views::keys | ranges::to<vector<OperationId>>();
}

vector<MachineId> Graph::get_machines_id() const
{
    return this->machines | views::keys | ranges::to<vector<OperationId>>();
}

vector<AGVId> Graph::get_AGVs_id() const
{
    return this->AGVs | views::keys | ranges::to<vector<OperationId>>();
}

OperationId Graph::add_operation(MachineType type, float process_time)
{
    return this->insert_operation(type, process_time, nullopt, nullopt);
}

void Graph::add_relation(OperationId from, OperationId to)
{
    assert(this->contains(from));
    assert(this->contains(to));
    auto p_node = this->operations.at(from);
    auto s_node = this->operations.at(to);
    if (p_node->successors.size() == 1 && p_node->successors.contains(this->end_operation_id))
    {
        p_node->successors.clear();
        this->operations.at(this->end_operation_id)->predecessors.erase(p_node->id);
    }
    if (s_node->predecessors.size() == 1 && s_node->predecessors.contains(this->begin_operation_id))
    {
        s_node->predecessors.clear();
        this->operations.at(this->begin_operation_id)->successors.erase(s_node->id);
    }
    p_node->successors.emplace(s_node->id);
    s_node->predecessors.emplace(p_node->id);
}

void Graph::remove_relation(OperationId from, OperationId to)
{
    assert(this->contains(from));
    assert(this->contains(to));
    auto p_node = this->operations.at(from);
    auto s_node = this->operations.at(to);
    p_node->successors.erase(s_node->id);
    s_node->predecessors.erase(p_node->id);
    if (p_node->successors.empty())
    {
        p_node->successors.emplace(this->end_operation_id);
        this->operations.at(this->end_operation_id)->predecessors.emplace(p_node->id);
    }
    if (s_node->predecessors.empty())
    {
        s_node->predecessors.emplace(this->begin_operation_id);
        this->operations.at(this->begin_operation_id)->successors.emplace(s_node->id);
    }
}

OperationId Graph::insert_operation(
    MachineType type,
    float process_time,
    optional<OperationId> predecessor,
    optional<OperationId> successor
)
{
    assert(!(predecessor.has_value() && successor.has_value()));
    if (this->operations.size() == 2)
    {
        assert(this->operations.at(this->begin_operation_id)->successors.size() == 1);
        assert(this->operations.at(this->begin_operation_id)->successors.contains(this->end_operation_id));
        assert(this->operations.at(this->end_operation_id)->predecessors.size() == 1);
        assert(this->operations.at(this->end_operation_id)->predecessors.contains(this->begin_operation_id));

        this->operations.at(this->begin_operation_id)->successors.clear();
        this->operations.at(this->end_operation_id)->predecessors.clear();
    }
    auto node = make_shared<Operation>(this->next_operation_id++, type, process_time);
    this->operations[node->id] = node;
    if (predecessor.has_value())
    {
        node->predecessors = {predecessor.value()};
        auto p_node = this->operations.at(predecessor.value());
        node->successors = p_node->successors;
        p_node->successors = {node->id};
        for (auto s_id : node->successors)
        {
            auto s_node = this->operations.at(s_id);
            s_node->predecessors.erase(p_node->id);
            s_node->predecessors.emplace(node->id);
        }
    }
    else if (successor.has_value())
    {
        node->successors = {successor.value()};
        auto s_node = this->operations.at(successor.value());
        node->predecessors = s_node->predecessors;
        s_node->predecessors = {node->id};
        for (auto p_id : node->predecessors)
        {
            auto p_node = this->operations.at(p_id);
            p_node->successors.erase(s_node->id);
            p_node->successors.emplace(node->id);
        }
    }
    else
    {
        node->predecessors = {this->begin_operation_id};
        node->successors = {this->end_operation_id};
        this->operations.at(this->begin_operation_id)->successors.emplace(node->id);
        this->operations.at(this->end_operation_id)->predecessors.emplace(node->id);
    }

    return node->id;
}

void Graph::remove_operation(OperationId id)
{
    assert(id != this->begin_operation_id && id != this->end_operation_id);

    auto node = this->operations.at(id);
    this->operations.erase(id);
    auto ps = node->predecessors | views::transform([this](OperationId p_id)
                                                    { return this->operations.at(p_id); });
    auto ss = node->successors | views::transform([this](OperationId s_id)
                                                  { return this->operations.at(s_id); });
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
            if (!(is_begin && is_end) && p_node->successors.empty())
            {
                p_node->successors.emplace(s_node->id);
                s_node->predecessors.emplace(p_node->id);
            }
        }
    }
    if (this->operations.size() == 2)
    {
        assert(this->operations.contains(this->begin_operation_id));
        assert(this->operations.contains(this->end_operation_id));
        assert(this->operations.at(this->begin_operation_id)->successors.empty());
        assert(this->operations.at(this->end_operation_id)->predecessors.empty());

        this->operations.at(this->begin_operation_id)->successors = {this->end_operation_id};
        this->operations.at(this->end_operation_id)->predecessors = {this->begin_operation_id};
    }
}

bool Graph::contains(OperationId id) const
{
    return this->operations.contains(id);
}

shared_ptr<Operation> Graph::get_operation(OperationId id) const
{
    return this->operations.at(id);
}

MachineId Graph::add_machine(MachineType type, Position pos)
{
    auto node = make_shared<Machine>(this->next_machine_id++, type, pos);
    this->machines[node->id] = node;
    for (auto prev : this->paths | views::values)
    {
        prev.emplace(node->id, vector<MachineId>{});
    }
    auto &new_row = this->paths[node->id];
    for (auto id : this->machines | views::keys)
    {
        new_row.emplace(id, vector<MachineId>{});
    }
    new_row.at(node->id).emplace_back(node->id);
    return node->id;
}

shared_ptr<Machine> Graph::get_machine(MachineId id) const
{
    return this->machines.at(id);
}

AGVId Graph::add_AGV(float speed, MachineId init_pos)
{
    auto new_AGV = make_shared<AGV>(this->next_AGV_id++, speed, init_pos);
    this->AGVs[new_AGV->id] = new_AGV;

    return new_AGV->id;
}

shared_ptr<AGV> Graph::get_AGV(AGVId id) const
{
    return this->AGVs.at(id);
}

float Graph::get_timestamp() const
{
    return this->timestamp;
}

void Graph::set_timestamp(float value)
{
    this->timestamp = value;
}

string Graph::repr() const
{
    stringstream ss;
    ss << format("<job (inited: {}, timestamp: {:.2f})", this->inited, this->timestamp);

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
    if (this->distances.empty())
    {
        ss << add_indent("not calculated");
    }
    else
    {
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
                ss << format("{:^6.2f}", this->distances.at(from).at(to));
            }
        }
    }
    ss << "\n>";
    return ss.str();
}

void Graph::add_path(MachineId a, MachineId b)
{
    this->distances.clear();
    this->direct_paths.emplace(a, b);
    this->direct_paths.emplace(b, a);
}

void Graph::remove_path(MachineId a, MachineId b)
{
    this->distances.clear();
    this->direct_paths.erase({a, b});
    this->direct_paths.erase({b, a});
}

void Graph::calc_distance()
{
    this->distances.clear();
    for (auto f_id : this->machines | views::keys)
    {
        for (auto t_id : this->machines | views::keys)
        {
            if (f_id == t_id)
            {
                this->distances[f_id][t_id] = 0;
            }
            else if (this->direct_paths.contains({f_id, t_id}))
            {
                this->distances[f_id][t_id] = this->machines.at(f_id)->pos.distance(this->machines.at(t_id)->pos);
            }
            else
            {
                this->distances[f_id][t_id] = numeric_limits<float>::infinity() / 3;
            }
        }
    }

    this->paths.clear();
    for (auto [f_id, t_id] : this->direct_paths)
    {
        this->paths[f_id][t_id].emplace_back(t_id);
    }

    for (auto m_id : this->machines | views::keys)
    {
        for (auto f_id : this->machines | views::keys)
        {
            for (auto t_id : this->machines | views::keys)
            {
                if (this->paths[f_id][m_id].empty() || this->paths[m_id][t_id].empty())
                {
                    continue;
                }
                float dist = this->distances[f_id][m_id] + this->distances[m_id][t_id];
                if (this->distances[f_id][t_id] > dist)
                {
                    this->distances[f_id][t_id] = dist;
                    auto &&path = this->paths[f_id][t_id];
                    path.clear();
                    path.append_range(this->paths[f_id][m_id]);
                    path.append_range(this->paths[m_id][t_id]);
                }
            }
        }
    }
}

float Graph::get_travel_time(MachineId from, MachineId to, AGVId agv) const
{
    return this->distances.at(from).at(to) / this->AGVs.at(agv)->speed;
}

shared_ptr<Graph> Graph::rand_generate(GenerateParam param)
{
    if (param.machine_count < param.machine_type_count)
    {
        throw py::value_error();
    }
    auto ret = make_shared<Graph>();

    mt19937 engine(random_device{}());

    MachineType min_machine_type = Graph::dummy_machine_type + 1;
    MachineType max_machine_type = Graph::dummy_machine_type + param.machine_type_count;
    uniform_int_distribution<MachineType> machine_type_dist(min_machine_type, max_machine_type);
    uniform_real_distribution<float> machine_pos_x_dist(5, 50);
    uniform_real_distribution<float> machine_pos_y_dist(5, 50);
    vector<MachineId> machines{Graph::dummy_machine_id};

    for (MachineType t = min_machine_type; t <= max_machine_type; t++)
    {
        Position pos{machine_pos_x_dist(engine), machine_pos_y_dist(engine)};
        machines.emplace_back(ret->add_machine(t, pos));
    }
    for (size_t i = param.machine_type_count; i < param.machine_count; i++)
    {
        Position pos{machine_pos_x_dist(engine), machine_pos_y_dist(engine)};
        machines.emplace_back(ret->add_machine(machine_type_dist(engine), pos));
    }

    uniform_int_distribution<size_t> machine_idx_dist(0, machines.size() - 1);
    UnionFind<MachineId> uf(machines);
    while (uf.count() != 1)
    {
        MachineId a = machines.at(machine_idx_dist(engine));
        MachineId b = machines.at(machine_idx_dist(engine));

        if (ret->direct_paths.contains({a, b}))
        {
            continue;
        }

        ret->add_path(a, b);
        uf.unite(a, b);
    }
    ret->calc_distance();
    float min_dist = numeric_limits<float>::max();
    float max_dist = numeric_limits<float>::min();

    for (auto [f_id, tos] : ret->distances)
    {
        for (auto [t_id, dist] : tos)
        {
            if (t_id == f_id)
            {
                continue;
            }
            min_dist = min(min_dist, dist);
            max_dist = max(max_dist, dist);
        }
    }
    float min_speed = max_dist / param.max_transport_time;
    uniform_real_distribution AGV_speed_dist(min_speed, min_speed / param.min_max_speed_ratio);
    for (size_t i = 0; i < param.AGV_count; i++)
    {
        ret->add_AGV(AGV_speed_dist(engine), Graph::dummy_machine_id);
    }

    uniform_real_distribution<float> process_time_dist(param.min_process_time, param.max_process_time);
    discrete_distribution<size_t> prev_count_dist({3, 6, 2, 1});
    set<OperationId> exist_ids;
    for (size_t i = 0; i < param.operation_count; i++)
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

shared_ptr<Graph> Graph::copy() const
{
    return make_shared<Graph>(*this);
}

shared_ptr<Graph> Graph::reset() const
{
    auto ret = this->copy();
    if (!ret->inited)
    {
        return ret;
    }
    for (auto operation : ret->operations | views::values)
    {
        operation->status = OperationStatus::blocked;
        operation->processing_machine = nullopt;
        operation->finish_timestamp = 0;
    }
    for (auto machine : ret->machines | views::values)
    {
        machine->status = MachineStatus::idle;
        machine->working_operation = nullopt;
        machine->waiting_operations.clear();
    }
    for (auto AGV : ret->AGVs | views::values)
    {
        AGV->status = AGVStatus::idle;
        AGV->target_machine = AGV->position;
        AGV->finish_timestamp = 0;
        AGV->loaded_item = nullopt;
        AGV->target_item = nullopt;
    }
    ret->processing_operations.clear();
    ret->moving_AGVs.clear();
    ret->timestamp = 0;
    ret->inited = false;
    return ret;
}

shared_ptr<Graph> Graph::init() const
{
    assert(!this->inited);
    auto ret = this->copy();
    auto dummy_machine = ret->machines.at(ret->dummy_machine_id);
    auto begin_operation = ret->operations.at(ret->begin_operation_id);
    begin_operation->status = OperationStatus::finished;
    for (OperationId id : begin_operation->successors)
    {
        ret->operations.at(id)->status = OperationStatus::unscheduled;
        dummy_machine->products.emplace(Product{ret->begin_operation_id, id});
    }
    ret->operations.at(ret->end_operation_id)->status = OperationStatus::blocked;
    ret->calc_distance();
    ret->timestamp = 0;
    ret->inited = true;
    return ret;
}

RepeatedTuple<float, Graph::operation_feature_size> Graph::get_operation_feature(shared_ptr<Operation> operation) const
{
    float status_is_blocked = operation->status == OperationStatus::blocked ? 1 : 0;
    float status_is_unscheduled = operation->status == OperationStatus::unscheduled ? 1 : 0;
    float status_is_waiting = operation->status == OperationStatus::waiting ? 1 : 0;
    float status_is_processing = operation->status == OperationStatus::processing ? 1 : 0;
    float status_is_finished = operation->status == OperationStatus::finished ? 1 : 0;

    float total_process_time = operation->process_time;
    float rest_process_time = status_is_processing ? (operation->finish_timestamp - this->timestamp) : 0;

    float total_material = operation->status <= OperationStatus::waiting ? operation->predecessors.size() : 0;
    float rest_material = operation->status <= OperationStatus::waiting ? (total_material - operation->arrived_preds.size()) : 0;

    float total_product = operation->status >= OperationStatus::processing ? operation->successors.size() : 0;
    float rest_pruduct = operation->status >= OperationStatus::processing ? total_product - operation->sent_succs.size() : 0;

    return {
        status_is_blocked,
        status_is_unscheduled,
        status_is_waiting,
        status_is_processing,
        status_is_finished,
        total_process_time,
        rest_process_time,
        total_product,
        rest_pruduct
    };
}

RepeatedTuple<float, Graph::machine_feature_size> Graph::get_machine_feature(shared_ptr<Machine> machine) const
{
    float status_is_idle = machine->status == MachineStatus::idle;
    float status_is_waiting_material = machine->status == MachineStatus::waiting_material;
    float status_is_working = machine->status == MachineStatus::working;

    float rest_process_time = machine->status == MachineStatus::working
                                  ? (this->operations.at(machine->working_operation.value())
                                         ->finish_timestamp
                                     - this->timestamp)
                                  : 0;

    float same_type_count = ranges::count_if(
        this->machines | views::values,
        [this, machine](auto other)
        { return other->type == machine->type; }
    );
    auto processable_operations = this->operations
                                  | views::values
                                  | views::filter(
                                      [this, machine](auto op)
                                      { return op->machine_type == machine->type; }
                                  );
    float processable_operation_count = ranges::distance(processable_operations);
    float rest_processable_operation_count = ranges::count_if(
        processable_operations,
        [](auto op)
        { return op->status < OperationStatus::processing; }
    );

    return {
        status_is_idle,
        status_is_waiting_material,
        status_is_working,
        rest_process_time,
        same_type_count,
        processable_operation_count,
        rest_processable_operation_count
    };
}
RepeatedTuple<float, Graph::AGV_feature_size> Graph::get_AGV_feature(shared_ptr<AGV> AGV) const
{
    float status_is_idle = AGV->status == AGVStatus::idle;
    float status_is_moving = AGV->status == AGVStatus::moving;
    float status_is_picking = AGV->status == AGVStatus::picking;
    float status_is_transporting = AGV->status == AGVStatus::transporting;

    float rest_act_time = AGV->status == AGVStatus::idle ? 0 : AGV->finish_timestamp - this->timestamp;

    return {status_is_idle, status_is_moving, status_is_picking, status_is_transporting, rest_act_time};
}

tuple<shared_ptr<GraphFeature>, shared_ptr<IdIdxMapper>> Graph::features() const
{
    assert(this->inited);

    auto feature = make_shared<GraphFeature>();
    auto mapper = make_shared<IdIdxMapper>();

    float total_task = 0;
    float finished_task = 0;

    feature->operation_features.resize(this->operations.size());
    auto operations_with_idx = this->operations | views::values | views::enumerate;
    for (auto [i, operation] : operations_with_idx)
    {
        feature->operation_features[i] = this->get_operation_feature(operation);
        mapper->operation[operation->id] = static_cast<size_t>(i);

        total_task += operation->predecessors.size();
        finished_task += operation->arrived_preds.size();
    }
    feature->global_feature = make_tuple(this->timestamp, total_task, finished_task / total_task);
    static_assert(tuple_size<decltype(feature->global_feature)>() == Graph::global_feature_size);

    feature->machine_features.resize(this->machines.size());
    auto machine_with_idx = this->machines | views::values | views::enumerate;
    map<MachineType, vector<float>> machine_type_idx_mask;
    for (auto [i, machine] : machine_with_idx)
    {
        feature->machine_features[i] = Graph::get_machine_feature(machine);
        mapper->machine[machine->id] = static_cast<size_t>(i);

        if (machine->working_operation.has_value())
        {
            auto operation = this->operations.at(machine->working_operation.value());
            feature->processing.emplace_back(
                i,
                mapper->operation.at(operation->id),
                operation->finish_timestamp - this->timestamp
            );
        }
        for (auto operation_id : machine->waiting_operations)
        {
            auto operation = this->operations.at(operation_id);
            feature->waiting.emplace_back(
                i,
                mapper->operation.at(operation_id),
                operation->predecessors.size(),
                operation->arrived_preds.size()
            );
        }
    }

    for (auto [i, operation] : operations_with_idx)
    {
        for (auto p_id : operation->predecessors)
        {
            feature->predecessor_idx.emplace_back(i, mapper->operation.at(p_id));
        }
        for (auto s_id : operation->successors)
        {
            feature->successor_idx.emplace_back(i, mapper->operation.at(s_id));
        }
        for (auto [m_id, m] : machine_with_idx)
        {
            if (m->type == operation->machine_type)
            {
                feature->processable_idx.emplace_back(m_id, i);
            }
        }
    }

    feature->AGV_features.resize(this->AGVs.size());
    auto AGV_with_idx = this->AGVs | views::values | views::enumerate;
    for (auto [i, AGV] : AGV_with_idx)
    {
        feature->AGV_features[i] = Graph::get_AGV_feature(AGV);
        mapper->AGV[AGV->id] = static_cast<size_t>(i);
        if (AGV->status == AGVStatus::idle)
        {
            feature->AGV_position.emplace_back(i, mapper->machine.at(AGV->position));
        }
        else
        {
            auto t = AGV->finish_timestamp - this->timestamp;
            feature->AGV_target.emplace_back(i, mapper->machine.at(AGV->target_machine), t);
        }
        if (AGV->loaded_item.has_value())
        {
            auto p = AGV->loaded_item.value();
            feature->AGV_loaded.emplace_back(i, mapper->operation.at(p.from), mapper->operation.at(p.to));
        }
    }

    return make_tuple(feature, mapper);
}

tuple<size_t, size_t> Graph::progress() const
{
    size_t total = 0;
    size_t finished = 0;

    for (auto operation : this->operations | views::values)
    {
        total += operation->predecessors.size();
        finished += operation->arrived_preds.size();
    }
    return {finished, total};
}

bool Graph::finished() const
{
    return this->operations.at(this->end_operation_id)->status == OperationStatus::finished;
}

float Graph::finish_time_lower_bound() const
{
    map<MachineType, float> remain_process_time;
    float remain_transport_distance = 0;

    map<pair<MachineType, MachineType>, float> min_type_distance;
    map<MachineType, size_t> type_count;
    for (auto [fid, from] : this->machines)
    {
        for (auto [tid, to] : this->machines)
        {
            if (fid == tid)
            {
                continue;
            }
            float distance = this->distances.at(fid).at(tid);
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
                                                                         { return this->operations.at(id); }))
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

    float total_speed = 0;
    for (auto AGV : this->AGVs | views::values)
    {
        total_speed += AGV->speed;
    }

    return this->timestamp + remain_transport_distance / total_speed;
}

vector<Action> Graph::get_available_actions()
{
    assert(this->inited);

    vector<Action> ret;
    if (!this->processing_operations.empty() || !this->moving_AGVs.empty())
    {
        ret.emplace_back(ActionType::wait);
    }

    vector<pair<shared_ptr<Machine>, Product>> pickable_products;
    for (auto [id, machine] : this->machines)
    {
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
            auto operation = this->operations.at(AGV->loaded_item->to);
            if (operation->processing_machine.has_value())
            {
                auto machine = this->machines.at(operation->processing_machine.value());
                assert(operation->status == OperationStatus::blocked || operation->status == OperationStatus::waiting);
                assert(operation->machine_type == machine->type);
                ret.emplace_back(ActionType::transport, AGV_id, machine->id);
            }
            else
            {
                for (auto machine : this->machines | views::values)
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

    return ret;
}

bool Graph::operation_time_compare(OperationId a, OperationId b)
{
    return this->operations.at(a)->finish_timestamp > this->operations.at(b)->finish_timestamp;
}

bool Graph::AGV_time_compare(AGVId a, AGVId b)
{
    return this->AGVs.at(a)->finish_timestamp > this->AGVs.at(b)->finish_timestamp;
}

void Graph::act_move(AGVId id, MachineId target)
{
    auto AGV = this->AGVs.at(id);
    assert(AGV->status == AGVStatus::idle);
    AGV->target_machine = target;
    AGV->status = AGVStatus::moving;
    AGV->finish_timestamp = this->timestamp + this->distances.at(AGV->position).at(target) / AGV->speed;
    this->moving_AGVs.push(id);
}

void Graph::act_pick(AGVId id, MachineId target, Product product)
{
    auto AGV = this->AGVs.at(id);
    assert(AGV->status == AGVStatus::idle && !AGV->loaded_item.has_value());

    auto target_machine = this->machines.at(target);
    assert(target_machine->products.contains(product));

    auto transport_operation = this->operations.at(product.to);

    AGV->target_machine = target;
    AGV->target_item = product;
    AGV->status = AGVStatus::picking;
    AGV->finish_timestamp = this->timestamp + this->distances.at(AGV->position).at(target) / AGV->speed;
    this->moving_AGVs.push(id);
}

void Graph::act_transport(AGVId id, MachineId target)
{
    auto AGV = this->AGVs.at(id);
    assert(AGV->status == AGVStatus::idle && AGV->loaded_item.has_value());
    auto product = AGV->loaded_item.value();
    auto target_machine = this->machines.at(target);
    auto target_operation = this->operations.at(product.to);
    assert(target_operation->predecessors.contains(AGV->loaded_item->from));
    assert(target_operation->machine_type == target_machine->type);

    if (!ranges::contains(target_machine->waiting_operations, target_operation->id))
    {
        assert(!target_operation->processing_machine.has_value());
        assert(target_operation->status <= OperationStatus::unscheduled);

        if (!target_machine->waiting_operations.empty())
        {
            if (target_machine->working_operation.has_value())
            {
                assert(target_machine->status == MachineStatus::working);
            }
            else
            {
                assert(target_machine->status == MachineStatus::waiting_material);
            }
        }
        else
        {
            if (target_machine->status == MachineStatus::idle)
            {
                target_machine->status = MachineStatus::waiting_material;
            }
        }

        target_machine->waiting_operations.emplace_back(target_operation->id);
        target_operation->processing_machine = target_machine->id;
        if (target_operation->status == OperationStatus::unscheduled)
        {
            target_operation->status = OperationStatus::waiting;
        }
    }
    else
    {
        assert(target_operation->status <= OperationStatus::waiting);
    }

    AGV->target_machine = target;
    AGV->status = AGVStatus::transporting;
    AGV->finish_timestamp = this->timestamp + this->distances.at(AGV->position).at(target) / AGV->speed;

    this->moving_AGVs.push(id);
}

void Graph::act_wait()
{
    float nearest_operation_finish_time = numeric_limits<float>::max();
    float nearest_AGV_finish_time = numeric_limits<float>::max();

    if (!this->processing_operations.empty())
    {
        nearest_operation_finish_time = this->operations.at(this->processing_operations.top())->finish_timestamp;
    }

    if (!this->moving_AGVs.empty())
    {
        nearest_AGV_finish_time = this->AGVs.at(this->moving_AGVs.top())->finish_timestamp;
    }

    if (nearest_operation_finish_time != numeric_limits<float>::max() && nearest_operation_finish_time <= nearest_AGV_finish_time)
    {
        wait_operation();
    }
    else if (nearest_AGV_finish_time != numeric_limits<float>::max())
    {
        wait_AGV();
    }
    else
    {
        assert(false);
    }
}

shared_ptr<Graph> Graph::act(Action action) const
{
    assert(this->inited);

    auto ret = this->copy();

    switch (action.type)
    {
    case ActionType::move:
        ret->act_move(action.act_AGV.value(), action.target_machine.value());
        break;

    case ActionType::pick:
        ret->act_pick(action.act_AGV.value(), action.target_machine.value(), action.target_product.value());
        break;

    case ActionType::transport:
        ret->act_transport(action.act_AGV.value(), action.target_machine.value());
        break;
    case ActionType::wait:
        ret->act_wait();
        break;
    default:
        assert(false);
    }
    return ret;
}

tuple<vector<shared_ptr<Graph>>, vector<float>> Graph::batch_step(const vector<shared_ptr<Graph>> &envs, const vector<shared_ptr<Action>> &actions)
{
    auto pairs = views::zip(vector(envs), vector<float>(envs.size(), 0), actions) | ranges::to<vector>();

    for_each(execution::par, pairs.begin(), pairs.end(), [](tuple<shared_ptr<Graph>, float, shared_ptr<Action>> &item)
             {
                 auto &[env, lb, action] = item;
                 env = env->act(*action);
                 lb = env->finish_time_lower_bound(); });
    return make_tuple(
        pairs | views::elements<0> | ranges::to<vector>(),
        pairs | views::elements<1> | ranges::to<vector>()
    );
}

void Graph::wait_operation()
{
    auto operation = this->operations.at(this->processing_operations.top());
    this->processing_operations.pop();
    assert(operation->status == OperationStatus::processing);
    assert(operation->processing_machine.has_value());
    this->timestamp = operation->finish_timestamp;
    operation->status = OperationStatus::finished;
    auto machine = this->machines.at(operation->processing_machine.value());
    assert(machine->status == MachineStatus::working);
    assert(machine->working_operation == operation->id);
    machine->status = MachineStatus::idle;
    machine->working_operation = nullopt;
    for (auto s_id : operation->successors)
    {
        machine->products.emplace(Product{operation->id, s_id});
        auto successor = this->operations.at(s_id);
        assert(successor->status == OperationStatus::blocked);
        if (ranges::all_of(successor->predecessors, [this](auto p_id)
                           { return this->operations.at(p_id)->status == OperationStatus::finished; }))
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

    if (!machine->waiting_operations.empty())
    {
        machine->status = MachineStatus::waiting_material;
        for (auto operation_id : machine->waiting_operations)
        {
            auto operation = this->operations.at(operation_id);
            assert(operation->status == OperationStatus::blocked || operation->status == OperationStatus::waiting);
            if (operation->status == OperationStatus::waiting
                && ranges::all_of(operation->predecessors, [machine, operation_id](auto id)
                                  { return machine->materials.contains(Product{id, operation_id}); }))
            {
                for (auto p_id : operation->predecessors)
                {
                    machine->materials.erase(Product{p_id, operation->id});
                }
                machine->working_operation = operation_id;
                machine->waiting_operations.erase(ranges::find(machine->waiting_operations, operation_id));
                machine->status = MachineStatus::working;
                operation->status = OperationStatus::processing;
                operation->finish_timestamp = this->timestamp + operation->process_time;
                this->processing_operations.push(operation->id);
                break;
            }
        }
    }
}

void Graph::wait_AGV()
{
    auto AGV = this->AGVs.at(this->moving_AGVs.top());
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
        auto target_machine = this->machines.at(AGV->target_machine);
        assert(target_machine->products.contains(AGV->target_item.value()));
        auto operation = this->operations.at(AGV->target_item->from);
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
        auto machine = this->machines.at(AGV->target_machine);
        assert(ranges::contains(machine->waiting_operations, AGV->loaded_item->to));
        assert(machine->status == MachineStatus::waiting_material || machine->status == MachineStatus::working);
        auto operation = this->operations.at(AGV->loaded_item->to);
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
            machine->working_operation = operation->id;
            machine->waiting_operations.erase(ranges::find(machine->waiting_operations, operation->id));

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
