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

#include "graph.h"
#include "utils.h"

using namespace std;
namespace py = pybind11;
namespace ph = placeholders;

TaskBase::TaskBase(TaskId id, MachineType mt) : id(id),
                                                machine_type(mt),
                                                status(TaskStatus::blocked)
{
}

JobBegin::JobBegin(TaskId id, MachineType mt) : TaskBase(id, mt),
                                                sent_count(0),
                                                arrived_count(0)
{
}

string JobBegin::repr()
{
    stringstream ss;
    ss << format("<begin (id: {}, total: {}, sent: {}, status: {})\n",
                 this->id, this->successors.size(), this->sent_count, enum_string(this->status));
    ss << add_indent(format("s: {}", id_set_string(this->successors)));
    ss << "\n>";
    return ss.str();
}

JobEnd::JobEnd(TaskId id, MachineType mt) : TaskBase(id, mt),
                                            recive_cound(0)
{
}

string JobEnd::repr()
{
    stringstream ss;
    ss << format("<end (id: {}, total: {}, recived: {}, status: {})\n",
                 this->id, this->predecessors.size(), this->recive_cound, enum_string(this->status));
    ss << add_indent(format("p: {}", id_set_string(this->predecessors)));
    ss << "\n>";
    return ss.str();
}

Task::Task(TaskId id, MachineType mt, double pt) : TaskBase(id, mt),
                                                   process_time(pt),
                                                   finish_timestamp(0),
                                                   processing_machine(nullopt)
{
}

string Task::repr()
{
    stringstream ss;
    if (this->status != TaskStatus::processing)
    {
        ss << format("<task (id: {}, type: {}, process_time: {:.2f}, status: {})\n",
                     this->id, this->machine_type, this->process_time, enum_string(this->status));
    }
    else
    {
        ss << format("<task (id: {}, type: {}, process_time: {:.2f}, status: {}, finish_timestamp: {:.2f})\n",
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
                                                 working_task(nullopt)
{
}

string Machine::repr()
{
    return format("<machine (id: {}, type: {}, status: {})>", this->id, this->type, enum_string(this->status));
}

AGV::AGV(AGVId id, double speed, MachineId init_pos) : id(id),
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
    string item_str = this->loaded_item.has_value() ? to_string(this->loaded_item.value()) : "none";

    switch (this->status)
    {
    case AGVStatus::idle:
        return format("<AGV (speed: {:.2f}, status: {}, position: {}, loaded_item: {})>",
                      this->speed, enum_string(this->status), this->position, item_str);
    default:
        return format("<AGV (speed: {:.2f}, status: {}, from: {}, to: {}, finish_timestamp: {:.2f}, loaded_item: {})>",
                      this->speed, enum_string(this->status), this->position, this->target, this->finish_timestamp, item_str);
    }
}

Action::Action(ActionType type, AGVId agv, MachineId m, optional<TaskId> t = nullopt) : type(type),
                                                                                        act_AGV(agv),
                                                                                        target_machine(m),
                                                                                        target_task(t)
{
}

string Action::repr()
{
    return format("<Action (type: {}, AGV_id: {}, target_machine: {}, target_task: {})>",
                  enum_string(this->type), this->act_AGV, this->target_machine, o2s(this->target_task));
}

Job::Job() : timestamp(0),
             processing_tasks(ProcessingTaskQueue(bind(&Job::task_time_compare, this, ph::_1, ph::_2))),
             moving_AGVs(MovingAGVQueue(bind(&Job::AGV_time_compare, this, ph::_1, ph::_2))),
             available_actions(nullopt)
{
    this->tasks[this->begin_task_id] = make_shared<JobBegin>(this->begin_task_id, this->dummy_machine_type);
    this->tasks[this->end_task_id] = make_shared<JobEnd>(this->end_task_id, this->dummy_machine_type);

    this->machines[this->dummy_machine_id] = make_shared<Machine>(this->dummy_machine_id, this->dummy_machine_type);

    this->next_task_id = 1;
    this->next_machine_id = 1;
    this->next_AGV_id = 0;
}

Job::Job(const Job &other) : processing_tasks(ProcessingTaskQueue(bind(&Job::task_time_compare, this, ph::_1, ph::_2))),
                             moving_AGVs(MovingAGVQueue(bind(&Job::AGV_time_compare, this, ph::_1, ph::_2))),
                             available_actions(nullopt)
{
    for (auto [id, ptr] : other.tasks)
    {
        visit([id, this]<typename T>(shared_ptr<T> p)
              { this->tasks[id] = make_shared<T>(*p); }, ptr);
    }
    this->next_task_id = other.next_task_id;

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

    this->processing_tasks.set_data(other.processing_tasks.get_data());
    this->moving_AGVs.set_data(other.moving_AGVs.get_data());
}

TaskId Job::add_task(
    MachineType type,
    double process_time,
    optional<TaskId> predecessor,
    optional<TaskId> successor)
{
    assert(!(predecessor.has_value() && successor.has_value()));
    auto node = make_shared<Task>(this->next_task_id++, type, process_time);
    this->tasks[node->id] = node;
    if (predecessor.has_value())
    {
        node->predecessor = predecessor.value();
        auto p_node = get<shared_ptr<Task>>(this->tasks[predecessor.value()]);
        node->successor = p_node->successor;
        p_node->successor = node->id;
        if (node->successor == this->end_task_id)
        {
            auto &&job_end_node = get<shared_ptr<JobEnd>>(this->tasks[this->end_task_id]);
            job_end_node->predecessors.erase(p_node->id);
            job_end_node->predecessors.emplace(node->id);
        }
    }
    else if (successor.has_value())
    {
        node->successor = successor.value();
        auto s_node = get<shared_ptr<Task>>(this->tasks[successor.value()]);
        node->predecessor = s_node->predecessor;
        s_node->predecessor = node->id;
        if (node->predecessor == this->begin_task_id)
        {
            auto &&job_begin_node = get<shared_ptr<JobBegin>>(this->tasks[this->begin_task_id]);
            job_begin_node->successors.erase(s_node->id);
            job_begin_node->successors.emplace(node->id);
        }
    }
    else
    {
        node->predecessor = this->begin_task_id;
        node->successor = this->end_task_id;
        get<shared_ptr<JobBegin>>(this->tasks[this->begin_task_id])->successors.emplace(node->id);
        get<shared_ptr<JobEnd>>(this->tasks[this->end_task_id])->predecessors.emplace(node->id);
    }

    return node->id;
}

void Job::remove_task(TaskId id)
{
    assert(id != this->begin_task_id && id != this->end_task_id);

    auto node_ptr = get<shared_ptr<Task>>(this->tasks[id]);
    if (holds_alternative<shared_ptr<JobBegin>>(this->tasks[node_ptr->predecessor]))
    {
        get<shared_ptr<JobBegin>>(this->tasks[node_ptr->predecessor])->successors.erase(id);
    }
    else
    {
        get<shared_ptr<Task>>(this->tasks[node_ptr->predecessor])->successor = node_ptr->successor;
    }

    if (holds_alternative<shared_ptr<JobEnd>>(this->tasks[node_ptr->successor]))
    {
        get<shared_ptr<JobEnd>>(this->tasks[node_ptr->successor])->predecessors.erase(id);
    }
    else
    {
        get<shared_ptr<Task>>(this->tasks[node_ptr->successor])->predecessor = node_ptr->predecessor;
    }
}

bool Job::contains(TaskId id)
{
    return this->tasks.contains(id);
}

Job::TaskNodePtr Job::get_task(TaskId id)
{
    return this->tasks[id];
}

MachineId Job::add_machine(MachineType type)
{
    auto node = make_shared<Machine>(this->next_machine_id++, type);
    this->machines[node->id] = node;
    return node->id;
}

shared_ptr<Machine> Job::get_machine(MachineId id)
{
    return this->machines[id];
}

AGVId Job::add_AGV(double speed, MachineId init_pos)
{
    auto new_AGV = make_shared<AGV>(this->next_AGV_id++, speed, init_pos);
    this->AGVs[new_AGV->id] = new_AGV;

    return new_AGV->id;
}

shared_ptr<AGV> Job::get_AGV(AGVId id)
{
    return this->AGVs[id];
}

double Job::get_timestamp()
{
    return this->timestamp;
}

void Job::set_timestamp(double value)
{
    this->timestamp = value;
}

string Job::repr()
{
    stringstream ss;
    ss << format("<job (timestamp: {:.2f})", this->timestamp);
    ss << format("\n task: {}\n", this->tasks.size());
    for (auto &&[id, task] : this->tasks)
    {
        ss << add_indent(to_task_base_ptr(task)->repr()) << '\n';
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
        ss << format("{:^5}", id);
    }
    for (auto &&[from, _] : this->machines)
    {
        ss << format("\n{:^5}", from);
        for (auto &&[to, _] : this->machines)
        {
            ss << format("{:^5.2f}", this->distances[from][to]);
        }
    }
    ss << "\n>";
    return ss.str();
}

void Job::set_distance(MachineId from, MachineId to, double distance)
{
    this->distances.try_emplace(from, map<MachineId, double>{}).first->second[to] = distance;
}

void Job::set_distance(map<MachineId, map<MachineId, double>> &other)
{
    this->distances = other;
}

void Job::set_rand_distance(double min_distance, double max_distance)
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

double Job::get_travel_time(MachineId from, MachineId to, AGVId agv)
{
    return this->distances[from][to] / this->AGVs[agv]->speed;
}

shared_ptr<Job> Job::rand_generate(
    vector<size_t> task_counts,
    size_t machine_count,
    size_t AGV_count,
    size_t machine_type_count,
    double min_transport_time,
    double max_transport_time,
    double min_max_speed_ratio,
    double min_process_time,
    double max_process_time)
{
    if (machine_count < machine_type_count)
    {
        throw py::value_error();
    }
    auto ret = make_shared<Job>();

    mt19937 engine(random_device{}());

    MachineType min_machine_type = Job::dummy_machine_type + 1;
    MachineType max_machine_type = Job::dummy_machine_type + machine_type_count;
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

    double max_speed = 1, min_speed = min_max_speed_ratio;
    double max_distance = max_transport_time * min_speed;
    double min_distance = min_transport_time * max_speed;
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

    uniform_real_distribution<double> process_time_dist(min_process_time, max_process_time);
    discrete_distribution<size_t> prev_count_dist({3, 5, 3, 1});
    discrete_distribution<size_t> repeat_count_dist({5, 3, 1});
    for (size_t task_count : task_counts)
    {
        TaskId prev_id;
        for (size_t i = 0; i < task_count; i++)
        {
            prev_id = ret->add_task(machine_type_dist(engine), process_time_dist(engine), i == 0 ? nullopt : optional{prev_id}, nullopt);
        }
    }

    return ret;
}

void Job::init_task_status()
{
    auto begin_task = get<shared_ptr<JobBegin>>(this->tasks[this->begin_task_id]);
    begin_task->status = TaskStatus::need_transport;
    this->machines[this->dummy_machine_id]->status = MachineStatus::holding_product;
    for (TaskId id : begin_task->successors)
    {
        auto task = get<shared_ptr<Task>>(this->tasks[id]);
        if (task->status == TaskStatus::blocked)
        {
            task->status = TaskStatus::waiting_machine;
        }
    }
    get<shared_ptr<JobEnd>>(this->tasks[this->end_task_id])->status = TaskStatus::blocked;
}

shared_ptr<Job> Job::copy()
{
    return make_shared<Job>(*this);
}

RepeatedTuple<double, Job::task_feature_size> Job::get_task_feature(shared_ptr<Task> task)
{
    double status_is_blocked = task->status == TaskStatus::blocked ? 1.0 : 0.0;
    double status_is_waiting_machine = task->status == TaskStatus::waiting_machine ? 1.0 : 0.0;
    double status_is_waiting_material = task->status == TaskStatus::waiting_material ? 1.0 : 0.0;
    double status_is_processing = task->status == TaskStatus::processing ? 1.0 : 0.0;
    double status_is_need_transport = task->status == TaskStatus::need_transport ? 1.0 : 0.0;
    double status_is_waiting_transport = task->status == TaskStatus::waiting_transport ? 1.0 : 0.0;
    double status_is_transporting = task->status == TaskStatus::transporting ? 1.0 : 0.0;
    double status_is_finished = task->status == TaskStatus::finished ? 1.0 : 0.0;

    double total_process_time = task->process_time;
    double rest_process_time = status_is_processing ? (task->finish_timestamp - this->timestamp) : 0.0;

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

RepeatedTuple<double, Job::machine_feature_size> Job::get_machine_feature(shared_ptr<Machine> machine)
{
    // TODO
    return {0, 0, 0, 0, 0};
}
RepeatedTuple<double, Job::AGV_feature_size> Job::get_AGV_feature(shared_ptr<AGV> AGV)
{
    // TODO
    return {0, 0, 0, 0, 0};
}

shared_ptr<JobFeatures> Job::features()
{
    auto ret = make_shared<JobFeatures>();

    vector<shared_ptr<Task>> normal_tasks;
    for (auto [_, ptr] : this->tasks)
    {
        if (holds_alternative<shared_ptr<Task>>(ptr))
        {
            normal_tasks.emplace_back(get<shared_ptr<Task>>(ptr));
        }
    }

    ret->task_features.resize(normal_tasks.size());
    auto tasks_with_idx = views::enumerate(normal_tasks);
    map<TaskId, size_t> task_id_idx_mapper;
    ::std::for_each(execution::par_unseq, tasks_with_idx.begin(), tasks_with_idx.end(),
                    [ret, this, &task_id_idx_mapper](tuple<ptrdiff_t, shared_ptr<Task>> i_task)
                    {
                        auto [i, task] = i_task;
                        ret->task_features[i] = this->get_task_feature(task);
                        task_id_idx_mapper[task->id] = static_cast<size_t>(i);
                    });

    ret->machine_features.resize(this->machines.size());
    auto machine_with_idx = views::enumerate(this->machines | views::transform([](auto kv)
                                                                               { return kv.second; }));
    map<MachineId, size_t> machine_id_idx_mapper;
    map<MachineType, vector<double>> machine_type_idx_mask;
    ::std::for_each(execution::par_unseq, machine_with_idx.begin(), machine_with_idx.end(),
                    [ret,
                     this,
                     &machine_id_idx_mapper,
                     &machine_type_idx_mask](tuple<ptrdiff_t, shared_ptr<Machine>> i_machine)
                    {
                        auto [i, machine] = i_machine;
                        ret->machine_features[i] = Job::get_machine_feature(machine);
                        machine_id_idx_mapper[machine->id] = static_cast<size_t>(i);
                        auto [iter, _] = machine_type_idx_mask.try_emplace(machine->type, vector<double>(this->machines.size(), 0));
                        iter->second[i] = 1;
                    });

    ret->task_relations.resize(normal_tasks.size());
    ret->processable_machine_mask.resize(normal_tasks.size());
    ::std::for_each(execution::par_unseq, tasks_with_idx.begin(), tasks_with_idx.end(),
                    [ret,
                     this,
                     &task_id_idx_mapper,
                     &machine_type_idx_mask](tuple<ptrdiff_t, shared_ptr<Task>> i_task)
                    {
                        auto [i, task] = i_task;
                        optional<size_t> p = task->predecessor == this->begin_task_id
                                                 ? nullopt
                                                 : optional{task_id_idx_mapper[task->predecessor]};
                        optional<size_t> s = task->successor == this->end_task_id
                                                 ? nullopt
                                                 : optional{task_id_idx_mapper[task->successor]};
                        ret->task_relations[i] = {p, s};
                        ret->processable_machine_mask[i] = machine_type_idx_mask[task->machine_type];
                    });

    ret->AGV_features.resize(this->AGVs.size());
    auto AGV_with_idx = views::enumerate(this->AGVs | views::transform([](auto kv)
                                                                       { return kv.second; }));
    ::std::for_each(execution::par_unseq, AGV_with_idx.begin(), AGV_with_idx.end(),
                    [ret,
                     this](tuple<ptrdiff_t, shared_ptr<AGV>> i_AGV)
                    {
                        auto [i, AGV] = i_AGV;
                        ret->AGV_features[i] = Job::get_AGV_feature(AGV);
                    });

    return ret;
}

bool Job::finished()
{
    return get<shared_ptr<JobEnd>>(this->tasks[this->end_task_id])->status == TaskStatus::finished;
}

vector<Action> Job::get_available_actions()
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
            if (id == this->dummy_machine_id || to_task_base_ptr(this->tasks[machine->working_task.value()])->status == TaskStatus::need_transport)
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
            visit([&]<typename T>(shared_ptr<T> task)
                  {
                    if constexpr (is_same_v<T, JobBegin>)
                    {
                        for(TaskId sid : task->successors)
                        {
                            auto s = get<shared_ptr<Task>>(this->tasks[sid]);
                            if(s->status == TaskStatus::waiting_machine)
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
                    else if constexpr (is_same_v<T, Task>)
                    {
                        if(task->successor == this->end_task_id)
                        {
                            ret.emplace_back(ActionType::transport, AGV_id, this->dummy_machine_id, task->successor);
                        }
                        else
                        {
                            for(auto machine : idle_machines)
                            {
                                if(machine->type == to_task_base_ptr(this->tasks[task->successor])->machine_type)
                                {
                                    ret.emplace_back(ActionType::transport, AGV_id, machine->id, task->successor);
                                }
                            }
                        }
                    }
                    else
                    {
                        unreachable();
                    } }, this->tasks[AGV->loaded_item.value()]);
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

bool Job::task_time_compare(TaskId a, TaskId b)
{
    return get<shared_ptr<Task>>(this->tasks[a])->finish_timestamp > get<shared_ptr<Task>>(this->tasks[b])->finish_timestamp;
}

bool Job::AGV_time_compare(AGVId a, AGVId b)
{
    return this->AGVs[a]->finish_timestamp > this->AGVs[b]->finish_timestamp;
}

void Job::act_move(AGVId id, MachineId target)
{
    auto AGV = this->AGVs[id];
    assert(AGV->status == AGVStatus::idle);
    AGV->target = target;
    AGV->status = AGVStatus::moving;
    AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;
    this->moving_AGVs.push(id);
}

void Job::act_pick(AGVId id, MachineId target)
{
    auto AGV = this->AGVs[id];
    assert(AGV->status == AGVStatus::idle && !AGV->loaded_item.has_value());

    if (target != this->dummy_machine_id)
    {
        auto target_machine = this->machines[target];
        assert(target_machine->status == MachineStatus::holding_product);
        assert(target_machine->working_task.has_value());

        auto transport_task = get<shared_ptr<Task>>(this->tasks[target_machine->working_task.value()]);
        assert(transport_task->status == TaskStatus::need_transport);

        AGV->target = target;
        AGV->status = AGVStatus::picking;
        AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;

        transport_task->status = TaskStatus::waiting_transport;
    }
    else
    {
        AGV->target = target;
        AGV->status = AGVStatus::picking;
        AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;

        auto begin_task = get<shared_ptr<JobBegin>>(this->tasks[this->begin_task_id]);
        begin_task->sent_count++;
        if (begin_task->sent_count == begin_task->successors.size())
        {
            begin_task->status = TaskStatus::transporting;
            this->machines[this->dummy_machine_id]->status = MachineStatus::lack_of_material;
        }
    }
    this->moving_AGVs.push(id);
}

void Job::act_transport(AGVId id, TaskId _target_task, MachineId _target_machine)
{
    auto AGV = this->AGVs[id];
    assert(AGV->status == AGVStatus::idle && AGV->loaded_item.has_value());
    assert(!AGV->transport_target.has_value());

    bool from_begin = holds_alternative<shared_ptr<JobBegin>>(this->tasks[AGV->loaded_item.value()]);

    if (_target_machine != this->dummy_machine_id)
    {
        auto target_machine = this->machines[_target_machine];
        assert(target_machine->status == MachineStatus::idle);
        target_machine->status = MachineStatus::lack_of_material;

        auto target_task = get<shared_ptr<Task>>(this->tasks[_target_task]);
        assert(target_task->status == TaskStatus::waiting_machine);
        assert(target_task->predecessor == AGV->loaded_item.value());
        assert(target_task->machine_type == target_machine->type);
        target_task->status = TaskStatus::waiting_material;

        target_task->processing_machine = target_machine->id;
        target_machine->working_task = target_task->id;
    }
    AGV->target = _target_machine;
    AGV->transport_target = _target_task;
    AGV->status = AGVStatus::transporting;
    AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][_target_machine] / AGV->speed;

    this->moving_AGVs.push(id);
}

shared_ptr<Job> Job::act(Action action)
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
        ret->act_transport(action.act_AGV, action.target_task.value(), action.target_machine);
        break;

    default:
        unreachable();
    }
    return ret;
}

void Job::wait_task()
{
    auto task = get<shared_ptr<Task>>(this->tasks[this->processing_tasks.top()]);
    this->processing_tasks.pop();
    assert(task->status == TaskStatus::processing);
    this->timestamp = task->finish_timestamp;
    task->status = TaskStatus::need_transport;
    auto machine = this->machines[task->processing_machine.value()];
    assert(machine->status == MachineStatus::working);
    machine->status = MachineStatus::holding_product;
}

void Job::wait_AGV()
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
            assert(target_machine->working_task.has_value());

            auto transport_task = get<shared_ptr<Task>>(this->tasks[target_machine->working_task.value()]);
            assert(transport_task->status == TaskStatus::waiting_transport);

            if (transport_task->successor != this->end_task_id)
            {
                auto target_task = get<shared_ptr<Task>>(this->tasks[transport_task->successor]);
                target_task->status = TaskStatus::waiting_machine;
            }

            target_machine->status = MachineStatus::idle;
            target_machine->working_task = nullopt;

            transport_task->status = TaskStatus::transporting;

            AGV->loaded_item = transport_task->id;
        }
        else
        {
            AGV->loaded_item = this->begin_task_id;
        }

        AGV->status = AGVStatus::idle;
        AGV->position = AGV->target;

        break;
    }

    case AGVStatus::transporting:
    {
        assert(AGV->loaded_item.has_value() && AGV->transport_target.has_value());
        if (AGV->loaded_item.value() != this->begin_task_id)
        {
            auto transport_task = get<shared_ptr<Task>>(this->tasks[AGV->loaded_item.value()]);
            assert(transport_task->status == TaskStatus::transporting);
            transport_task->status = TaskStatus::finished;
        }
        else
        {
            auto begin_task = get<shared_ptr<JobBegin>>(this->tasks[this->begin_task_id]);
            begin_task->arrived_count++;
            if (begin_task->arrived_count == begin_task->successors.size())
            {
                begin_task->status = TaskStatus::finished;
            }
        }

        if (AGV->target != this->dummy_machine_id)
        {
            auto target_machine = this->machines[AGV->target];
            assert(target_machine->status == MachineStatus::lack_of_material);
            target_machine->status = MachineStatus::working;

            auto target_task = get<shared_ptr<Task>>(this->tasks[AGV->transport_target.value()]);
            assert(target_task->status == TaskStatus::waiting_material);
            assert(target_task->machine_type == target_machine->type);
            target_task->status = TaskStatus::processing;
            target_task->finish_timestamp = this->timestamp + target_task->process_time;
            this->processing_tasks.push(target_task->id);
        }
        else
        {
            auto end_task = get<shared_ptr<JobEnd>>(this->tasks[this->end_task_id]);
            end_task->recive_cound++;
            if (end_task->recive_cound == end_task->predecessors.size())
            {
                end_task->status = TaskStatus::finished;
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

void Job::_wait(double delta_time)
{
    double nearest_task_finish_time = DBL_MAX;
    double nearest_AGV_finish_time = DBL_MAX;
    if (!this->processing_tasks.empty())
    {
        nearest_task_finish_time = get<shared_ptr<Task>>(this->tasks[this->processing_tasks.top()])->finish_timestamp;
    }

    if (!this->moving_AGVs.empty())
    {
        nearest_AGV_finish_time = this->AGVs[this->moving_AGVs.top()]->finish_timestamp;
    }

    if (nearest_task_finish_time <= this->timestamp + delta_time)
    {
        wait_task();
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

shared_ptr<Job> Job::wait(double delta_time)
{
    auto ret = this->copy();
    ret->_wait(delta_time);
    return ret;
}