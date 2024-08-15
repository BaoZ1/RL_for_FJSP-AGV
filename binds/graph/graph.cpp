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
#include <array>
#include <unordered_set>
#include <sstream>
#include <string>
#include <queue>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;
namespace ph = placeholders;

#ifdef NDEBUG
#define DEBUG_OUT(...) (void)0
#else
#define DEBUG_OUT(...) std::cout << format(__VA_ARGS__) << std::endl
#endif

using TaskId = size_t;
using MachineId = size_t;
using AGVId = size_t;
using MachineType = size_t;

enum class TaskStatus
{
    blocked,
    waiting,
    processing,
    need_transport,
    waiting_transport,
    transporting,
    finished
};

constexpr string enum_string(TaskStatus s)
{
    switch (s)
    {
    case TaskStatus::blocked:
        return string("blocked");
    case TaskStatus::waiting:
        return string("waiting");
    case TaskStatus::processing:
        return string("processing");
    case TaskStatus::need_transport:
        return string("need_transport");
    case TaskStatus::waiting_transport:
        return string("waiting_transport");
    case TaskStatus::transporting:
        return string("transporting");
    case TaskStatus::finished:
        return string("finished");
    default:
        throw pybind11::value_error();
    }
}

enum class MachineStatus
{
    idle,
    lack_of_material,
    working,
    holding_product,
};

constexpr string enum_string(MachineStatus s)
{
    switch (s)
    {
    case MachineStatus::idle:
        return string("idle");
    case MachineStatus::lack_of_material:
        return string("lack_of_material");
    case MachineStatus::working:
        return string("working");
    case MachineStatus::holding_product:
        return string("holding_product");
    default:
        throw pybind11::value_error();
    }
}

enum class AGVStatus
{
    idle,
    moving,
    picking,
    transporting
};

constexpr string enum_string(AGVStatus s)
{
    switch (s)
    {
    case AGVStatus::idle:
        return string("idle");
    case AGVStatus::moving:
        return string("moving");
    case AGVStatus::picking:
        return string("picking");
    case AGVStatus::transporting:
        return string("transporting");
    default:
        throw pybind11::value_error();
    }
}

string id_set_string(const unordered_set<TaskId> &s)
{
    stringstream ss;
    ss << "[";
    for (auto &&[i, pid] : s | views::enumerate)
    {
        ss << pid;
        if (i != s.size() - 1)
        {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

string add_indent(const string &s, size_t indent_count = 1)
{
    stringstream oss(s), iss;
    string line;
    string indent;
    indent.append(indent_count, ' ');
    while (getline(oss, line))
    {
        iss << indent << line << '\n';
    }

    string ret = iss.str();
    if (s[s.length() - 1] != '\n')
    {
        ret.pop_back();
    }
    return ret;
}

int comp2int(const strong_ordering &od)
{
    if (od == strong_ordering::less)
    {
        return -1;
    }
    else if (od == strong_ordering::greater)
    {
        return 1;
    }
    return 0;
}

template <typename T>
string o2s(optional<T> o, string on_empty = "null")
{
    if (o.has_value())
    {
        return to_string(o.value());
    }
    return on_empty;
}

struct TaskBase
{
    TaskBase(TaskId id, MachineType mt) : id(id), machine_type(mt) {}

    virtual string repr() = 0;

    TaskId id;

    MachineType machine_type;
};

template <typename... Types>
    requires(derived_from<Types, TaskBase> && ...)
shared_ptr<TaskBase> to_task_base_ptr(variant<shared_ptr<Types>...> wrapped)
{
    return visit([](auto &&a)
                 { return static_pointer_cast<TaskBase>(a); }, wrapped);
}

struct JobBegin : public TaskBase
{
    JobBegin(TaskId id, MachineType mt) : TaskBase(id, mt) {}

    string repr() override
    {
        stringstream ss;
        ss << format("<begin (id: {})\n", this->id);
        ss << add_indent(format("s: {}", id_set_string(this->successors)));
        ss << "\n>";
        return ss.str();
    }

    unordered_set<TaskId> successors;
};

struct JobEnd : public TaskBase
{
    JobEnd(TaskId id, MachineType mt) : TaskBase(id, mt) {}

    string repr() override
    {
        stringstream ss;
        ss << format("<end (id: {})\n", this->id);
        ss << add_indent(format("p: {}", id_set_string(this->predecessors)));
        ss << "\n>";
        return ss.str();
    }

    unordered_set<TaskId> predecessors;
};

struct Task : public TaskBase
{
    Task(TaskId id, MachineType mt, double pt) : TaskBase(id, mt),
                                                 status(TaskStatus::blocked),
                                                 process_time(pt),
                                                 finish_timestamp(0),
                                                 processing_machine(nullopt) {}

    string repr() override
    {
        stringstream ss;
        ss << format("<task (id: {}, type: {}, process_time: {:.2f}, status: {})\n",
                     this->id, this->machine_type, this->process_time, enum_string(this->status));
        ss << add_indent(format("p: {}", this->predecessor));
        ss << add_indent(format("s: {}", this->successor));
        ss << "\n>";
        return ss.str();
    }

    TaskStatus status;

    double process_time;
    double finish_timestamp;

    TaskId predecessor;
    TaskId successor;

    optional<MachineId> processing_machine;
};

struct Machine
{
    Machine(MachineId id, MachineType tp) : id(id),
                                            type(tp),
                                            status(MachineStatus::idle),
                                            working_task(nullopt) {}
    string repr()
    {
        return format("<machine (id: {}, type: {}, status: {})>", this->id, this->type, enum_string(this->status));
    }

    MachineId id;
    MachineType type;
    MachineStatus status;
    optional<TaskId> working_task;
};

struct AGV
{
    AGV(AGVId id, double speed, MachineId init_pos) : id(id),
                                                      speed(speed),
                                                      status(AGVStatus::idle),
                                                      position(init_pos),
                                                      target(0),
                                                      finish_timestamp(0),
                                                      loaded_item(nullopt) {}
    string repr()
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

    AGVId id;
    double speed;
    AGVStatus status;
    MachineId position, target;
    double finish_timestamp;
    optional<TaskId> loaded_item;
};

template <size_t N, typename T, typename... Ts>
struct TupleGenerator
{
    using type = TupleGenerator<N - 1, T, T, Ts...>::type;
};
template <typename T, typename... Ts>
struct TupleGenerator<0, T, Ts...>
{
    using type = tuple<Ts...>;
};
template <typename T, size_t N>
using RepeatedTuple = TupleGenerator<N, T>::type;

template <typename T, class Comp>
class copyable_priority_queue
{
public:
    copyable_priority_queue(Comp comp) : comp(comp) {}

    void push(T value)
    {
        data.push_back(value);
        push_heap(data.begin(), data.end(), comp);
    }

    void pop()
    {
        if (!data.empty())
        {
            pop_heap(data.begin(), data.end(), comp);
            data.pop_back();
        }
    }

    T top()
    {
        if (!data.empty())
        {
            return data.front();
        }
        throw runtime_error("Empty priority queue");
    }

    bool empty()
    {
        return this->data.empty();
    }

    void set_data(const vector<T> &other_data)
    {
        this->data = other_data;
        push_heap(data.begin(), data.end(), comp);
    }

    const vector<T> &get_data() const
    {
        return this->data;
    }

private:
    vector<T> data;
    Comp comp;
};

struct JobFeatures;

class Job
{
    using TaskNodePtr = variant<shared_ptr<JobBegin>, shared_ptr<JobEnd>, shared_ptr<Task>>;

    using ProcessingTaskQueue = copyable_priority_queue<TaskId, function<bool(TaskId, TaskId)>>;
    using MovingAGVQueue = copyable_priority_queue<AGVId, function<bool(AGVId, AGVId)>>;

public:
    static const TaskId begin_task_id = 0, end_task_id = 9999;
    static const MachineId dummy_machine_id = 0;
    static const MachineType dummy_machine_type = 0;

    static const size_t task_feature_size = 9;
    static const size_t machine_feature_size = 5;
    static const size_t AGV_feature_size = 5;

    Job() : timestamp(0),
            processing_tasks(ProcessingTaskQueue(bind(&Job::task_time_compare, this, ph::_1, ph::_2))),
            moving_AGVs(MovingAGVQueue(bind(&Job::AGV_time_compare, this, ph::_1, ph::_2)))
    {
        this->tasks[this->begin_task_id] = make_shared<JobBegin>(this->begin_task_id, this->dummy_machine_type);
        this->tasks[this->end_task_id] = make_shared<JobEnd>(this->end_task_id, this->dummy_machine_type);

        this->machines[this->dummy_machine_id] = make_shared<Machine>(this->dummy_machine_id, this->dummy_machine_type);

        this->next_task_id = 1;
        this->next_machine_id = 1;
        this->next_AGV_id = 0;
    }

    Job(const Job &other) : processing_tasks(ProcessingTaskQueue(bind(&Job::task_time_compare, this, ph::_1, ph::_2))),
                            moving_AGVs(MovingAGVQueue(bind(&Job::AGV_time_compare, this, ph::_1, ph::_2)))
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

    TaskId add_task(
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

    void remove_task(TaskId id)
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

    bool contains(TaskId id)
    {
        return this->tasks.contains(id);
    }

    TaskNodePtr get_task(TaskId id)
    {
        return this->tasks[id];
    }

    MachineId add_machine(MachineType type)
    {
        auto node = make_shared<Machine>(this->next_machine_id++, type);
        this->machines[node->id] = node;
        return node->id;
    }

    shared_ptr<Machine> get_machine(MachineId id)
    {
        return this->machines[id];
    }

    AGVId add_AGV(double speed, MachineId init_pos)
    {
        auto new_AGV = make_shared<AGV>(this->next_AGV_id++, speed, init_pos);
        this->AGVs[new_AGV->id] = new_AGV;

        return new_AGV->id;
    }

    shared_ptr<AGV> get_AGV(AGVId id)
    {
        return this->AGVs[id];
    }

    double get_timestamp()
    {
        return this->timestamp;
    }

    void set_timestamp(double value)
    {
        this->timestamp = value;
    }

    string repr()
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

    double get_travel_time(MachineId from, MachineId to, AGVId agv)
    {
        return this->distances[from][to] / this->AGVs[agv]->speed;
    }

    void set_distance(MachineId from, MachineId to, double distance)
    {
        this->distances.try_emplace(from, map<MachineId, double>{}).first->second[to] = distance;
    }

    void set_distance(map<MachineId, map<MachineId, double>> &other)
    {
        this->distances = other;
    }

    void set_rand_distance(double min_distance, double max_distance)
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

    static shared_ptr<Job> rand_generate(
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

    shared_ptr<Job> copy()
    {
        return make_shared<Job>(*this);
    }

    /**
     * - status == blocked (focus on predecessor)
     * - status == waiting (... transport)
     * - status == processing (... rest time)
     * - status == need_transport (... transport)
     * - status == waiting_transport (... ?)
     * - status == transporting (... successor)
     * - status == finished (... nothing, maybe)
     * - total process time
     * - status rest time (be 0 if not processing)
     */
    RepeatedTuple<double, task_feature_size> get_task_feature(shared_ptr<Task> task)
    {
        double status_is_blocked = task->status == TaskStatus::blocked ? 1.0 : 0.0;
        double status_is_waiting = task->status == TaskStatus::waiting ? 1.0 : 0.0;
        double status_is_processing = task->status == TaskStatus::processing ? 1.0 : 0.0;
        double status_is_need_transport = task->status == TaskStatus::need_transport ? 1.0 : 0.0;
        double status_is_waiting_transport = task->status == TaskStatus::waiting_transport ? 1.0 : 0.0;
        double status_is_transporting = task->status == TaskStatus::transporting ? 1.0 : 0.0;
        double status_is_finished = task->status == TaskStatus::finished ? 1.0 : 0.0;

        double total_process_time = task->process_time;
        double rest_process_time = status_is_processing ? (task->finish_timestamp - this->timestamp) : 0.0;

        return {
            status_is_blocked,
            status_is_waiting,
            status_is_processing,
            status_is_need_transport,
            status_is_waiting_transport,
            status_is_transporting,
            status_is_finished,
            total_process_time,
            rest_process_time,
        };
    }

    RepeatedTuple<double, machine_feature_size> get_machine_feature(shared_ptr<Machine> machine)
    {
        return {0, 0, 0, 0, 0};
    }

    RepeatedTuple<double, AGV_feature_size> get_AGV_feature(shared_ptr<AGV> AGV)
    {
        return {0, 0, 0, 0, 0};
    }

    shared_ptr<JobFeatures> features();

    void refresh_task_status()
    {
        // TODO
    }

    bool task_time_compare(TaskId a, TaskId b)
    {
        return get<shared_ptr<Task>>(this->tasks[a])->finish_timestamp > get<shared_ptr<Task>>(this->tasks[b])->finish_timestamp;
    }

    bool AGV_time_compare(AGVId a, AGVId b)
    {
        DEBUG_OUT("cmp {} : {:.2f} & ", a, this->AGVs[a]->finish_timestamp);
        DEBUG_OUT("& {} : {:.2f}", b, this->AGVs[b]->finish_timestamp);
        return this->AGVs[a]->finish_timestamp > this->AGVs[b]->finish_timestamp;
    }

    void _act_move(AGVId id, MachineId target)
    {
        auto AGV = this->AGVs[id];
        assert(AGV->status == AGVStatus::idle);
        AGV->target = target;
        AGV->status = AGVStatus::moving;
        AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][target] / AGV->speed;
        this->moving_AGVs.push(id);
    }

    shared_ptr<Job> act_move(AGVId id, MachineId target)
    {
        auto ret = this->copy();
        ret->_act_move(id, target);
        return ret;
    }

    void _act_pick(AGVId id, MachineId target)
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
        }
        DEBUG_OUT("push...");
        this->moving_AGVs.push(id);
        DEBUG_OUT("pushed!");
    }

    shared_ptr<Job> act_pick(AGVId id, MachineId target)
    {
        auto ret = this->copy();
        ret->_act_pick(id, target);
        return ret;
    }

    void _act_transport(AGVId id, TaskId _target_task, MachineId _target_machine)
    {
        auto AGV = this->AGVs[id];
        assert(AGV->status == AGVStatus::idle && AGV->loaded_item.has_value());

        bool from_begin = holds_alternative<shared_ptr<JobBegin>>(this->tasks[AGV->loaded_item.value()]);

        if (_target_machine != this->dummy_machine_id)
        {
            auto target_machine = this->machines[_target_machine];
            assert(target_machine->status == MachineStatus::idle);

            auto target_task = get<shared_ptr<Task>>(this->tasks[_target_task]);
            assert(target_task->status == TaskStatus::blocked);
            assert(target_task->predecessor == AGV->loaded_item.value());
            assert(target_task->machine_type == target_machine->type);

            AGV->target = _target_machine;
            AGV->status = AGVStatus::transporting;
            AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][_target_machine] / AGV->speed;

            target_task->status = TaskStatus::waiting;

            target_machine->status = MachineStatus::lack_of_material;
            target_machine->working_task = target_task->id;
        }
        else
        {
            AGV->target = _target_machine;
            AGV->status = AGVStatus::transporting;
            AGV->finish_timestamp = this->timestamp + this->distances[AGV->position][_target_machine] / AGV->speed;
        }
        this->moving_AGVs.push(id);
    }

    shared_ptr<Job> act_transport(AGVId id, TaskId _target_task, MachineId _target_machine)
    {
        auto ret = this->copy();
        ret->_act_transport(id, _target_task, _target_machine);
        return ret;
    }

    void _wait(double delta_time)
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

    shared_ptr<Job> wait(double delta_time)
    {
        auto ret = this->copy();
        ret->_wait(delta_time);
        return ret;
    }

    void wait_task()
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

    void wait_AGV()
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
            auto target_machine = this->machines[AGV->target];
            if (target_machine->id != this->dummy_machine_id)
            {
                assert(target_machine->status == MachineStatus::holding_product);
                assert(target_machine->working_task.has_value());
            }

            auto transport_task = get<shared_ptr<Task>>(this->tasks[target_machine->working_task.value()]);
            assert(transport_task->status == TaskStatus::need_transport);

            AGV->status = AGVStatus::idle;
            AGV->position = AGV->target;
            AGV->loaded_item = transport_task->id;

            target_machine->status = MachineStatus::idle;
            target_machine->working_task = nullopt;

            transport_task->status = TaskStatus::transporting;

            break;
        }

        case AGVStatus::transporting:
        {
            auto target_machine = this->machines[AGV->target];
            assert(target_machine->status == MachineStatus::lack_of_material);

            auto transport_task = get<shared_ptr<Task>>(this->tasks[AGV->loaded_item.value()]);
            assert(transport_task->status == TaskStatus::transporting);

            auto target_task = get<shared_ptr<Task>>(this->tasks[transport_task->successor]);
            assert(target_task->status == TaskStatus::waiting);
            assert(target_task->machine_type == target_machine->type);

            AGV->status = AGVStatus::idle;
            AGV->loaded_item = nullopt;

            target_machine->status = MachineStatus::working;

            transport_task->status = TaskStatus::finished;

            target_task->status = TaskStatus::processing;
            target_task->finish_timestamp = this->timestamp + target_task->process_time;

            break;
        }

        default:
            throw logic_error(format("wrong AGV status: {}", enum_string(AGV->status)));
        }
    }

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
};

struct JobFeatures
{
    vector<RepeatedTuple<double, Job::task_feature_size>> task_features;
    vector<RepeatedTuple<optional<size_t>, 2>> task_relations;
    vector<RepeatedTuple<double, Job::machine_feature_size>> machine_features;
    vector<vector<double>> processable_machine_mask;
    vector<RepeatedTuple<double, Job::AGV_feature_size>> AGV_features;
};

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

using namespace py::literals;

PYBIND11_MODULE(graph, m)
{
    py::enum_<TaskStatus>(m, "TaskStatus")
        .value("blocked", TaskStatus::blocked)
        .value("waiting", TaskStatus::waiting)
        .value("need_transport", TaskStatus::need_transport)
        .value("waiting_transport", TaskStatus::waiting_transport)
        .value("transporting", TaskStatus::transporting)
        .value("finished", TaskStatus::finished);
    py::enum_<MachineStatus>(m, "MachineStatus")
        .value("idle", MachineStatus::idle)
        .value("lack_of_material", MachineStatus::lack_of_material)
        .value("working", MachineStatus::working)
        .value("holding_product", MachineStatus::holding_product);

    py::class_<JobBegin, shared_ptr<JobBegin>>(m, "JobBegin")
        .def("__repr__", &JobBegin::repr);
    py::class_<JobEnd, shared_ptr<JobEnd>>(m, "JobEnd")
        .def("__repr__", &JobEnd::repr);
    py::class_<Task, shared_ptr<Task>>(m, "Task")
        .def("__repr__", &Task::repr);

    py::class_<Machine, shared_ptr<Machine>>(m, "Machine")
        .def("__repr__", &Machine::repr);

    py::class_<AGV, shared_ptr<AGV>>(m, "AGV")
        .def("__repr__", &AGV::repr);

    py::class_<Job, shared_ptr<Job>>(m, "Job")
        .def(py::init<>())
        .def("add_task", &Job::add_task, "machine_type"_a, "process_time"_a, "predecessors"_a = nullopt, "successors"_a = nullopt)
        .def("remove_task", &Job::remove_task, "task_id"_a)
        .def("get_task", &Job::get_task, "task_id"_a)
        .def("add_machine", &Job::add_machine, "machine_type"_a)
        .def("get_machine", &Job::get_machine, "machine_id"_a)
        .def("add_AGV", &Job::add_AGV, "speed"_a, "init_pos"_a)
        .def("get_AGV", &Job::get_AGV, "AGV_id"_a)
        .def("get_travel_time", &Job::get_travel_time, "from"_a, "to"_a, "agv"_a)
        .def("set_distance", py::overload_cast<MachineId, MachineId, double>(&Job::set_distance), "from"_a, "to"_a, "distance"_a)
        .def("set_distance", py::overload_cast<map<MachineId, map<MachineId, double>> &>(&Job::set_distance), "data"_a)
        .def("set_rand_distance", &Job::set_rand_distance, "min_dist"_a, "max_dist"_a)
        .def_static("rand_generate", &Job::rand_generate, "task_count"_a, "machine_count"_a, "AGV_count"_a, "machine_type_count"_a, "min_transport_time"_a, "max_transport_time"_a, "min_max_speed_ratio"_a, "min_process_time"_a, "max_process_time"_a)
        .def("features", &Job::features)
        .def("copy", &Job::copy)
        .def("act_move", &Job::act_move, "AGV_id"_a, "target_machine"_a)
        .def("act_pick", &Job::act_pick, "AGV_id"_a, "target_machine"_a)
        .def("act_transport", &Job::act_transport, "AGV_id"_a, "target_task"_a, "target_machine"_a)
        .def("wait", &Job::wait, "delta_time"_a)
        .def("__repr__", &Job::repr)
        .def_readonly_static("begin_task_id", &Job::begin_task_id)
        .def_readonly_static("end_task_id", &Job::end_task_id)
        .def_readonly_static("dummy_machine_id", &Job::dummy_machine_id)
        .def_readonly_static("task_feature_size", &Job::task_feature_size)
        .def_readonly_static("machine_feature_size", &Job::machine_feature_size)
        .def_readonly_static("AGV_feature_size", &Job::AGV_feature_size);

    py::class_<JobFeatures, shared_ptr<JobFeatures>>(m, "JobFeatures")
        .def_readonly("task_features", &JobFeatures::task_features)
        .def_readonly("task_relations", &JobFeatures::task_relations)
        .def_readonly("machine_features", &JobFeatures::machine_features)
        .def_readonly("processable_machine_mask", &JobFeatures::processable_machine_mask)
        .def_readonly("AGV_features", &JobFeatures::AGV_features);
}
