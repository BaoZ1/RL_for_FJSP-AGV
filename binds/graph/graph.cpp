#include <variant>
#include <optional>
#include <memory>
#include <format>
#include <ranges>
#include <iostream>
#include <concepts>
#include <random>
#include <functional>

#include <map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <queue>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;
namespace ph = placeholders;

#define OUT(...) cout << format(__VA_ARGS__) << endl;

using TaskId = size_t;
using MachineId = size_t;
using AGVId = size_t;
using MachineType = size_t;

enum class TaskStatus
{
    blocked,
    lack_of_materials,
    waiting,
    processing,
    need_transport,
    transporting,
    finished
};

constexpr string enum_string(TaskStatus s)
{
    switch (s)
    {
    case TaskStatus::blocked:
        return string("blocked");
    case TaskStatus::lack_of_materials:
        return string("lack_of_materials");
    case TaskStatus::waiting:
        return string("waiting");
    case TaskStatus::processing:
        return string("processing");
    case TaskStatus::need_transport:
        return string("need_transport");
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
    working,
    holding_product,
};

constexpr string enum_string(MachineStatus s)
{
    switch (s)
    {
    case MachineStatus::idle:
        return string("idle");
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
    moving
};

constexpr string enum_string(AGVStatus s)
{
    switch (s)
    {
    case AGVStatus::idle:
        return string("idle");
    case AGVStatus::moving:
        return string("moving");
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

string add_indent(const string &s, size_t indent_count=1)
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
    TaskBase(TaskId id) : id(id) {}

    virtual string repr() = 0;

    TaskId id;
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
    JobBegin(TaskId id) : TaskBase(id) {}

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
    JobEnd(TaskId id) : TaskBase(id) {}

    string repr() override
    {
        stringstream ss;
        ss << format("<end (id: {})\n", this->id);
        ss << add_indent(format("p: {}", id_set_string(this->precursors)));
        ss << "\n>";
        return ss.str();
    }

    unordered_set<TaskId> precursors;
};

struct Task : public TaskBase
{
    Task(TaskId id, MachineType mt, double pt) : TaskBase(id),
                                                 machine_type(mt),
                                                 status(TaskStatus::blocked),
                                                 process_time(pt),
                                                 finish_timestamp(0),
                                                 processing_machine(nullopt) {}

    string repr() override
    {
        stringstream ss;
        ss << format("<task (id: {}, type: {}, process_time: {:.2f}, status: {})\n",
                     this->id, this->machine_type, this->process_time, enum_string(this->status));
        ss << add_indent(format("p: {}", this->precursor));
        ss << add_indent(format("s: {}", this->successor));
        ss << "\n>";
        return ss.str();
    }

    void start_process(double current_timestamp, MachineId process_machine)
    {
        assert(this->status != TaskStatus::waiting);

        this->status = TaskStatus::processing;
        this->processing_machine = process_machine;
        this->finish_timestamp = current_timestamp + this->process_time;
    }

    MachineType machine_type;
    TaskStatus status;

    double process_time;
    double finish_timestamp;

    TaskId precursor;
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
    AGV(AGVId id, double speed) : id(id),
                                  speed(speed),
                                  status(AGVStatus::idle),
                                  finish_timestamp(0),
                                  loaded_item(nullopt) {}
    string repr()
    {
        switch (this->status)
        {
        case AGVStatus::idle:
            return format("<AGV (speed: {:.2f}, status: {}, position: {}, loaded_item: {})>",
                          this->speed, enum_string(this->status), this->position, this->loaded_item.value_or(0));
        case AGVStatus::moving:
            return format("<AGV (speed: {:.2f}, status: {}, position: {}, finish_timestamp: {}, loaded_item: {})>",
                          this->speed, enum_string(this->status), this->position, this->finish_timestamp, this->loaded_item.value_or(0));
        default:
            throw exception();
        }
    }

    void start_move(MachineId target, optional<TaskId> item, double current_timestamp, double distance)
    {
        this->position = target;
        this->loaded_item = item;
        this->status = AGVStatus::moving;
        this->finish_timestamp = current_timestamp + distance / this->speed;
    }

    void finish()
    {
        this->status = AGVStatus::idle;
        this->loaded_item = nullopt;
    }

    AGVId id;
    double speed;
    AGVStatus status;
    MachineId position;
    double finish_timestamp;
    optional<TaskId> loaded_item;
};

class Job
{
    using TaskNodePtr = variant<shared_ptr<JobBegin>, shared_ptr<JobEnd>, shared_ptr<Task>>;
    using ProcessingTaskQueue = priority_queue<TaskId, vector<TaskId>, function<bool(TaskId, TaskId)>>;
    using MovingAGVQueue = priority_queue<AGVId, vector<AGVId>, function<bool(AGVId, AGVId)>>;

public:
    Job() : processing_tasks(ProcessingTaskQueue(bind(&Job::task_time_compare, this, ph::_1, ph::_2))),
            moving_AGVs(MovingAGVQueue(bind(&Job::AGV_time_compare, this, ph::_1, ph::_2)))
    {
        this->tasks[this->job_begin_node_id] = make_shared<JobBegin>(this->job_begin_node_id);
        this->tasks[this->job_end_node_id] = make_shared<JobEnd>(this->job_end_node_id);

        this->next_task_id = 2;
        this->next_machine_id = 1;
        this->next_AGV_id = 0;
    };

    TaskId add_task(
        MachineType type,
        double process_time,
        optional<TaskId> precursor,
        optional<TaskId> successor)
    {
        assert(!(precursor.has_value() && successor.has_value()));
        auto node = make_shared<Task>(this->next_task_id++, type, process_time);
        this->tasks[node->id] = node;
        if (precursor.has_value())
        {
            node->precursor = precursor.value();
            auto p_node = get<shared_ptr<Task>>(this->tasks[precursor.value()]);
            node->successor = p_node->successor;
            p_node->successor = node->id;
            if (node->successor == this->job_end_node_id)
            {
                auto &&job_end_node = get<shared_ptr<JobEnd>>(this->tasks[this->job_end_node_id]);
                job_end_node->precursors.erase(p_node->id);
                job_end_node->precursors.emplace(node->id);
            }
        }
        else if (successor.has_value())
        {
            node->successor = successor.value();
            auto s_node = get<shared_ptr<Task>>(this->tasks[successor.value()]);
            node->precursor = s_node->precursor;
            s_node->precursor = node->id;
            if (node->precursor == this->job_begin_node_id)
            {
                auto &&job_begin_node = get<shared_ptr<JobBegin>>(this->tasks[this->job_begin_node_id]);
                job_begin_node->successors.erase(s_node->id);
                job_begin_node->successors.emplace(node->id);
            }
        }
        else
        {
            node->precursor = this->job_begin_node_id;
            node->successor = this->job_end_node_id;
            get<shared_ptr<JobBegin>>(this->tasks[this->job_begin_node_id])->successors.emplace(node->id);
            get<shared_ptr<JobEnd>>(this->tasks[this->job_end_node_id])->precursors.emplace(node->id);
        }

        return node->id;
    }

    void remove_task(TaskId id)
    {
        assert(id != this->job_begin_node_id && id != this->job_end_node_id);

        auto node_ptr = get<shared_ptr<Task>>(this->tasks[id]);
        if (holds_alternative<shared_ptr<JobBegin>>(this->tasks[node_ptr->precursor]))
        {
            get<shared_ptr<JobBegin>>(this->tasks[node_ptr->precursor])->successors.erase(id);
        }
        else
        {
            get<shared_ptr<Task>>(this->tasks[node_ptr->precursor])->successor = node_ptr->successor;
        }

        if (holds_alternative<shared_ptr<JobEnd>>(this->tasks[node_ptr->successor]))
        {
            get<shared_ptr<JobEnd>>(this->tasks[node_ptr->successor])->precursors.erase(id);
        }
        else
        {
            get<shared_ptr<Task>>(this->tasks[node_ptr->successor])->precursor = node_ptr->precursor;
        }
    }

    bool contains(TaskId id)
    {
        return this->tasks.contains(id);
    }

    TaskNodePtr get_task_node(TaskId id)
    {
        return this->tasks[id];
    }

    MachineId add_machine(MachineType type)
    {
        auto node = make_shared<Machine>(this->next_machine_id++, type);
        this->machines[node->id] = node;
        return node->id;
    }

    shared_ptr<Machine> get_machine_node(MachineId id)
    {
        return this->machines[id];
    }

    AGVId add_AGV(double speed, MachineId init_pos)
    {
        auto new_AGV = make_shared<AGV>(this->next_AGV_id++, speed);
        new_AGV->position = init_pos;

        this->AGVs[new_AGV->id] = new_AGV;

        return new_AGV->id;
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
        ss << format("<job\n task: {}\n", this->tasks.size());
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
        for(auto&& [id, _] : this->machines)
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

        uniform_int_distribution<MachineType> machine_type_dist(0, machine_type_count - 1);
        vector<MachineId> machines;
        for (MachineType t = 0; t < machine_type_count; t++)
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

    bool task_time_compare(TaskId a, TaskId b)
    {
        return get<shared_ptr<Task>>(this->tasks[a])->finish_timestamp < get<shared_ptr<Task>>(this->tasks[b])->finish_timestamp;
    }

    bool AGV_time_compare(AGVId a, AGVId b)
    {
        return this->AGVs[a]->finish_timestamp < this->AGVs[b]->finish_timestamp;
    }

    void schedule_AGV(AGVId id, MachineId target, bool with_item)
    {
        auto AGV = this->AGVs[id];
        assert(AGV->status == AGVStatus::idle);

        if (!with_item)
        {
            AGV->start_move(target, nullopt, this->timestamp, this->distances[AGV->position][target]);
        }
        else
        {
            auto current_machine = this->machines[AGV->position];
            assert(current_machine->status != MachineStatus::holding_product);
            assert(current_machine->working_task.has_value());
            auto transport_task = get<shared_ptr<Task>>(this->tasks[current_machine->working_task.value()]);
            assert(transport_task->status == TaskStatus::need_transport);
            transport_task->status = TaskStatus::transporting;
            current_machine->status = MachineStatus::idle;
            current_machine->working_task = nullopt;
            AGV->start_move(target, transport_task->id, this->timestamp, this->distances[AGV->position][target]);
        }
    }

    void wait()
    {
        if (!this->processing_tasks.empty() && !this->moving_AGVs.empty())
        {
            auto nearest_task = get<shared_ptr<Task>>(this->tasks[this->processing_tasks.top()]);
            auto nearest_AGV = this->AGVs[this->moving_AGVs.top()];

            if (nearest_task->finish_timestamp < nearest_AGV->finish_timestamp)
            {
                this->processing_tasks.pop();
                return this->wait_task(nearest_task->id);
            }
            else
            {
                this->moving_AGVs.pop();
                return this->wait_AGV(nearest_AGV->id);
            }
        }

        if (!this->processing_tasks.empty())
        {
            TaskId id = this->processing_tasks.top();
            this->processing_tasks.pop();
            return this->wait_task(id);
        }

        if (!this->moving_AGVs.empty())
        {
            AGVId id = this->moving_AGVs.top();
            this->moving_AGVs.pop();
            return this->wait_AGV(id);
        }
    }

    void wait_task(TaskId id)
    {
        auto task = get<shared_ptr<Task>>(this->tasks[id]);
        this->timestamp = task->finish_timestamp;
        assert(task->status == TaskStatus::processing);
        task->status = TaskStatus::need_transport;
        auto machine = this->machines[task->processing_machine.value()];
        assert(machine->status == MachineStatus::working);
        machine->status = MachineStatus::holding_product;
    }

    void wait_AGV(AGVId id)
    {
        auto AGV = this->AGVs[id];
        this->timestamp = AGV->finish_timestamp;
        if (AGV->loaded_item.has_value())
        {
            auto task = get<shared_ptr<Task>>(this->tasks[AGV->loaded_item.value()]);
            assert(task->status == TaskStatus::transporting);
            task->status = TaskStatus::finished;
            if (holds_alternative<shared_ptr<Task>>(this->tasks[task->successor]))
            {
                auto &&succ_task = get<shared_ptr<Task>>(this->tasks[task->successor]);
                assert(succ_task->status == TaskStatus::lack_of_materials);
                succ_task->status = TaskStatus::waiting;
            }
        }

        AGV->finish();
    }

protected:
    const TaskId job_begin_node_id = 0, job_end_node_id = 1;
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

using namespace py::literals;

PYBIND11_MODULE(graph, m)
{
    py::enum_<TaskStatus>(m, "TaskStatus")
        .value("blocked", TaskStatus::blocked)
        .value("lack_of_materials", TaskStatus::lack_of_materials)
        .value("waiting", TaskStatus::waiting)
        .value("need_transport", TaskStatus::need_transport)
        .value("transporting", TaskStatus::transporting)
        .value("finished", TaskStatus::finished);
    py::enum_<MachineStatus>(m, "MachineStatus")
        .value("idle", MachineStatus::idle)
        .value("working", MachineStatus::working)
        .value("holding_product", MachineStatus::holding_product);

    py::class_<JobBegin>(m, "JobBegin")
        .def("__repr__", &JobBegin::repr);
    py::class_<JobEnd>(m, "JobEnd")
        .def("__repr__", &JobEnd::repr);
    py::class_<Task>(m, "Task")
        .def("__repr__", &Task::repr);

    py::class_<Machine>(m, "Machine")
        .def("__repr__", &Machine::repr);

    py::class_<AGV>(m, "AGV")
        .def("__repr__", &AGV::repr);

    py::class_<Job, shared_ptr<Job>>(m, "Job")
        .def(py::init<>())
        .def("add_task", &Job::add_task, "machine_type"_a, "process_time"_a, "precursors"_a, "successors"_a)
        .def("remove_task", &Job::remove_task, "task_id"_a)
        .def("get_task_node", &Job::get_task_node, "task_id"_a)
        .def("add_machine", &Job::add_machine, "machine_type"_a)
        .def("get_machine_node", &Job::get_machine_node, "machine_id"_a)
        .def("get_travel_time", &Job::get_travel_time, "from"_a, "to"_a, "agv"_a)
        .def("set_distance", py::overload_cast<MachineId, MachineId, double>(&Job::set_distance))
        .def("set_distance", py::overload_cast<map<MachineId, map<MachineId, double>> &>(&Job::set_distance))
        .def_static("rand_generate", &Job::rand_generate, "task_count"_a, "machine_count"_a, "AGV_count"_a, "machine_type_count"_a, "min_transport_time"_a, "max_transport_time"_a, "min_max_speed_ratio"_a, "min_process_time"_a, "max_process_time"_a)
        .def("__repr__", &Job::repr);
}
