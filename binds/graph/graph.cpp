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

typedef size_t TaskId;
typedef size_t MachineId;
typedef size_t AGVId;
typedef size_t MachineType;

enum class TaskStatus
{
    blocked,
    waiting,
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
    case TaskStatus::finished:
        return string("finished");
    default:
        throw pybind11::value_error();
    }
}

enum class MachineStatus
{
    idle,
    working
};

constexpr string enum_string(MachineStatus s)
{
    switch (s)
    {
    case MachineStatus::idle:
        return string("idle");
    case MachineStatus::working:
        return string("working");
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

string add_indent(const string &s, size_t indent_count)
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

struct Task
{
    Task(TaskId id) : id(id) {}
    Task(TaskId id, MachineType mt) : id(id), machine_type(mt) {}

    virtual string identifier() = 0;
    virtual string base_info_string() = 0;

    string repr()
    {
        stringstream ss;
        ss << format("<{} ({})\n", this->identifier(), this->base_info_string());
        ss << add_indent(format("p: {}\n", id_set_string(this->precursors)), 1);
        ss << add_indent(format("s: {}", id_set_string(this->successors)), 1);
        ss << '>';
        return ss.str();
    }

    TaskId id;

    MachineType machine_type;
    TaskStatus status;
    // TODO 时间相关

    unordered_set<TaskId> precursors;
    unordered_set<TaskId> successors;
};

struct JobBegin : public Task
{
    JobBegin(TaskId id) : Task(id) {}
    string identifier() override
    {
        return string("begin");
    }
    string base_info_string() override
    {
        return format("id: {} status: {}", this->id, enum_string(this->status));
    }
};

struct JobEnd : public Task
{
    JobEnd(TaskId id) : Task(id) {}
    string identifier() override
    {
        return string("end");
    }
    string base_info_string() override
    {
        return format("id: {} status: {}", this->id, enum_string(this->status));
    }
};

struct SimplifiedTask : public Task
{
    SimplifiedTask(TaskId id, MachineType mt, TaskId rid, size_t idx) : Task(id, mt), raw_id(rid), index(idx) {}
    string identifier() override
    {
        return string("simplified");
    }
    string base_info_string() override
    {
        return format("id: {}, type: {}, raw_id: {}, index: {}, status: {}",
                      this->id, this->machine_type, this->raw_id, this->index, enum_string(this->status));
    }

    TaskId raw_id;
    size_t index;
};

struct RawTask : public Task
{
    RawTask(TaskId id, MachineType mt, size_t c) : Task(id, mt), count(c), current(0) {}
    string identifier() override
    {
        return string("raw");
    }
    string base_info_string() override
    {
        return format("id: {}, type: {}, count: {}, current: {}, status: {}",
                      this->id, this->machine_type, this->count, this->current, enum_string(this->status));
    }

    size_t count;
    size_t current;
};

struct Machine
{
    Machine(MachineId id, MachineType tp) : id(id), type(tp) {}
    string repr()
    {
        return format("<machine (type: {}, status: {})>", this->type, enum_string(this->status));
    }

    MachineId id;
    MachineType type;
    MachineStatus status;
    optional<TaskId> working_task;
};

struct AGV
{
    AGV(AGVId id, double speed) : id(id), speed(speed) {}
    string repr()
    {
        switch (this->status)
        {
        case AGVStatus::idle:
            return format("<AGV (speed: {:.2f}, status: {}, position: {}, loaded: {})>",
                          this->speed, enum_string(this->status), this->target, this->loaded);
        case AGVStatus::moving:
            return format("<AGV (speed: {:.2f}, status: {}, target: {}, time_remained: {}, loaded: {})>",
                          this->speed, enum_string(this->status), this->target, this->time_remained, this->loaded);
        default:
            throw exception();
        }
    }

    AGVId id;
    double speed;
    AGVStatus status;
    MachineId target;
    double time_remained;
    bool loaded;
};

template <typename T>
    requires derived_from<T, Task>
class Job
{
    typedef variant<shared_ptr<JobBegin>, shared_ptr<JobEnd>, shared_ptr<T>> TaskNodePtr;

public:
    Job()
    {
        auto job_begin_node = make_shared<JobBegin>(this->job_begin_node_id);
        job_begin_node->status = TaskStatus::waiting;
        this->tasks[this->job_begin_node_id] = job_begin_node;

        auto job_end_node = make_shared<JobEnd>(this->job_end_node_id);
        job_end_node->status = TaskStatus::blocked;
        this->tasks[this->job_end_node_id] = job_end_node;

        this->add_relation(this->job_begin_node_id, this->job_end_node_id);

        this->next_task_id = 2;
        this->next_machine_id = 0;
        this->next_AGV_id = 0;
    };

    void add_relation(TaskId from, TaskId to)
    {
        auto process = [from, to, this](auto &&from_v, auto &&to_v)
        {
            auto from_ptr = static_pointer_cast<Task>(from_v);
            auto to_ptr = static_pointer_cast<Task>(to_v);

            this->remove_relation(from, this->job_end_node_id);
            this->remove_relation(this->job_begin_node_id, to);

            from_ptr->successors.emplace(to);
            to_ptr->precursors.emplace(from);
        };

        visit(process, this->tasks[from], this->tasks[to]);
    }

    void remove_relation(TaskId from, TaskId to)
    {
        auto process = [from, to, this](auto &&from_v, auto &&to_v)
        {
            auto from_ptr = static_pointer_cast<Task>(from_v);
            auto to_ptr = static_pointer_cast<Task>(to_v);

            from_ptr->successors.erase(to);
            if (from_ptr->successors.empty() && to != this->job_end_node_id)
            {
                this->add_relation(from, this->job_end_node_id);
            }

            to_ptr->precursors.erase(from);
            if (to_ptr->precursors.empty() && from != this->job_begin_node_id)
            {
                this->add_relation(this->job_begin_node_id, to);
            }
        };

        visit(process, this->tasks[from], this->tasks[to]);
    }

    void set_task_relation(TaskId node_id, optional<unordered_set<TaskId>> precursors, optional<unordered_set<TaskId>> successors)
    {

        unordered_set<TaskId> real_precursors;
        if (!precursors.has_value() || precursors.value().empty())
        {
            real_precursors = unordered_set{this->job_begin_node_id};
        }
        else
        {
            real_precursors = precursors.value();
        }
        for (TaskId pid : real_precursors)
        {
            this->add_relation(pid, node_id);
        }

        unordered_set<TaskId> real_successors;
        if (!successors.has_value() || successors.value().empty())
        {
            real_successors = unordered_set{this->job_end_node_id};
        }
        else
        {
            real_successors = successors.value();
        }
        for (TaskId sid : real_successors)
        {
            this->add_relation(node_id, sid);
        }
    }

    void remove_task(TaskId id)
    {
        if (id == this->job_begin_node_id || id == this->job_end_node_id)
        {
            throw py::value_error();
        }

        auto node_ptr = get<shared_ptr<T>>(this->tasks[id]);
        for (TaskId pid : node_ptr->precursors)
        {
            this->remove_relation(pid, id);
        }
        for (TaskId sid : node_ptr->successors)
        {
            this->remove_relation(id, sid);
        }
        get<shared_ptr<JobBegin>>(this->tasks[this->job_begin_node_id])->successors.erase(id);
        get<shared_ptr<JobBegin>>(this->tasks[this->job_begin_node_id])->successors.erase(id);
        this->tasks.erase(id);
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
        node->status = MachineStatus::idle;

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
        new_AGV->target = init_pos;
        new_AGV->time_remained = 0;
        new_AGV->status = AGVStatus::idle;
        new_AGV->loaded = false;

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

    virtual string identifier() = 0;

    string repr()
    {
        stringstream ss;
        ss << format("<{}\n task: {}\n", this->identifier(), this->tasks.size());
        for (auto &&[id, task] : this->tasks)
        {
            visit([&ss](auto &&node_ptr)
                  { ss << add_indent(static_pointer_cast<Task>(node_ptr)->repr(), 1) << '\n'; }, task);
        }
        ss << format("\n machine: {}\n", this->machines.size());
        for (auto &&[_, machine] : this->machines)
        {
            ss << add_indent(machine->repr(), 1) << '\n';
        }
        ss << format("\n AGV: {}\n", this->AGVs.size());
        for (auto &&[_, agv] : this->AGVs)
        {
            ss << add_indent(agv->repr(), 1) << '\n';
        }
        ss << '>';
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
};

class RawJob;

class SimplifiedJob : public Job<SimplifiedTask>
{
public:
    using ProcessingTaskQueue = priority_queue<TaskId, vector<TaskId>, function<bool(TaskId, TaskId)>>;
    using MovingAGVQueue = priority_queue<AGVId, vector<AGVId>, function<bool(AGVId, AGVId)>>;

    SimplifiedJob() : Job<SimplifiedTask>(),
                      processing_tasks(ProcessingTaskQueue(bind(&SimplifiedJob::task_time_compare, this, ph::_1, ph::_2))),
                      moving_AGVs(MovingAGVQueue(bind(&SimplifiedJob::AGV_time_compare, this, ph::_1, ph::_2))) {}
    shared_ptr<RawJob> to_raw();

    TaskId add_task(
        MachineType type,
        TaskId raw_id,
        size_t index,
        optional<unordered_set<TaskId>> precursors,
        optional<unordered_set<TaskId>> successors)
    {
        auto node = make_shared<SimplifiedTask>(this->next_task_id++, type, raw_id, index);
        node->status = TaskStatus::blocked;
        this->tasks[node->id] = node;
        this->set_task_relation(node->id, precursors, successors);
        return node->id;
    }

    bool task_time_compare(TaskId a, TaskId b)
    {
        // TODO
    }

    bool AGV_time_compare(AGVId a, AGVId b)
    {
        // TODO
    }

    void step()
    {
        // TODO
    }

    string identifier() override
    {
        return string("job(simplified)");
    }

private:
    ProcessingTaskQueue processing_tasks;
    MovingAGVQueue moving_AGVs;
};

class RawJob : public Job<RawTask>
{
public:
    RawJob() : Job<RawTask>() {}
    shared_ptr<SimplifiedJob> simplify();

    TaskId add_task(MachineType type, size_t count, optional<unordered_set<TaskId>> precursors, optional<unordered_set<TaskId>> successors)
    {
        auto node = make_shared<RawTask>(this->next_task_id++, type, count);
        node->status = TaskStatus::blocked;
        this->tasks[node->id] = node;
        this->set_task_relation(node->id, precursors, successors);
        return node->id;
    }

    static shared_ptr<RawJob> rand_generate(
        size_t task_count,
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
        auto ret = make_shared<RawJob>();

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
        for (MachineId from_id = 0; from_id < machine_count; from_id++)
        {
            for (MachineId to_id = 0; to_id < machine_count; to_id++)
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

        vector<TaskId> exist_tasks;
        discrete_distribution<size_t> prev_count_dist({3, 5, 3, 1});
        discrete_distribution<size_t> repeat_count_dist({5, 3, 1});
        for (size_t i = 0; i < task_count; i++)
        {
            size_t prev_count = min(exist_tasks.size(), prev_count_dist(engine));
            binomial_distribution<TaskId> prev_sampler(exist_tasks.size() - 1, 0.8);
            unordered_set<TaskId> prevs;
            for (size_t j = 0; j < prev_count; j++)
            {
                TaskId sample = exist_tasks[prev_sampler(engine)];
                while (prevs.contains(sample))
                {
                    sample = exist_tasks[prev_sampler(engine)];
                }
                prevs.insert(sample);
            }
            exist_tasks.emplace_back(ret->add_task(machine_type_dist(engine), 1 + repeat_count_dist(engine), prevs, nullopt));
        }

        return ret;
    }

    string identifier() override
    {
        return string("job(raw)");
    }
};

shared_ptr<RawJob> SimplifiedJob::to_raw()
{
    auto raw = make_shared<RawJob>();
    map<TaskId, size_t> indegrees;
    for (auto &&[id, ptr] : this->tasks)
    {
        visit([&indegrees, id](auto p)
              { indegrees[id] = static_pointer_cast<Task>(p)->precursors.size(); }, ptr);
    }
    map<TaskId, TaskId> id_mapper;

    bool finished;
    do
    {
        finished = true;
        auto filter = views::filter([&id_mapper](const pair<TaskId, size_t> &pair)
                                    { return pair.second == 0 && !id_mapper.contains(pair.first); });
        for (auto [id, ind] : indegrees | filter)
        {
            finished = false;
            auto &&simplified_node = this->tasks[id];
            if (holds_alternative<shared_ptr<SimplifiedTask>>(simplified_node))
            {
                auto &&task_node = get<shared_ptr<SimplifiedTask>>(simplified_node);
                id_mapper[id] = task_node->raw_id;

                if (!raw->contains(task_node->raw_id))
                {
                    unordered_set<TaskId> precursors;
                    for (auto &&precursor : task_node->precursors)
                    {
                        precursors.insert(id_mapper[precursor]);
                    }
                    auto new_id = raw->add_task(task_node->machine_type, 1, precursors, nullopt);
                    auto new_node = get<shared_ptr<RawTask>>(raw->get_task_node(new_id));
                    new_node->current = task_node->status == TaskStatus::finished ? 1 : 0;
                    new_node->status = task_node->status;
                }
                else
                {
                    auto exist_node = raw->get_task_node(task_node->raw_id);
                    if (holds_alternative<shared_ptr<RawTask>>(exist_node))
                    {
                        auto &&exist_raw = get<shared_ptr<RawTask>>(exist_node);
                        exist_raw->count++;
                        if (task_node->status == TaskStatus::finished)
                        {
                            exist_raw->current++;
                        }
                        else if (exist_raw->status != TaskStatus::waiting)
                        {
                            exist_raw->status = task_node->status;
                        }
                    }
                }
            }
            else
            {
                id_mapper[id] = id;
            }
            visit([&indegrees](auto &&node)
                  { for(auto&& successor : static_pointer_cast<Task>(node)->successors)
                            indegrees[successor]--; },
                  simplified_node);
        }
    } while (!finished);
    if (id_mapper.size() != this->tasks.size())
    {
        cout << "graph has a cycle" << endl;
        throw exception();
    }

    for (auto &&[id, machine] : this->machines)
    {
        raw->add_machine(machine->type);
    }

    raw->set_distance(this->distances);

    return raw;
}

shared_ptr<SimplifiedJob> RawJob::simplify()
{
    auto simplified = make_shared<SimplifiedJob>();
    map<TaskId, size_t> indegrees;
    for (auto &&[id, ptr] : this->tasks)
    {
        visit([&indegrees, id](auto p)
              { indegrees[id] = static_pointer_cast<Task>(p)->precursors.size(); }, ptr);
    }
    map<TaskId, unordered_set<TaskId>> id_mapper;

    bool finished;
    do
    {
        finished = true;
        auto filter = views::filter([&id_mapper](const pair<TaskId, size_t> &pair)
                                    { return pair.second == 0 && !id_mapper.contains(pair.first); });
        for (auto [id, ind] : indegrees | filter)
        {
            finished = false;
            auto &&raw_node = this->tasks[id];
            id_mapper[id] = unordered_set<TaskId>{};
            if (holds_alternative<shared_ptr<RawTask>>(raw_node))
            {
                auto &&task_node = get<shared_ptr<RawTask>>(raw_node);
                TaskId prev_id;
                for (size_t idx = 0; idx < task_node->count; idx++)
                {
                    unordered_set<TaskId> precursors;
                    for (TaskId raw_precursor : task_node->precursors)
                    {
                        precursors.insert_range(id_mapper[raw_precursor]);
                    }
                    if (idx != 0)
                    {
                        precursors.insert(prev_id);
                    }
                    TaskId new_id = simplified->add_task(task_node->machine_type, id, idx, precursors, nullopt);
                    auto new_task = get<shared_ptr<SimplifiedTask>>(simplified->get_task_node(new_id));
                    switch (comp2int(new_task->index <=> task_node->current))
                    {
                    case 1:
                        new_task->status = TaskStatus::blocked;
                        break;
                    case 0:
                        new_task->status = task_node->status;
                        break;
                    case -1:
                        new_task->status = TaskStatus::finished;
                        break;
                    }
                    id_mapper[id].insert(new_id);
                    id_mapper[id].erase(prev_id);
                    prev_id = new_id;
                }
            }
            else
            {
                id_mapper[id].insert(id);
            }
            visit([&indegrees](auto &&node)
                  { for(auto&& successor : static_pointer_cast<Task>(node)->successors)
                            indegrees[successor]--; },
                  raw_node);
        }
    } while (!finished);
    if (id_mapper.size() != this->tasks.size())
    {
        cout << "graph has a cycle" << endl;
        throw exception();
    }

    for (auto &&[id, machine] : this->machines)
    {
        simplified->add_machine(machine->type);
    }

    simplified->set_distance(this->distances);

    return simplified;
}

using namespace py::literals;

PYBIND11_MODULE(graph, m)
{
    py::enum_<TaskStatus>(m, "TaskStatus")
        .value("blocked", TaskStatus::blocked)
        .value("waiting", TaskStatus::waiting)
        .value("finished", TaskStatus::finished);
    py::enum_<MachineStatus>(m, "MachineStatus")
        .value("idle", MachineStatus::idle)
        .value("working", MachineStatus::working);

    py::class_<JobBegin>(m, "JobBegin")
        .def("__repr__", &JobBegin::repr);
    py::class_<JobEnd>(m, "JobEnd")
        .def("__repr__", &JobEnd::repr);
    py::class_<RawTask>(m, "RawTask")
        .def("__repr__", &RawTask::repr);
    py::class_<SimplifiedTask>(m, "SimplifiedTask")
        .def("__repr__", &SimplifiedTask::repr);

    py::class_<Machine>(m, "Machine")
        .def("__repr__", &Machine::repr);

    py::class_<AGV>(m, "AGV")
        .def("__repr__", &AGV::repr);

    py::class_<SimplifiedJob, shared_ptr<SimplifiedJob>>(m, "SimplifiedJob")
        .def(py::init<>())
        .def("add_task", &SimplifiedJob::add_task)
        .def("add_relation", &SimplifiedJob::add_relation)
        .def("remove_relation", &SimplifiedJob::remove_relation)
        .def("get_task_node", &SimplifiedJob::get_task_node)
        .def("add_machine", &SimplifiedJob::add_machine)
        .def("get_machine_node", &SimplifiedJob::get_machine_node)
        .def("to_raw", &SimplifiedJob::to_raw)
        .def("__repr__", &SimplifiedJob::repr);
    py::class_<RawJob, shared_ptr<RawJob>>(m, "RawJob")
        .def(py::init<>())
        .def("add_task", &RawJob::add_task, "machine_type"_a, "repeat_count"_a = 1, "precursors"_a = nullopt, "successors"_a = nullopt)
        .def("add_relation", &RawJob::add_relation, "from"_a, "to"_a)
        .def("remove_relation", &RawJob::remove_relation, "from"_a, "to"_a)
        .def("remove_task", &RawJob::remove_task, "task_id"_a)
        .def("get_task_node", &RawJob::get_task_node, "task_id"_a)
        .def("add_machine", &RawJob::add_machine, "machine_type"_a)
        .def("get_machine_node", &RawJob::get_machine_node, "machine_id"_a)
        .def("get_travel_time", &RawJob::get_travel_time, "from"_a, "to"_a, "agv"_a)
        .def("set_distance", py::overload_cast<MachineId, MachineId, double>(&RawJob::set_distance))
        .def("set_distance", py::overload_cast<map<MachineId, map<MachineId, double>> &>(&RawJob::set_distance))
        .def("simplify", &RawJob::simplify)
        .def_static("rand_generate", &RawJob::rand_generate, "task_count"_a, "machine_count"_a, "AGV_count"_a, "machine_type_count"_a, "min_transport_time"_a, "max_transport_time"_a, "min_max_speed_ratio"_a, "min_process_time"_a, "max_process_time"_a)
        .def("__repr__", &RawJob::repr);
}
