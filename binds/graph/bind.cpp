#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "graph.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(graph, m)
{
    py::enum_<TaskStatus>(m, "TaskStatus")
        .value("blocked", TaskStatus::blocked)
        .value("waiting_machine", TaskStatus::waiting_machine)
        .value("waiting_material", TaskStatus::waiting_material)
        .value("need_transport", TaskStatus::need_transport)
        .value("waiting_transport", TaskStatus::waiting_transport)
        .value("transporting", TaskStatus::transporting)
        .value("finished", TaskStatus::finished);
    py::enum_<MachineStatus>(m, "MachineStatus")
        .value("idle", MachineStatus::idle)
        .value("lack_of_material", MachineStatus::lack_of_material)
        .value("working", MachineStatus::working)
        .value("holding_product", MachineStatus::holding_product);
    py::enum_<AGVStatus>(m, "AGVStatus")
        .value("idle", AGVStatus::idle)
        .value("moving", AGVStatus::moving)
        .value("picking", AGVStatus::picking)
        .value("transporting", AGVStatus::transporting);
    py::enum_<ActionType>(m, "ActionType")
        .value("move", ActionType::move)
        .value("pick", ActionType::pick)
        .value("transport", ActionType::transport);

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

    py::class_<Action, shared_ptr<Action>>(m, "Action")
        .def(py::init<ActionType, AGVId, MachineId, optional<TaskId>>(), "action_type"_a, "AGV_id"_a, "target_machine"_a, "target_task"_a = nullopt)
        .def_readwrite("action_type", &Action::type)
        .def_readwrite("AGV_id", &Action::act_AGV)
        .def_readwrite("target_machine", &Action::target_machine)
        .def_readwrite("target_task", &Action::target_task)
        .def("__repr__", &Action::repr);

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
        .def("init_task_status", &Job::init_task_status)
        .def("copy", &Job::copy)
        .def("features", &Job::features)
        .def("finished", &Job::finished)
        .def("get_available_actions", &Job::get_available_actions)
        .def("act", &Job::act, "action"_a)
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