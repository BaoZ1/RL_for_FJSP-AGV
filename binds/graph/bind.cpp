#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FJSP_graph.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(graph, m)
{
    py::enum_<OperationStatus>(m, "OperationStatus")
        .value("blocked", OperationStatus::blocked)
        .value("waiting_machine", OperationStatus::waiting_machine)
        .value("waiting_material", OperationStatus::waiting_material)
        .value("need_transport", OperationStatus::need_transport)
        .value("waiting_transport", OperationStatus::waiting_transport)
        .value("transporting", OperationStatus::transporting)
        .value("finished", OperationStatus::finished);
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

    py::class_<GraphBegin, shared_ptr<GraphBegin>>(m, "GraphBegin")
        .def("__repr__", &GraphBegin::repr);
    py::class_<GraphEnd, shared_ptr<GraphEnd>>(m, "GraphEnd")
        .def("__repr__", &GraphEnd::repr);
    py::class_<Operation, shared_ptr<Operation>>(m, "Operation")
        .def("__repr__", &Operation::repr);

    py::class_<Machine, shared_ptr<Machine>>(m, "Machine")
        .def("__repr__", &Machine::repr);

    py::class_<AGV, shared_ptr<AGV>>(m, "AGV")
        .def("__repr__", &AGV::repr);

    py::class_<Action, shared_ptr<Action>>(m, "Action")
        .def(py::init<ActionType, AGVId, MachineId, optional<OperationId>>(), "action_type"_a, "AGV_id"_a, "target_machine"_a, "target_operation"_a = nullopt)
        .def_readwrite("action_type", &Action::type)
        .def_readwrite("AGV_id", &Action::act_AGV)
        .def_readwrite("target_machine", &Action::target_machine)
        .def_readwrite("target_operation", &Action::target_operation)
        .def("__repr__", &Action::repr);

    py::class_<Graph, shared_ptr<Graph>>(m, "Graph")
        .def(py::init<>())
        .def("add_operation", &Graph::add_operation, "machine_type"_a, "process_time"_a, "predecessors"_a = nullopt, "successors"_a = nullopt)
        .def("remove_operation", &Graph::remove_operation, "operation_id"_a)
        .def("get_operation", &Graph::get_operation, "operation_id"_a)
        .def("add_machine", &Graph::add_machine, "machine_type"_a)
        .def("get_machine", &Graph::get_machine, "machine_id"_a)
        .def("add_AGV", &Graph::add_AGV, "speed"_a, "init_pos"_a)
        .def("get_AGV", &Graph::get_AGV, "AGV_id"_a)
        .def("get_timestamp", &Graph::get_timestamp)
        .def("get_travel_time", &Graph::get_travel_time, "from"_a, "to"_a, "agv"_a)
        .def("set_distance", py::overload_cast<MachineId, MachineId, float>(&Graph::set_distance), "from"_a, "to"_a, "distance"_a)
        .def("set_distance", py::overload_cast<map<MachineId, map<MachineId, float>> &>(&Graph::set_distance), "data"_a)
        .def("set_rand_distance", &Graph::set_rand_distance, "min_dist"_a, "max_dist"_a)
        .def_static("rand_generate", &Graph::rand_generate, "operation_count"_a, "machine_count"_a, "AGV_count"_a, "machine_type_count"_a, "min_transport_time"_a, "max_transport_time"_a, "min_max_speed_ratio"_a, "min_process_time"_a, "max_process_time"_a)
        .def("init_operation_status", &Graph::init_operation_status)
        .def("copy", &Graph::copy)
        .def("features", &Graph::features)
        .def("finished", &Graph::finished)
        .def("finish_time_lower_bound", &Graph::finish_time_lower_bound)
        .def("get_available_actions", &Graph::get_available_actions)
        .def("act", &Graph::act, "action"_a)
        .def("wait", &Graph::wait, "delta_time"_a)
        .def("__repr__", &Graph::repr)
        .def_readonly_static("begin_operation_id", &Graph::begin_operation_id)
        .def_readonly_static("end_operation_id", &Graph::end_operation_id)
        .def_readonly_static("dummy_machine_id", &Graph::dummy_machine_id)
        .def_readonly_static("operation_feature_size", &Graph::operation_feature_size)
        .def_readonly_static("machine_feature_size", &Graph::machine_feature_size)
        .def_readonly_static("AGV_feature_size", &Graph::AGV_feature_size);

    py::class_<GraphFeatures, shared_ptr<GraphFeatures>>(m, "GraphFeatures")
        .def_readonly("operation_features", &GraphFeatures::operation_features)
        .def_readonly("operation_relations", &GraphFeatures::operation_relations)
        .def_readonly("machine_features", &GraphFeatures::machine_features)
        .def_readonly("processable_machine_mask", &GraphFeatures::processable_machine_mask)
        .def_readonly("AGV_features", &GraphFeatures::AGV_features);
}