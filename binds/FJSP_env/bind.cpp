#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FJSP_env.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(FJSP_env, m)
{
    py::enum_<OperationStatus>(m, "OperationStatus")
        .value("blocked", OperationStatus::blocked)
        .value("unscheduled", OperationStatus::unscheduled)
        .value("waiting", OperationStatus::waiting)
        .value("processing", OperationStatus::processing)
        .value("finished", OperationStatus::finished);
    py::enum_<MachineStatus>(m, "MachineStatus")
        .value("idle", MachineStatus::idle)
        .value("waiting_material", MachineStatus::waiting_material)
        .value("working", MachineStatus::working);
    py::enum_<AGVStatus>(m, "AGVStatus")
        .value("idle", AGVStatus::idle)
        .value("moving", AGVStatus::moving)
        .value("picking", AGVStatus::picking)
        .value("transporting", AGVStatus::transporting);
    py::enum_<ActionType>(m, "ActionType")
        .value("move", ActionType::move)
        .value("pick", ActionType::pick)
        .value("transport", ActionType::transport)
        .value("wait", ActionType::wait);

    py::class_<Position, shared_ptr<Position>>(m, "Position")
        .def("__repr__", &Position::repr);

    py::class_<Product, shared_ptr<Product>>(m, "Product")
        .def_readonly("operation_from", &Product::from)
        .def_readonly("operation_to", &Product::to)
        .def("__repr__", &Product::repr);

    py::class_<Operation, shared_ptr<Operation>>(m, "Operation")
        .def("__repr__", &Operation::repr);

    py::class_<Machine, shared_ptr<Machine>>(m, "Machine")
        .def("__repr__", &Machine::repr);

    py::class_<AGV, shared_ptr<AGV>>(m, "AGV")
        .def("__repr__", &AGV::repr);

    py::class_<Action, shared_ptr<Action>>(m, "Action")
        .def(py::init<ActionType>(), "action_type"_a)
        .def(py::init<ActionType, AGVId, MachineId>(), "action_type"_a, "AGV_id"_a, "target_machine"_a)
        .def(py::init<ActionType, AGVId, MachineId, Product>(), "action_type"_a, "AGV_id"_a, "target_machine"_a, "target_product"_a)
        .def_readwrite("action_type", &Action::type)
        .def_readwrite("AGV_id", &Action::act_AGV)
        .def_readwrite("target_machine", &Action::target_machine)
        .def_readwrite("target_product", &Action::target_product)
        .def("__repr__", &Action::repr);

    py::class_<GenerateParam, shared_ptr<GenerateParam>>(m, "GenerateParam")
        .def(py::init<size_t, size_t, size_t, size_t, float, float, float, float, float>(), "operation_count"_a, "machine_count"_a, "AGV_count"_a, "machine_type_count"_a, "min_transport_time"_a, "max_transport_time"_a, "min_max_speed_ratio"_a, "min_process_time"_a, "max_process_time"_a);

    py::class_<Graph, shared_ptr<Graph>>(m, "Graph")
        .def(py::init<>())
        .def("get_state", &Graph::get_state)
        .def_static("from_state", &Graph::from_state, "state"_a)
        .def("get_operations_id", &Graph::get_operations_id)
        .def("get_machines_id", &Graph::get_machines_id)
        .def("get_AGVs_id", &Graph::get_AGVs_id)
        .def("insert_operation", &Graph::insert_operation, "machine_type"_a, "process_time"_a, "predecessors"_a = nullopt, "successors"_a = nullopt)
        .def("remove_operation", &Graph::remove_operation, "operation_id"_a)
        .def("get_operation", &Graph::get_operation, "operation_id"_a)
        .def("add_machine", &Graph::add_machine, "machine_type"_a, "position"_a)
        .def("get_machine", &Graph::get_machine, "machine_id"_a)
        .def("add_AGV", &Graph::add_AGV, "speed"_a, "init_pos"_a)
        .def("get_AGV", &Graph::get_AGV, "AGV_id"_a)
        .def("get_timestamp", &Graph::get_timestamp)
        .def("get_travel_time", &Graph::get_travel_time, "from"_a, "to"_a, "agv"_a)
        .def("add_path", &Graph::add_path, "a"_a, "b"_a)
        .def("remove_path", &Graph::remove_path, "a"_a, "b"_a)
        .def("calc_distance", &Graph::calc_distance)
        .def("get_paths", &Graph::get_paths)
        .def_static("rand_generate", &Graph::rand_generate, "param"_a)
        .def("copy", &Graph::copy)
        .def("reset", &Graph::reset)
        .def("init", &Graph::init)
        .def("features", &Graph::features)
        .def("progress", &Graph::progress)
        .def("finished", &Graph::finished)
        .def("finish_time_lower_bound", &Graph::finish_time_lower_bound)
        .def("get_available_actions", &Graph::get_available_actions)
        .def("act", &Graph::act, "action"_a)
        .def_static("batch_step", &Graph::batch_step, "envs"_a, "actions"_a)
        .def("__repr__", &Graph::repr)
        .def_readonly_static("begin_operation_id", &Graph::begin_operation_id)
        .def_readonly_static("end_operation_id", &Graph::end_operation_id)
        .def_readonly_static("dummy_machine_id", &Graph::dummy_machine_id)
        .def_readonly_static("global_feature_size", &Graph::global_feature_size)
        .def_readonly_static("operation_feature_size", &Graph::operation_feature_size)
        .def_readonly_static("machine_feature_size", &Graph::machine_feature_size)
        .def_readonly_static("AGV_feature_size", &Graph::AGV_feature_size);

    py::class_<GraphFeature, shared_ptr<GraphFeature>>(m, "GraphFeature")
        .def_readonly("global_feature", &GraphFeature::global_feature)
        .def_readonly("operation_features", &GraphFeature::operation_features)
        .def_readonly("predecessor_idx", &GraphFeature::predecessor_idx)
        .def_readonly("successor_idx", &GraphFeature::successor_idx)
        .def_readonly("machine_features", &GraphFeature::machine_features)
        .def_readonly("processable_idx", &GraphFeature::processable_idx)
        .def_readonly("processing", &GraphFeature::processing)
        .def_readonly("waiting", &GraphFeature::waiting)
        .def_readonly("AGV_features", &GraphFeature::AGV_features)
        .def_readonly("AGV_position", &GraphFeature::AGV_position)
        .def_readonly("AGV_target", &GraphFeature::AGV_target)
        .def_readonly("AGV_loaded", &GraphFeature::AGV_loaded);
    
    py::class_<IdIdxMapper, shared_ptr<IdIdxMapper>>(m, "IdIdxMapper")
        .def_readonly("operation", &IdIdxMapper::operation)
        .def_readonly("machine", &IdIdxMapper::machine)
        .def_readonly("AGV", &IdIdxMapper::AGV);
}