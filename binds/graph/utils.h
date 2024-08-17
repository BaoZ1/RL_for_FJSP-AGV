#pragma once

#include <string>
#include <ranges>
#include <sstream>
#include <unordered_set>
#include <optional>
#include <variant>

#include "graph.h"

using namespace std;

#ifdef NDEBUG
#define DEBUG_OUT(...) (void)0
#else
#define DEBUG_OUT(...) std::cout << format(__VA_ARGS__) << std::endl
#endif

constexpr string enum_string(TaskStatus s)
{
    switch (s)
    {
    case TaskStatus::blocked:
        return string("blocked");
    case TaskStatus::waiting_machine:
        return string("waiting_machine");
    case TaskStatus::waiting_material:
        return string("waiting_material");
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
        unreachable();
    }
}



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
        unreachable();
    }
}


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
        unreachable();
    }
}



constexpr string enum_string(ActionType t)
{
    switch (t)
    {
    case ActionType::move:
        return string("move");
    case ActionType::pick:
        return string("pick");
    case ActionType::transport:
        return string("transport");
    default:
        unreachable();
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

template <typename... Types>
    requires(derived_from<Types, TaskBase> && ...)
shared_ptr<TaskBase> to_task_base_ptr(variant<shared_ptr<Types>...> wrapped)
{
    return visit([](auto &&a)
                 { return static_pointer_cast<TaskBase>(a); }, wrapped);
}

