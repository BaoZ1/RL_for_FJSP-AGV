#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <ranges>
#include <sstream>
#include <set>
#include <optional>
#include <variant>
#include <type_traits>
#include <concepts>
#include <execution>

#include "FJSP_env.h"

using namespace std;

#ifdef NDEBUG
#define DEBUG_OUT(...) (void)0
#else
#define DEBUG_OUT(...) std::cout << format(__VA_ARGS__) << std::endl
#endif

constexpr string enum_string(OperationStatus s)
{
    switch (s)
    {
    case OperationStatus::blocked:
        return string("blocked");
    case OperationStatus::unscheduled:
        return string("unscheduled");
    case OperationStatus::waiting:
        return string("waiting");
    case OperationStatus::processing:
        return string("processing");
    case OperationStatus::finished:
        return string("finished");
    default:
        assert(false);
        return string("unknown");
    }
}

constexpr string enum_string(MachineStatus s)
{
    switch (s)
    {
    case MachineStatus::idle:
        return string("idle");
    case MachineStatus::waiting_material:
        return string("waiting_material");
    case MachineStatus::working:
        return string("working");
    default:
        assert(false);
        return string("unknown");
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
        assert(false);
        return string("unknown");
    }
}

constexpr string enum_string(ActionType t)
{
    switch (t)
    {
    case ActionType::wait:
        return string("wait");
    case ActionType::move:
        return string("move");
    case ActionType::pick:
        return string("pick");
    case ActionType::transport:
        return string("transport");
    default:
        assert(false);
        return string("unknown");
    }
}

string id_set_string(const set<OperationId> &s)
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
string v2s(vector<T> vec)
{
    if (vec.empty())
    {
        return string("[]");
    }
    if (vec.size() == 1)
    {
        return format("[{}]", vec[0]);
    }
    stringstream ss;
    ss << "[" << vec[0];
    for (auto it = vec.begin() + 1; it != vec.end(); it++)
    {
        ss << "," << *it;
    }
    ss << "]";
    return ss.str();
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

template <typename T>
concept reprable = requires(T x) {
    { x.repr() } -> same_as<string>;
};

template <reprable T>
string o2s(optional<T> o, string on_empty = "null")
{
    if (o.has_value())
    {
        return o->repr();
    }
    return on_empty;
}

template <class Container>
concept iterable_container = requires(Container x) {
    { x.begin() } -> input_or_output_iterator;
    { x.end() } -> same_as<decltype(x.begin())>;
    { *(x.end()) } -> same_as<decltype(*(x.begin()))>;
};

template <iterable_container Container, typename T = remove_cvref_t<decltype(*declval<Container>().begin())>>
vector<T> random_unique(const Container &data, size_t num)
{
    vector<T> cp(data.begin(), data.end());
    auto begin = cp.begin();
    auto end = cp.end();
    size_t left = distance(begin, end);
    assert(left >= num);
    for (size_t i = 0; i < num; i++)
    {
        auto r = begin;
        advance(r, rand() % left);
        swap(*begin, *r);
        ++begin;
        --left;
    }
    cp.resize(num);
    return cp;
}



#endif