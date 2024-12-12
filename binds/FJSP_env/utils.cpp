#include <string>
#include <ranges>
#include <sstream>
#include <set>
#include <optional>
#include <variant>
#include <type_traits>
#include <concepts>
#include <execution>

#include "utils.h"

using namespace std;

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
string o2s(optional<T> o, string on_empty)
{
    if (o.has_value())
    {
        return to_string(o.value());
    }
    return on_empty;
}


template <reprable T>
string o2s(optional<T> o, string on_empty)
{
    if (o.has_value())
    {
        return o->repr();
    }
    return on_empty;
}

template <iterable_container Container, typename T>
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

vector<shared_ptr<Graph>> batch_step(const vector<shared_ptr<Graph>> &envs, const vector<shared_ptr<Action>> &actions)
{
    auto pairs = views::zip(vector(envs), actions) | ranges::to<vector>();

    for_each(execution::par_unseq,
             pairs.begin(), pairs.end(),
             [](tuple<shared_ptr<Graph>, shared_ptr<Action>> &item)
             {
                 auto &[env, action] = item;
                 env = env->act(*action);
             });
    return pairs | views::elements<0> | ranges::to<vector>();
}
