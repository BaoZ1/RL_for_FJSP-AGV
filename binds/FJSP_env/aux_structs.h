#pragma once

#include <tuple>
#include <vector>
#include <cmath>

using namespace std;


using OperationId = size_t;
using MachineId = size_t;
using AGVId = size_t;
using MachineType = size_t;

enum class OperationStatus
{
    blocked,
    unscheduled,
    waiting,
    processing,
    // need_transport,
    // waiting_transport,
    // transporting,
    finished
};

struct Position
{
    float x;
    float y;

    string repr()
    {
        return format("({:.2f},{:.2f})", this->x, this->y);
    }

    float distance(Position other) {
        return hypotf(x - other.x, y - other.y);
    }
};

enum class MachineStatus
{
    idle,
    waiting_material,
    working,
    // holding_product,
};

enum class AGVStatus
{
    idle,
    moving,
    picking,
    transporting
};

enum class ActionType
{
    move,
    pick,
    transport,
    wait
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

template<typename T>
class UnionFind
{
public:
    UnionFind(vector<T> elements)
    {
        next_idx = 0;
        for(auto e : elements)
        {
            forward_mapper[e] = next_idx;
            backward_mapper[next_idx] = e;
            father[next_idx] = next_idx;
            next_idx++;
        }
    }

    T find(T target)
    {
        int idx = forward_mapper[target];
        return backward_mapper[father[idx] == idx ? idx : father[idx] = this->find(father[idx])];
    }

    void unite(T from, T to)
    {
        this->father[this->find(this->forward_mapper[from])] = this->find(this->forward_mapper[to]);
    }

    int count()
    {
        set<int> roots;
        for(auto [idx, _] : this->father)
        {
            roots.emplace(forward_mapper[find(idx)]);
        }
        return roots.size();
    }
private:
    int next_idx;
    map<T, int> forward_mapper;
    map<int, T> backward_mapper;
    map<int, int> father;
};