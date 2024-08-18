#pragma once

#include <tuple>
#include <vector>

using namespace std;

using OperationId = size_t;
using MachineId = size_t;
using AGVId = size_t;
using MachineType = size_t;

enum class OperationStatus
{
    blocked,
    waiting_machine,
    waiting_material,
    processing,
    need_transport,
    waiting_transport,
    transporting,
    finished
};

enum class MachineStatus
{
    idle,
    lack_of_material,
    working,
    holding_product,
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
    transport
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