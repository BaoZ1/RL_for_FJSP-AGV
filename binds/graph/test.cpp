#include "graph.cpp"

int main()
{
    auto g = make_shared<RawJob>();
    cout << "aaaa\n";
    g->add_task(0, 1, nullopt, nullopt);
    cout << g->repr();
    return 0;
}