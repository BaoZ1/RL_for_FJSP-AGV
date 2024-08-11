from binds import graph

g = graph.RawJob()
# g.add_machine(0)
# g.add_machine(0)
# g.add_machine(1)
# t0 = g.add_task(0, 3)
# t1 = g.add_task(1)
# g.add_relation(t0, t1)
# g.remove_task(t1)
# print(g)
# s = g.simplify()
# print(s)
# print(s.to_raw())
# print(t0, g.get_task_node(t0))

# g = graph.SimplifiedJob()
# g.add_task(1, 1, 1, None, None)
# print(g)
r = graph.RawJob.rand_generate(
    5,
    2,
    1,
    1,
    10,
    3,
    0.6,
    15,
    5,
)
print(r)
print(r.simplify())
