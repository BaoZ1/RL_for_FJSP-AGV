from binds import graph

r = graph.Job.rand_generate(
    [5, 3, 4],
    4,
    5,
    3,
    3,
    10,
    0.6,
    15,
    5,
)
print(r)
