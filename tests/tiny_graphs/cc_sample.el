# Manual CC sample (undirected after --undirected).
# Nodes: 6 (ids 0..5)
# Topology:
#   Component A (size 4):  0 — 1 — 2
#                          |       |
#                          +—— 3 ——+
#   Component B (size 2):  4 — 5
#
# Expected: num_components=2, largest=4
# Symmetrized edge count: 8 + 2 = 10
0 1
1 2
2 3
3 0
4 5
