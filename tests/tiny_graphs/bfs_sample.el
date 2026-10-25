# Manual BFS sample (undirected after --undirected).
# Nodes: 6 (ids 0..5)
# Topology:
#   0 — 1 — 2 — 3
#   |         /
#   4 ————— 5
#
# Known distances from source 0:
#   0:0  1:1  2:2  3:3  4:1  5:2
# Known distances from source 3:
#   3:0  2:1  1:2  5:2  0:3  4:3
0 1
1 2
2 3
0 4
4 5
2 5
