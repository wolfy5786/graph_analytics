# Manual sample: directed 5-node graph for CSR build validation
# Nodes: 5 (ids 0..4)
# Raw directed edges listed below (includes one duplicate and one self-loop
# so the builder can exercise drop/dedupe). After drop self-loop + dedupe:
# Edges: 6
#   0->1, 0->2, 1->2, 2->3, 3->4, 4->0
0	1
0	2
1	2
2	3
3	4
4	0
0	1
2	2
