# Manual PageRank sample (directed).
# Nodes: 5 (ids 0..4)
# Edges after drop self-loop + dedupe:
#   0->1, 0->2, 1->2, 2->3, 3->4, 4->0
#
# With damping=0.85, tolerance=1e-12, uniform init:
#   scores ≈ [0.21514102539680685, 0.12143493579373046,
#             0.2246546312184993, 0.220956436535761, 0.21781297105520223]
#   (119 iterations to L1 < 1e-12)
0	1
0	2
1	2
2	3
3	4
4	0
0	1
2	2
