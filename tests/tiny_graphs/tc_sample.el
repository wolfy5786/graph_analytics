# Manual triangle-counting sample (undirected after --undirected).
# Nodes: 7 (ids 0..6)
#
# Topology:
#   Clique K4 on {0,1,2,3}:  C(4,3) = 4 triangles
#   Extra triangle {3,4,5}:  +1
#   Pendant edge 5-6:        +0
#   Bridge edge 2-4:         creates no new K3 beyond {3,4,5} wait —
#     edges 2-4, 3-4, 2-3 already in K4⇒{2,3,4} is a 5th triangle
#
# Final expected undirected triangle count: 5
#   K4 {0,1,2,3}: (0,1,2) (0,1,3) (0,2,3) (1,2,3)
#   {2,3,4}: one more
#   {3,4,5}: one more → wait that's 6.
#
# Recount carefully:
#   K4 edges: 0-1,0-2,0-3,1-2,1-3,2-3
#   Plus: 2-4, 3-4, 3-5, 4-5, 5-6
#   Triangles:
#     {0,1,2} {0,1,3} {0,2,3} {1,2,3}  (from K4) = 4
#     {2,3,4} (edges 2-3,2-4,3-4) = 1
#     {3,4,5} (edges 3-4,3-5,4-5) = 1
#   Total = 6
#
0 1
0 2
0 3
1 2
1 3
2 3
2 4
3 4
3 5
4 5
5 6
