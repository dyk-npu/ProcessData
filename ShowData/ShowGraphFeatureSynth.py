import dgl
import torch

# 指定.bin文件的路径
bin_file_path = "../Data/MFInstSeg/bin/20221121_154647_1.bin"

# print("bin_file_path:",type(bin_file_path))
# 加载图
graphfile = dgl.data.utils.load_graphs(bin_file_path)
print("graphfile:",graphfile)

graphs, geo = dgl.data.utils.load_graphs(bin_file_path)

# print("graphs,geo",graphs,geo)
# 获取第一个图（如果有多个图存储在文件中）
graph = graphs[0]
# print("geo:", geo)

print("type(geo):",type(geo))
keys = dict.keys(geo)
print("keys:", list(keys))

edges_path = geo['edges_path']
d2_distance = geo['d2_distance']
angle_distance = geo['angle_distance']
spatial_pos = geo['spatial_pos']

print("spatial_pos.shape:",spatial_pos.shape)
print("spatial_pos:",spatial_pos)

print("d2_distance.shape:" ,d2_distance.shape)
print("angle_distance.shape:",angle_distance.shape)

edges_pathTemp = edges_path[34]
print("edges_path.shape:",edges_path.shape)
print("edges_path:",edges_path[0])

print("graph:", graph)
print("type:", type(graph),type(geo))

print("show geo feature: ",)

# node_feature_a = graph.ndata['a']
# print("node_feature_a.shape:", node_feature_a.shape)
# print("node_feature_a:", node_feature_a)

#
# node_feature_y = graph.ndata['y']
# print("node_feature_y.shape:", node_feature_y.shape)
# print("node_feature_y:", node_feature_y)
#

# node_feature_z = graph.ndata['z']
# print("node_feature_z:", node_feature_z)
# print("node_feature_z.shape:", node_feature_z.shape)

#
# node_feature_l = graph.ndata['l']
# print("node_feature_l.shape:", node_feature_l.shape)
# print("node_feature_l:", node_feature_l)
#

# node_feature_f = graph.ndata['f']
# print("node_feature_f.shape:", node_feature_f.shape)
# print("node_feature_f:", node_feature_f)

#--------------------------------

# node_feature_t = graph.ndata['t']
# print("node_feature_t.shape:", node_feature_t.shape)
# print("node_feature_t:", node_feature_t)
#
# node_feature_a = graph.ndata['a']
# print("node_feature_a.shape:", node_feature_a.shape)
# print("node_feature_a:", node_feature_a)
#
# node_feature_r = graph.ndata['r']
# print("node_feature_r.shape:", node_feature_r.shape)
# print("node_feature_r:", node_feature_r)
#
# node_feature_c = graph.ndata['c']
# print("node_feature_c.shape:", node_feature_c.shape)
# print("node_feature_c:", node_feature_c)
#
#
# node_feature_l = graph.ndata['l']
# print("node_feature_l.shape:", node_feature_l.shape)
# print("node_feature_l:", node_feature_l)
#
#
#
node_feature_x = graph.ndata['x']
# print("graph.ndata['x'].type",type(graph.ndata['x']))
# print("node_feature_x.shape:", node_feature_x.shape)

# print("node_feature_x:\n", node_feature_x[0])




edge_feature_x = graph.edata['x']
print("edge_feature_x.shape:", edge_feature_x.shape)
# print("edge_feature_x:\n", edge_feature_x[0])

# edge_feature_l = graph.edata['l']
# print("edge_feature_l.shape:", edge_feature_l.shape)
# # print("edge_feature_l:", edge_feature_l)
#
# edge_feature_t = graph.edata['t']
# print("edge_feature_t.shape:", edge_feature_t.shape)
# # print("edge_feature_t:", edge_feature_t)
#
# edge_feature_a = graph.edata['a']
# print("edge_feature_a.shape:", edge_feature_a.shape)
# # print("edge_feature_z:", edge_feature_z)
#
# edge_feature_c = graph.edata['c']
# print("edge_feature_c.shape:", edge_feature_c.shape)
# # print("edge_feature_a:", edge_feature_a)

#--------------------------------
# edge_feature_c = graph.edata['c']
# print("edge_feature_c.shape:", edge_feature_c.shape)
# print("edge_feature_c:", edge_feature_c)
#
# edge_feature_l = graph.edata['l']
# print("edge_feature_l.shape:", edge_feature_l.shape)
# print("edge_feature_l:", edge_feature_l)
#
# edge_feature_t = graph.edata['t']
# print("edge_feature_t.shape:", edge_feature_t.shape)
# print("edge_feature_t:", edge_feature_t)




