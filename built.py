import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
def generate_graph(adjacency_matrix):
    G = nx.Graph()
    num_nodes = len(adjacency_matrix)

    # 添加节点
    G.add_nodes_from(range(num_nodes))

    # 添加边和权重
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            weight = adjacency_matrix[i][j]
            if weight != 0:
                G.add_edge(i, j, weight=weight)

    return G

def draw_pitures(m1,m2,i1,i2,r):
    # 创建两个矩阵
    matrix1 = m1

    matrix2 = m2

    # 创建两个图对象
    G1 = generate_graph(matrix1)
    G2 = generate_graph(matrix2)

    # 绘制图形
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    #
    # ax1.set_title('Matrix 1')
    # nx.draw(G1, with_labels=True, node_color='lightblue', node_size=500, ax=ax1)
    #
    # ax2.set_title('Matrix 2')
    # nx.draw(G2, with_labels=True, node_color='lightblue', node_size=500, ax=ax2)
    #
    # # 显示图形
    # plt.show()
    graph1 = G1
    graph2 = G2

    # 绘制图形
    plt.figure(figsize=(10, 4))  # 设置图形的大小
    plt.subplot(121)  # 子图1
    pos1 = nx.spring_layout(graph1)
    edge_labels1 = nx.get_edge_attributes(graph1, 'weight')
    nx.draw_networkx(graph1, pos1)
    nx.draw_networkx_edge_labels(graph1, pos1, edge_labels=edge_labels1)
    plt.title('Graph 1')

    plt.subplot(122)  # 子图2
    pos2 = nx.spring_layout(graph2)
    edge_labels2 = nx.get_edge_attributes(graph2, 'weight')
    nx.draw_networkx(graph2, pos2)
    nx.draw_networkx_edge_labels(graph2, pos2, edge_labels=edge_labels2)
    plt.title('Graph 2')

    plt.tight_layout()  # 调整子图的布局
    plt.savefig(f'graph_{i1}_{i2}_{r}.png')
    plt.cla()
    # plt.show()

