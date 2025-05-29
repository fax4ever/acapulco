"""
@article{han2022g,
  title={G-Mixup: Graph Data Augmentation for Graph Classification},
  author={Han, Xiaotian and Jiang, Zhimeng and Liu, Ninghao and Hu, Xia},
  journal={arXiv preprint arXiv:2202.07179},
  year={2022}
}
"""
import random
import numpy as np
from src.mix_up_utils import split_class_graphs, align_graphs, universal_svd, stat_graph, two_graphons_mixup
import logging

def mix_up(dataset, aug_ratio=0.15, aug_num=10, lam_range=(0.005, 0.01)):
    original_size = len(dataset)

    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(dataset)
    logging.info(f"avg num nodes of training graphs: { avg_num_nodes }")
    logging.info(f"avg num edges of training graphs: { avg_num_edges }")
    logging.info(f"avg density of training graphs: { avg_density }")
    logging.info(f"median num nodes of training graphs: { median_num_nodes }")
    logging.info(f"median num edges of training graphs: { median_num_edges }")
    logging.info(f"median density of training graphs: { median_density }")

    resolution = int(median_num_nodes)

    class_graphs = split_class_graphs(dataset)
    graphons = []
    for label, graphs in class_graphs:

        logging.info(f"label: {label}, num_graphs:{len(graphs)}" )
        align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
            graphs, padding=True, N=resolution)
        logging.info(f"aligned graph {align_graphs_list[0].shape}" )

        graphon = universal_svd(align_graphs_list, threshold=0.2)
        graphons.append((label, graphon))


    for label, graphon in graphons:
        logging.info(f"graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}")

    num_sample = int( original_size * aug_ratio / aug_num )
    lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(aug_num,))

    new_graph = []
    for lam in lam_list:
        logging.info( f"lam: {lam}" )
        logging.info(f"num_sample: {num_sample}")
        two_graphons = random.sample(graphons, 2)
        new_graph += two_graphons_mixup(two_graphons, la=lam, num_sample=num_sample)
        logging.info(f"label: {new_graph[-1].y}")

    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(new_graph)
    logging.info(f"avg num nodes of new graphs: { avg_num_nodes }")
    logging.info(f"avg num edges of new graphs: { avg_num_edges }")
    logging.info(f"avg density of new graphs: { avg_density }")
    logging.info(f"median num nodes of new graphs: { median_num_nodes }")
    logging.info(f"median num edges of new graphs: { median_num_edges }")
    logging.info(f"median density of new graphs: { median_density }")

    logging.info( f"real aug ratio: {len( new_graph ) / original_size }" )
    result = new_graph + list(dataset)
    random.shuffle(result)
    return result
