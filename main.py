from src.pnd_ocsvm import pnd_ocsvm

if __name__ == '__main__':
    dataset_path = "./dataset/fraud.npz"
    nodes_list = [40]
    neighbors_list = [2]
    #nodes_list = [20, 30, 40, 50, 60, 70, 80]
    #neighbors_list = [2, 4, 6, 8]
    for nodes in nodes_list:
        for neighbors in neighbors_list:
            pnd_ocsvm(dataset_path, nodes, neighbors)