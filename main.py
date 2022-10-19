from graph import Graph
from model import SiNE, fit_model
import networkx as nx


def main():
    path = "data/data.csv"
    graph = Graph(nx.Graph(), nx.Graph()).read_from_file(path)

    dim1 = 2
    dim2 = 2
    delta = 1
    batch_size = 20
    epochs = 15
    alpha = 1

    print(graph.__len__())
    model = SiNE(graph.get_triplets(), dim1, dim2)

    fit_model(model, delta=delta, batch_size=batch_size,
              epochs=epochs, alpha=alpha)


if __name__ == "__main__":
    main()
