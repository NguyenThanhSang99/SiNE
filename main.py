from graph import Graph
from model import SiNE, fit_model
import networkx as nx


def main():
    path = "data/data.csv"
    graph = Graph(nx.Graph(), nx.Graph()).read_from_file(path)

    dim1 = 10
    dim2 = 5
    delta = 1
    batch_size = 20
    epochs = 1000
    alpha = 1

    print(graph)

    model = SiNE(graph.__len__(), dim1, dim2)

    fit_model(model, triplets=graph.get_triplets(), delta=delta, batch_size=batch_size,
              epochs=epochs, alpha=alpha)

    print(model.embeddings)


if __name__ == "__main__":
    main()
