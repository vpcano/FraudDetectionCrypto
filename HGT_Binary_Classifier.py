import torch
from torch.nn import Sigmoid, LeakyReLU, ModuleDict
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class HGT_BinaryClassifier(torch.nn.Module):
    """
    Utiliza el modelo HGT para realizar una clasificación binaria sobre
    todos los nodos de todos los tipos en el grafo, además de guardar
    embeddings para cada nodo.
    """
    def __init__(self, embedding_dim, metadata, num_heads, num_layers):
        super().__init__()
        self.metadata = metadata

        # Se podría usar HeteroDictLinear pero no permite funciones de activación
        self.in_lin_dict = ModuleDict()
        self.out_lin_dict = ModuleDict()
        self.l = LeakyReLU()
        for node_type in self.metadata[0]:   # Node types
            self.in_lin_dict[node_type] = Linear(-1, embedding_dim)
            self.out_lin_dict[node_type] = Linear(embedding_dim, 1)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(embedding_dim, embedding_dim, self.metadata,
                           num_heads)
            self.convs.append(conv)


    def forward(self, x_dict, edge_index_dict):
        """
        Primero se aplica una transformación lineal seguida de la función de
        activación LeakyReLU para cada tipo de nodo, con el objetivo de que
        todos los tipos trabajen en las mismas dimensiones.
        """
        x_dict = {
            node_type: self.l(self.in_lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        """
        Se aplican todas las capas convolucionales del modelo HGT
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: self.l(x_dict[node_type])
                for node_type in x_dict.keys()
            }


        """
        Se aplica una última transformación lineal para obtener un único
        valor (Logit) de salida.
        """
        x_dict = {
            node_type: self.out_lin_dict[node_type](x).t()[0]
            for node_type, x in x_dict.items()
        }

        return x_dict

    def reset_parameters(self):
        for node_type in self.metadata[0]:
            self.in_lin_dict[node_type].reset_parameters()
            self.out_lin_dict[node_type].reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()
