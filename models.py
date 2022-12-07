from imports import *


def weight_init(layer, first_layer):
    with torch.no_grad():
        if first_layer:
            if hasattr(layer, "weight"):
                num_inputs = layer.weight.size(-1)

                layer.weight.uniform_(-1 / num_inputs, 1 / num_inputs)
        else:
            if hasattr(layer, "weight"):
                num_inputs = layer.weight.size(-1)

                layer.weight.uniform_(-np.sqrt(6/num_inputs) / 30, np.sqrt(6/num_inputs) / 30)


class GINR(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, bn=False, skip=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.model = nn.ModuleList()

        in_dim = self.input_dim
        out_dim = self.hidden_dim
        for i in range(num_layers):
            layer = nn.Linear(in_dim, out_dim)

            weight_init(layer, i == 0)

            self.model.append(layer)

            if i < self.num_layers - 1:
                self.model.append(nn.ReLU())

                if bn:
                    self.model.append(nn.LayerNorm(out_dim))

            # add a skip connection to the middle of the model
            in_dim = hidden_dim
            if (i + 1) == int(np.ceil(self.num_layers/2)) and skip:
                self.skip_at = len(self.model)
                in_dim += input_dim

            # reached last layer
            out_dim = hidden_dim 
            if (i + 1) == self.num_layers - 1:
                out_dim = output_dim

    def forward(self, x):
        # feedforward through MLP
        x_in = x
        for i, layer in enumerate(self.model):
            if i == self.skip_at:
                x = torch.cat([x, x_in], dim=-1)

            x = layer(x)

        return x
            
