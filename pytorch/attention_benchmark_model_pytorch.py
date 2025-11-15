import torch
import torch.nn as nn


class MHAModel(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x):

        output, _ = self.mha(x, x, x)
        return output


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    seq_length = 128
    d_model = 512

    model = MHAModel(d_model=d_model, n_heads=8)
    x = torch.randn(batch_size, seq_length, d_model)

    torch.onnx.export(
        model,
        x,
        "mha.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes={"input": {0: "B", 1: "T"}, "output": {0: "B", 1: "T"}},
    )
