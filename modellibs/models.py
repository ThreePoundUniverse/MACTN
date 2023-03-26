import torch
import torch.nn as nn
from utils import MODEL_REGISTOR, MODEL_REGISTOR_MT, timer_wrap
from modellibs import *
from modellibs.modules import *


def get_model(model_name, input_shape, output_shape, *args, **kwargs):
    if model_name in MODEL_REGISTOR.registered_names():
        return MODEL_REGISTOR.get(model_name)(input_shape, output_shape, *args, **kwargs)
    else:
        raise NotImplementedError


def get_model_MT(model_name, *args, **kwargs):
    if model_name in MODEL_REGISTOR_MT.registered_names():
        return MODEL_REGISTOR_MT.get(model_name)(*args, **kwargs)
    else:
        raise NotImplementedError


class BaseModel(nn.Module):
    def __init__(self, input_shape=None, output_shape=None):
        super(BaseModel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __build_pseudo_input(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        temp_x_ = torch.rand(input_shape)
        temp_x = temp_x_.unsqueeze(0)
        return temp_x

    def get_tensor_shape(self, forward_func, input_shape=None):
        pseudo_x = self.__build_pseudo_input(input_shape)
        pseudo_y = forward_func(pseudo_x)
        return pseudo_y.shape


@MODEL_REGISTOR.register()
class MACTN(BaseModel):
    def __init__(self, input_shape=(30, 1750), output_shape: int = 9,
                 dropoutRate: float = 0.5, kernLength: int = 15):
        super().__init__()
        chans: int = input_shape[0]
        samples: int = input_shape[-1]
        F1 = chans * 2
        F2 = F1 * 2
        downSample_1, downSample_2 = 4, 5

        self.depth = [1, 2]
        self.stage0 = nn.Sequential(
            nn.Conv1d(chans, F2, kernLength, groups=chans),
        )
        self.stage1 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1.append(
                nn.Sequential(
                    nn.Conv1d(F2, F2, kernLength, groups=F2),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )
        self.stage2 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2.append(
                nn.Sequential(
                    SeparableConv1d(F2, F2, kernel_size=kernLength, padding=kernLength // 2),
                    SeparableConv1d(F2, F2, kernel_size=kernLength, padding=kernLength // 4),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(dropoutRate)
                )
            )

        self.merge_s1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        self.merge_s2 = nn.Sequential(
            nn.AvgPool1d(downSample_2),
            SKAttention1D(F2, reduction=4),
            Permute([0, 2, 1])
        )


        # transformer
        seq_len = int((samples-30) // (downSample_1 * downSample_2))
        embed_dim = F2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        self.transformer = TransformerVit(dim=embed_dim, depth=6, heads=8, dim_head=256, mlp_dim=128, dropout=0)
        self.layernorm = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, output_shape),
        )

    # @timer_wrap
    def forward(self, x: torch.Tensor):
        seq_embed = self.forward_embed(x)
        batch_size, seq_len, embed_dim = seq_embed.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, seq_embed), dim=1)
        embeddings += self.pos_embedding
        tr_output = self.transformer(embeddings)
        sequence_output = self.layernorm(tr_output)
        cls_token = sequence_output[:, 0, :]
        y2 = self.classifier(cls_token)

        return y2

    def forward_embed(self, x):
        x = self.stage0(x)
        for stage1 in self.stage1:
            x = stage1(x)
        x = self.merge_s1(x)
        for stage2 in self.stage2:
            x = stage2(x)
        x = self.merge_s2(x)
        return x