# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import torch
# from torch import nn
# from torch.nn import functional as F
import jittor as jt
from jittor import nn
import numpy as np
import pdb

from typing import List, Optional, Tuple, Type

from .common import LayerNorm2d


class MaskDecoderCLS(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        image_embedding_size: Tuple[int, int],
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.image_embedding_size = image_embedding_size
        # self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)

        self.num_multimask_outputs = num_multimask_outputs

        # self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(transformer_dim // 4),
        #     activation(),
        #     nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        #     activation(),
        # )
        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_mask_tokens)
        #     ]
        # )

        # self.iou_prediction_head = MLP(
        #     transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        # )

        self.last_layer =  nn.ModuleList(
            [
                nn.Conv2d(transformer_dim, transformer_dim //2 , kernel_size=3, stride=1, padding=1),
                activation(),
                nn.Conv2d(transformer_dim // 2, transformer_dim // 4, kernel_size=3, stride=1, padding=1),
                activation(),
                nn.Conv2d(transformer_dim // 4, num_multimask_outputs + 1, 1, 1)
            ]
        )
        
        # nn.init.kaiming_normal_(self.last_layer.weight,
        #                                 mode='fan_out',
        #                                 nonlinearity='relu')
        
        for m in self.last_layer:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    # def forward(
    #     self,
    #     image_embeddings: torch.Tensor,
    #     image_pe: torch.Tensor,
    #     sparse_prompt_embeddings: torch.Tensor,
    #     dense_prompt_embeddings: torch.Tensor,
    #     multimask_output: bool,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    def execute(
        self,
        image_embeddings: jt.array,
        # image_pe: jt.array,
        multimask_output: bool,
    ) -> Tuple[jt.array, jt.array]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        # 由于暂时没有prompts，只用一个最简单的nn.Conv2D来生成51类结果

        return self.last_layer(image_embeddings)

        # masks, iou_pred = self.predict_masks(
        #     image_embeddings=image_embeddings,
        #     image_pe=image_pe,
        # )

        # # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # # Prepare output
        # return masks, iou_pred


    # def predict_masks(
    #     self,
    #     image_embeddings: torch.Tensor,
    #     image_pe: torch.Tensor,
    #     sparse_prompt_embeddings: torch.Tensor,
    #     dense_prompt_embeddings: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    def predict_masks(
        self,
        image_embeddings: jt.array,
        image_pe: jt.array,
        sparse_prompt_embeddings: jt.array,
        dense_prompt_embeddings: jt.array,
    ) -> Tuple[jt.array, jt.array]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = jt.contrib.concat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        tokens = jt.contrib.concat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = jt.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        pos_src = jt.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        # hyper_in_list: List[torch.Tensor] = []
        hyper_in_list: List[jt.array] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # hyper_in = torch.stack(hyper_in_list, dim=1)
        hyper_in = jt.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    
    # # def get_dense_pe(self) -> torch.Tensor:
    # def get_dense_pe(self) -> jt.array:
    #     """
    #     Returns the positional encoding used to encode point prompts,
    #     applied to a dense set of points the shape of the image encoding.

    #     Returns:
    #         torch.Tensor: Positional encoding with shape
    #         1x(embed_dim)x(embedding_h)x(embedding_w)
    #     """
    #     return self.pe_layer(self.image_embedding_size).unsqueeze(0)


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        )
        self.sigmoid_output = sigmoid_output

    # def forward(self, x):
    #     for i, layer in enumerate(self.layers):
    #         x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
    #     if self.sigmoid_output:
    #         x = F.sigmoid(x)
    #     return x
    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = nn.sigmoid(x)
        return x


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # self.register_buffer(
        #     "positional_encoding_gaussian_matrix",
        #     scale * torch.randn((2, num_pos_feats)),
        # )
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * jt.randn((2, num_pos_feats)),
        )

    # def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
    def _pe_encoding(self, coords: jt.array) -> jt.array:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        # return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        return jt.contrib.concat([jt.sin(coords), jt.cos(coords)], dim=-1)

    # def forward(self, size: Tuple[int, int]) -> torch.Tensor:
    def execute(self, size: Tuple[int, int]) -> jt.array:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        # device: Any = self.positional_encoding_gaussian_matrix.device
        # grid = torch.ones((h, w), device=device, dtype=torch.float32)
        grid = jt.ones((h, w), dtype=jt.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        # pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        pe = self._pe_encoding(jt.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    # def forward_with_coords(
    #     self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    # ) -> torch.Tensor:
    def forward_with_coords(
        self, coords_input: jt.array, image_size: Tuple[int, int]
    ) -> jt.array:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(jt.float32))  # B x N x C
