import torch
import torch.nn as nn
from einops import rearrange, repeat


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


# copy from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class HyperProjector(nn.Module):
    def __init__(
        self, input_dim=13, base_output_channels=10, output_size=(64, 96, 128)
    ):
        super(HyperProjector, self).__init__()
        self.base_output_channels = base_output_channels
        self.output_size = output_size

        self.fc = nn.Linear(input_dim, 512)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # 使用3D卷积来处理深度维度
        self.final_conv = nn.Conv3d(
            64, base_output_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.activation(self.fc(x)).view(batch_size, 512, 1, 1)
        x = self.activation(self.deconv1(x))
        x = self.activation(self.deconv2(x))
        x = self.activation(self.deconv3(x))

        # 调整大小并添加深度维度
        x = nn.functional.interpolate(
            x, size=self.output_size[:2], mode="bilinear", align_corners=False
        )
        x = x.unsqueeze(2).repeat(1, 1, self.output_size[2], 1, 1)

        # 应用3D卷积
        x = self.final_conv(x)
        #
        return x.view(batch_size * self.base_output_channels, *self.output_size)


# class AttentionProjector(nn.Module):
#     """
#     batch_size x 512(Clip-32B) =>(linear, view) batch_size x C x H x W
#
#     => (SpatialSelfAttention) =>  batch_size x C x H x W
#     # 这里需要一个额外的维度来表示时间序列
#
#     => batch_size x output_seq_size(2) x _C x _H x _W(64, 96, 128)
#
#     """


class AttentionProjector(nn.Module):
    """
    Projects a 1D vector (e.g., CLIP embedding) to a 5D tensor (B, S, C, H, W)
    using linear projection, spatial self-attention, and 3D transposed convolutions.

    batch_size x input_dim
    => (Linear, Reshape) batch_size x C_inter x H_inter x W_inter
    => (SpatialSelfAttention) batch_size x C_inter x H_inter x W_inter
    => (Repeat) batch_size x C_inter x output_seq_size x H_inter x W_inter
    => (ConvTranspose3d Upsampling) batch_size x output_channels x output_seq_size x H x W
    => (Rearrange) batch_size x output_seq_size x output_channels x H x W
    """

    def __init__(
        self,
        input_dim=512,  # Dimension of the input vector (e.g., CLIP embedding size)
        output_seq_size=2,  # The desired sequence length in the output
        output_channels=10,  # Number of channels in the output (_C)
        output_h=96,  # Spatial height of the output (_H)
        output_w=128,  # Spatial width of the output (_W)
        intermediate_channels=64,  # Channels after initial projection and before attention
        num_upsample_layers=2,  # Number of 2x spatial upsampling steps
        activation_fn="silu",  # Activation function: "relu" or "silu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_seq_size = output_seq_size
        self.output_channels = output_channels
        self.output_h = output_h
        self.output_w = output_w
        self.intermediate_channels = intermediate_channels
        self.num_upsample_layers = num_upsample_layers

        # Choose activation function
        if activation_fn.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation_fn.lower() == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        # Calculate intermediate spatial size needed before upsampling
        scale_factor = 2**num_upsample_layers
        if output_h % scale_factor != 0 or output_w % scale_factor != 0:
            raise ValueError(
                f"Output H ({output_h}) and W ({output_w}) must be divisible by 2^num_upsample_layers ({scale_factor})"
            )
        self.intermediate_h = output_h // scale_factor
        self.intermediate_w = output_w // scale_factor

        # 1. Initial Projection Layer
        self.initial_projection = nn.Linear(
            input_dim,
            intermediate_channels * self.intermediate_h * self.intermediate_w,
        )

        # 2. Spatial Self-Attention Layer
        self.attention = SpatialSelfAttention(in_channels=intermediate_channels)

        # 3. Upsampling Layers using ConvTranspose3d
        # Input to ConvTranspose3d: (N, C_in, D, H_in, W_in)
        # Our intermediate tensor after repeat will be: (B, C_inter, S, H_inter, W_inter)
        # We want output: (B, C_out, S, H_out, W_out)
        upsample_layers = []
        current_channels = intermediate_channels
        for i in range(num_upsample_layers):
            # Determine output channels for this layer
            # Progressively reduce channels towards the target `output_channels`
            # Example: 64 -> 32 -> 10 (if num_upsample_layers=2, output_channels=10)
            # Or simply keep intermediate channels until the last layer
            # Let's choose the latter for simplicity unless intermediate_channels == output_channels
            if num_upsample_layers == 1:
                out_ch = output_channels
            elif i < num_upsample_layers - 1:
                # Halve channels, but don't go below output_channels if it's large
                out_ch = max(
                    current_channels // 2,
                    output_channels
                    if num_upsample_layers > 2
                    else intermediate_channels // 2,
                )
                out_ch = max(out_ch, 4)  # Prevent channels from becoming too small
            else:  # Last upsampling layer
                out_ch = output_channels

            # Kernel size: (Temporal, Height, Width) - Use 3 for time to mix adjacent sequence elements slightly?
            # Stride: (Temporal, Height, Width) - Use 1 for time, 2 for H/W for upsampling
            # Padding: Adjust to maintain sequence length and achieve target H/W
            # Kernel (3, 4, 4), Stride (1, 2, 2), Padding (1, 1, 1) doubles H, W and keeps D same.
            upsample_layers.append(
                nn.ConvTranspose3d(
                    current_channels,
                    out_ch,
                    kernel_size=(3, 4, 4),  # Kernel 3 in time dim, 4 in spatial dims
                    stride=(1, 2, 2),  # Stride 1 in time, 2 in spatial dims
                    padding=(1, 1, 1),  # Padding 1 in time, 1 in spatial dims
                )
            )
            current_channels = out_ch

            # Add activation after each upsample, except maybe the last one? Often included.
            # Normalization could also be added here (e.g., GroupNorm3d)
            upsample_layers.append(self.activation)
            # Consider adding Normalize3D equivalent here if needed

        self.upsampler = nn.Sequential(*upsample_layers)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input tensor must have shape (batch_size, {self.input_dim}), but got {x.shape}"
            )
        batch_size = x.shape[0]

        # 1. Initial Projection and Reshape
        # batch_size x input_dim -> batch_size x (C_inter * H_inter * W_inter)
        h = self.initial_projection(x)
        h = self.activation(h)

        # Reshape to 4D tensor for spatial attention
        # batch_size x (C_inter * H_inter * W_inter) -> batch_size x C_inter x H_inter x W_inter
        try:
            h = rearrange(
                h,
                "b (c h w) -> b c h w",
                c=self.intermediate_channels,
                h=self.intermediate_h,
                w=self.intermediate_w,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to rearrange tensor with shape {h.shape} into B x {self.intermediate_channels} x {self.intermediate_h} x {self.intermediate_w}. Original error: {e}"
            )

        # 2. Apply Spatial Self-Attention
        # Input: b x C_inter x H_inter x W_inter
        # Output: b x C_inter x H_inter x W_inter
        h = self.attention(h)

        # 3. Introduce Sequence Dimension by Repeating and Rearranging
        # Repeat the spatial feature map for each element in the desired output sequence
        # b x C_inter x H_inter x W_inter -> b x C_inter x output_seq_size x H_inter x W_inter
        # 这里，直接repeat就行了。
        # 因为
        h = repeat(h, "b c h w -> b c s h w", s=self.output_seq_size)
        # Now h has shape: batch_size x intermediate_channels x output_seq_size x intermediate_h x intermediate_w

        # 4. Upsample using 3D Transposed Convolutions
        # Input shape for ConvTranspose3d: (N, C_in, D, H_in, W_in)
        # Our shape is (b, C_inter, seq_size, H_inter, W_inter). This matches.
        # Output shape expected: b x output_channels x output_seq_size x output_h x output_w
        h = self.upsampler(h)

        # 5. Final Rearrangement to put sequence dimension second
        # b x output_channels x output_seq_size x output_h x output_w -> b x output_seq_size x output_channels x output_h x output_w
        h = rearrange(h, "b c s h w -> b s c h w")

        # Final Check (optional, good for debugging)
        expected_shape = (
            batch_size,
            self.output_seq_size,
            self.output_channels,
            self.output_h,
            self.output_w,
        )
        if h.shape != expected_shape:
            print(
                f"Warning: Final shape mismatch in AttentionProjector. Got {h.shape}, expected {expected_shape}"
            )
            # You might need interpolation as a fallback if ConvTranspose3d output size isn't exact
            # h = F.interpolate(h.view(batch_size * self.output_seq_size, self.output_channels, h.shape[-2], h.shape[-1]),
            #                   size=(self.output_h, self.output_w), mode='bilinear', align_corners=False)
            # h = h.view(batch_size, self.output_seq_size, self.output_channels, self.output_h, self.output_w)

        return h


if __name__ == "__main__":
    from loguru import logger

    labels_features = torch.rand(size=(8, 512))
    logger.debug(
        f"labels_features.shape: {labels_features.shape}"
    )  # labels_features.shape: torch.Size([8, 512])
    net = AttentionProjector(output_channels=64, output_h=8, output_w=8)
    out = net(labels_features)
    logger.debug(f"out.shape:{out.shape}")  # ut.shape:torch.Size([8, 2, 10, 96, 128])
