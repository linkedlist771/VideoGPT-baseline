import torch.nn as nn
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

        return x.view(batch_size * self.base_output_channels, *self.output_size)
