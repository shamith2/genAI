# FSRCNN Model (PyTorch)
# from Accelerating the Super-Resolution
# Convolutional Neural Network (Chao Dong et al.) https://arxiv.org/pdf/1608.00367.pdf

import collections

import torch


class FSRCNN(torch.nn.Module):
    """
    FSRCNN model
    """

    def __init__(
            self,
            d: int,
            s: int,
            m: int,
            upscale_factor: int,
            input_channels: int = 3,
    ) -> None:
        super(FSRCNN, self).__init__()

        # model layers
        self.conv_layers = collections.OrderedDict()
        self.deconv_layers = collections.OrderedDict()

        # part 1 (Feature Extraction) := Conv(f = 5, n = d, c = input_channels)
        self.conv_layers['feature_extraction'] = self.conv_kxk(in_channels=input_channels, out_channels=d,
                                                               kernel_size=5, stride=1)
        self.init_weight(self.conv_layers['feature_extraction'].weight)
        self.conv_layers['fe_prelu'] = self.prelu(num_parameters=d)

        # part 2 (Shrinking) := Conv(f = 1, n = s, c = d)
        self.conv_layers['shrinking'] = self.conv_kxk(in_channels=d, out_channels=s, kernel_size=1, stride=1)
        self.init_weight(self.conv_layers['shrinking'].weight)
        self.conv_layers['sh_prelu'] = self.prelu(num_parameters=s)

        # part 3 (Non-linear mapping) := m x Conv(f = 3, n = s, c = s)
        for i in range(m):
            self.conv_layers['non_linear_map_' + str(i)] = self.conv_kxk(in_channels=s, out_channels=s,
                                                                         kernel_size=3, stride=1)
            self.init_weight(self.conv_layers['non_linear_map_' + str(i)].weight)
            self.conv_layers['nlm_prelu_' + str(i)] = self.prelu(num_parameters=s)

        # part 4 (Expanding) := Conv(f = 1, n = d, c = s)
        self.conv_layers['expanding'] = self.conv_kxk(in_channels=s, out_channels=d, kernel_size=1, stride=1)
        self.init_weight(self.conv_layers['expanding'].weight)
        self.conv_layers['ex_prelu'] = self.prelu(num_parameters=d)

        # part 5 (Deconvolution) := Conv(f = 9, n = input_channels, c = d)
        self.deconv_layers['deconvolution'] = self.deconv_kxk(in_channels=d, out_channels=input_channels, kernel_size=9,
                                                              stride=upscale_factor, output_padding=upscale_factor-1)
        # initialize deconvolution layer weight with normal distribution
        torch.nn.init.normal_(self.deconv_layers['deconvolution'].weight, mean=0.0, std=0.001)
        self.deconv_layers['deconv_prelu'] = self.prelu(num_parameters=input_channels)

        # fsrcnn model
        self.conv_model = torch.nn.Sequential(self.conv_layers)
        self.deconv_model = torch.nn.Sequential(self.deconv_layers)

    def forward(self, input_image: torch.Tensor):
        conv_output = self.conv_model(input_image)
        output_image = self.deconv_model(conv_output)

        return output_image

    def conv_kxk(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: str = "same",
            groups: int = 1,
            dilation: int = 1
    ) -> torch.nn.Conv2d:
        """k x k convolution with same padding where k is the kernel size"""
        return torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            dtype=torch.float32,
        )

    def deconv_kxk(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int = 4,
            output_padding: int = 0,
            groups: int = 1,
            dilation: int = 1
    ) -> torch.nn.ConvTranspose2d:
        """
        k x k (partial) de-convolution with the required padding to get the upscaled output
        where k is the kernel size
        """
        return torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=False,
            dilation=dilation,
            dtype=torch.float32,
        )

    def prelu(
            self,
            num_parameters: int,
            init: float = 0.25
    ) -> torch.nn.PReLU:
        """PReLU activation with learnable parameter for each channel of input to this layer"""
        return torch.nn.PReLU(
            num_parameters=num_parameters,
            init=init,
            dtype=torch.float32,
        )

    def init_weight(
            self,
            weight: torch.Tensor,
            a: float = 0.25,
            mode: str = 'fan_out',
            nonlinearity: str = 'leaky_relu'  # gain calculation is same for leaky relu and prelu
    ) -> torch.nn.init.kaiming_normal_:
        """ He Normal Initialization for conv2d layer weights using in-place transformation"""
        return torch.nn.init.kaiming_normal_(
            weight,
            a,
            mode,
            nonlinearity,
        )

    def num_parameters(
            self
    ) -> tuple[int, int]:
        """Number of trainable, total parameters and total parameters without prelu parameters in the model"""
        conv_t_params = sum(p.numel() for p in self.conv_model.parameters() if p.requires_grad)
        conv_params = sum(p.numel() for p in self.conv_model.parameters())

        deconv_t_params = sum(p.numel() for p in self.deconv_model.parameters() if p.requires_grad)
        deconv_params = sum(p.numel() for p in self.deconv_model.parameters())

        conv_prt_params = sum(p.numel() for n, p in self.conv_model.named_parameters() if 'prelu' in n)
        deconv_prt_params = sum(p.numel() for n, p in self.deconv_model.named_parameters() if 'prelu' in n)

        # compare number of total parameters without PReLU parameters with the number of parameters
        # given in table 1 of research paper
        # assert conv_params + deconv_params - (conv_prt_params + deconv_prt_params) == 12464

        return conv_t_params + deconv_t_params, conv_params + deconv_params


if __name__ == '__main__':
    model = FSRCNN(d=56, s=12, m=4, upscale_factor=4)

    print(model)

    out = model(torch.rand(2, 3, 162, 162))

    print(out.shape)

    print("\n(Trainable Parameters, Total Parameters): {}".format(model.num_parameters()))
