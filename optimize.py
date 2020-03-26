import argparse
import torch
from torch import nn
import torchvision
import torch.onnx
import torch.optim as optim

import cv2
import numpy as np

@torch.jit.script
def blend(im0, im1, alpha):
    # alpha = nn.functional.interpolate(alpha, im0.shape[2:], mode="bilinear", align_corners=True)
    return im0 * alpha + im1 * (1 - alpha)


@torch.jit.script
def brightness(im, factor):
    return blend(im, torch.zeros_like(im), factor)


@torch.jit.script
def contrast(im, factor):
    return blend(im, im.mean(), factor)


@torch.jit.script
def saturation(im, factor):
    return blend(im, im.mean(1).unsqueeze(1), factor)


@torch.jit.script
def gamma(im, gamma):
    # gamma = nn.functional.interpolate(gamma, im.shape[2:], mode="bilinear", align_corners=True)
    return 255 * (im.clamp(0.1, 254.9) / 255).pow(gamma)


class Model(nn.Module):
    def __init__(self, input_file, weights_file):
        self.device = 'cuda'
        super().__init__()

        self.im_ori = cv2.imread(input_file)[..., [2, 1, 0]]
        self.im_ori = cv2.resize(self.im_ori, (1920, 1280), interpolation=cv2.INTER_CUBIC)
        self.im_ori = np.int64(self.im_ori)
        self.im = torch.tensor(self.im_ori, dtype=torch.float, device=self.device).permute(2, 0, 1).unsqueeze(0)
        
        # Make original image trainable
        # self.im = nn.Parameter(self.im)

        # Create Tunable Post-processing Parameters
        self.gamma = nn.Parameter(torch.zeros((1, 3, 1, 1)))
        self.contrast = nn.Parameter(torch.zeros((1,))+0.1)
        self.brightness = nn.Parameter(torch.zeros((1,))+0.1)
        self.saturation = nn.Parameter(torch.zeros((1,))+0.1)
        self.channel_mix = nn.Parameter(torch.eye(3).view(3, 3, 1, 1))
        self.channel_bias = nn.Parameter(torch.zeros((3,)))

        self.model = torchvision.models.mnasnet1_0(num_classes=10)
        state_dict = torch.load(weights_file)
        
        self.model.load_state_dict(state_dict)
        for p in self.model.parameters():
            p.requires_grad = False

        self.to(self.device)
    
    def forward(self):
        im = self.post_procress()
        crop = torch.randint(8,(4,))
        im = nn.functional.interpolate(
            im[..., crop[0]:-crop[1]-1, crop[2]:-crop[3]-1],
            (224, 224),
            mode="bilinear"
        )

        im = im / 255.
        im[:, 0] -= 0.485
        im[:, 1] -= 0.456
        im[:, 2] -= 0.406
        im[:, 0] /= 0.229
        im[:, 1] /= 0.224
        im[:, 2] /= 0.225

        return self.model(im).softmax(-1)
    
    def post_procress(self):
        im = self.im
        t_im = gamma(im, self.gamma.exp())
        t_im = brightness(t_im, self.brightness.exp())
        # mix = nn.functional.interpolate(self.channel_mix, im.shape[2:], mode="bilinear", align_corners=True)
        # t_im = (t_im * mix).sum(0, keepdim=True)
        t_im = nn.functional.conv2d(t_im, self.channel_mix, bias=self.channel_bias)
        t_im = contrast(t_im, self.contrast.exp())
        t_im = saturation(t_im, self.saturation.exp())
        return t_im


def main():
    # Image Optimization settings
    parser = argparse.ArgumentParser(description='PyTorch AVA')
    parser.add_argument('input', type=str,
                        help='input file name')
    parser.add_argument('output', type=str,
                        help='output file name')
    parser.add_argument('--weights', type=str, default="ava_cnn.pt")
    parser.add_argument('--iters', type=int, default=500, metavar='N',
                        help='number of iters to optimize (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                        help='Momentum (default: 0.1)')
    args = parser.parse_args()

    # Buile model and optimizer
    model = Model(args.input, args.weights)
    model.train(False)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for _ in range(args.iters):
        optimizer.zero_grad()
        pred = model()[0]
        mean = (torch.arange(10, dtype=torch.float, device=pred.device) * pred).sum()

        (-mean).backward()
        # (model.channel_mix.std([0, 1], keepdim=True).sum()).backward()
        optimizer.step()
        if torch.rand((1,)) > 0.95:
            im = model.post_procress().squeeze(0).detach().clamp(0, 255).permute(1, 2, 0).cpu()
            cv2.imwrite(args.output, np.uint8(im)[..., [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("Mean: {:.2f}".format(mean.item()), end='\r')
    im = model.post_procress().squeeze(0).detach().clamp(0, 255).permute(1, 2, 0).cpu()
    cv2.imwrite(args.output, np.uint8(im)[..., [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print("\nDone!")

if __name__ == '__main__':
    main()
