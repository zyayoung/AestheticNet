from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from lib import AVA

@torch.jit.script
def cross_entropy(x, y, w):
    return -torch.sum(y * x.log(), dim=-1) * w
    
def mse(label, pred):
    return (torch.arange(10, dtype=torch.float, device=pred.device).unsqueeze(0) / 10 * (label-pred)).sum(-1).pow(2).mean()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    run_loss = torch.zeros((1,))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data).softmax(-1)
        weight = target.sum(-1) / target.sum()
        target /= target.sum(-1, keepdim=True)
        loss = cross_entropy(output, target, weight).sum()
        run_loss = 0.9 * run_loss + 0.1 * loss.detach()

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f} RMSE: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), run_loss.item(),
                mse(target, output)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch AVA')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=24, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        AVA('./ava', transform=normalize),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = torchvision.models.mnasnet1_0(pretrained=True, num_classes=10).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "ava_cnn.pt")


if __name__ == '__main__':
    main()
