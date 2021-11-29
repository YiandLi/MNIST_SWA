from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import matplotlib.pyplot as plt


# Adjust the model to get a higher performance
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1152, 32)
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # plt.figure()
    # pic = None
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx in (1, 2, 3, 4, 5):
        #     if batch_idx == 1:
        #         pic = data[0, 0, :, :]
        #     else:
        #         pic = torch.cat((pic, data[0, 0, :, :]), dim=1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        # F.cross_entropy: softmax-log-compute
        loss = F.cross_entropy(output, target)
        # Calculate gradients
        loss.backward()
        # Optimize the parameters according to the calculated gradients
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
    # plt.imshow(pic.cpu(), cmap='gray')
    # plt.show()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    false_pred = []
    false_pic, right_pic = [], []
    right_pic_num, wrong_pic_num = 0, 0
    pred_confidence, target_confidence, right_confidence = [], [], []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # update false_pictures
            if not (wrong_pic_num >= 5 and right_pic_num >= 5):
                (false_pic, right_pic, false_pred), \
                (pred_confidence, target_confidence, right_confidence), \
                wrong_pic_num, right_pic_num = show_figure(target,
                                                           output,
                                                           data,
                                                           false_pred,
                                                           false_pic,
                                                           right_pic,
                                                           wrong_pic_num,
                                                           right_pic_num,
                                                           pred_confidence,
                                                           target_confidence,
                                                           right_confidence)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    
    accuray = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuray))
    
    return (false_pic, right_pic, false_pred), (pred_confidence, target_confidence, right_confidence), accuray


def show_figure(target,
                output,
                data,
                false_pred,
                false_pic,
                right_pic,
                false_pic_num,
                right_pic_num,
                pred_confidence,
                target_confidence,
                right_confidence):
    # predict_output
    predict = torch.argmax(output, dim=1)
    false_index = predict != target
    # get softmax confidence
    softmax_result = torch.softmax(output, dim=1)
    
    for i in range(len(false_index)):
        # save right-classified
        if (right_pic_num < 5) and (not false_index[i]):
            if right_pic == []:
                right_pic = data[i, 0, :, :]
            else:
                right_pic = torch.cat((right_pic, data[i, 0, :, :]), dim=1)
            right_confidence.append(softmax_result[i][predict[i]])
            right_pic_num += 1
        
        # save false-classified
        if false_pic_num < 5 and false_index[i]:
            if false_pic == []:
                false_pic = data[i, 0, :, :]
            else:
                false_pic = torch.cat((false_pic, data[i, 0, :, :]), dim=1)
            false_pred.append(predict[i])
            pred_confidence.append(softmax_result[i][predict[i]])
            target_confidence.append(softmax_result[i][target[i]])
            false_pic_num += 1
    
    return (false_pic, right_pic, false_pred), (
        pred_confidence, target_confidence, right_confidence), false_pic_num, right_pic_num


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--swa', action='store_true', default=False,
                        help='use swa algorithm to get the average parameter')
    parser.add_argument('--seed', type=int, default=0.5, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # batch_size is a crucial hyper-parameter
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        # Adjust num worker and pin memory according to your computer performance
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    # Normalize the input (black and white image)
    # 图像预处理包：一般用Compose把多个步骤整合到一起
    train_transform = transforms.Compose([
        # 先crop，再resize到指定尺寸。
        # transforms.RandomResizedCrop(28, scale=(0.8, 1)),
        # transforms.RandomHorizontalFlip(),
        # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W)
        #                           in the range [0.0,1.0]
        transforms.ToTensor(),
        # Normalized an tensor image with mean and standard deviation，这样处理后的数据符合标准正态分布，即均值为0，标准差为1。
        # x = (x - mean(x))/std(x)
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W)
        #                           in the range [0.0,1.0]（
        transforms.ToTensor(),
        # Normalized an tensor image with mean and standard deviation
        # x = (x - mean(x))/std(x)
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Make train dataset split
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                              transform=train_transform)
    # Make test dataset split
    dataset2 = datasets.MNIST('./data', train=False,
                              transform=test_transform)
    
    # Convert the dataset to dataloader, including train_kwargs and test_kwargs
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    # Put the model on the GPU or CPU
    model = Net().to(device)
    print("Model has {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # SGD
    
    # Create a schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    
    ## use swa
    if args.swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy="linear", anneal_epochs=args.epochs, swa_lr=0.5)
        swa_start = int(args.epochs * 0.6)
    
    # Begin training and testing
    best_acc = -9
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        print("==========")
        train(args, model, device, train_loader, optimizer, epoch)
        (false_pic, right_pic, false_pred), (pred_confidence, target_confidence, right_confidence), acc = test(
            model,
            device,
            test_loader)
        
        # Save the model
        if best_acc < acc:
            if args.save_model:
                torch.save(model.state_dict(), "mnist_cnn.pt")
                print("model saved.")
            best_acc = acc
            best_epoch = epoch
        if args.swa and epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
    
    if args.swa:
        print("==== Using SWA model to predict:")
        torch.optim.swa_utils.update_bn(test_loader, swa_model)
        (false_pic, right_pic, false_pred), (pred_confidence, target_confidence, right_confidence), acc = test(
            swa_model,
            device,
            test_loader)
    
    # plot false and right-classified pics
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(false_pic.cpu(), cmap='gray')
    plt.title("false pics")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 1, 2)
    plt.imshow(right_pic.cpu(), cmap='gray')
    plt.title("right pics")
    plt.xticks([])
    plt.yticks([])
    print("best accuracy {} at epoch {} \n "
          "False classfied: \n"
          "predict label: {} \n"
          "predict softmax confidence: {} \n "
          "target softmax confidence: {} \n "
          "Right classfied: \n"
          "right softmax confidence: {}".format(best_acc, best_epoch, [i.item() for i in false_pred], pred_confidence,
                                                target_confidence, right_confidence))
    plt.show()


if __name__ == '__main__':
    main()
