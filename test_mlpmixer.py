from model_structures import *
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import logging
import pytz
import os
from datetime import datetime

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO, # 记录级别为INFO
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "log.txt"))]
    )
    logger = logging.getLogger(__name__)
    return logger

device = 'cuda:3'

def eval(net, testloader, trainloader, device, logger):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logger.info(f'Accuracy of the network on the {total} train images: {100 * correct // total} %')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logger.info(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 224
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


model = MLPMixerClassifier(in_channels=3, image_size=32, patch_size=4, num_classes=10,
                 dim=768, depth=12, token_dim=196, channel_dim=3072).to(device)

logging_dir = './results'
now = datetime.now().astimezone(pytz.timezone('US/Pacific')).strftime("%Y%m%d-%H%M")
os.makedirs(os.path.join(logging_dir, f"{now}-mlpmixer"))
logging_dir = os.path.join(logging_dir, f"{now}-mlpmixer")

logger = create_logger(logging_dir)
logger.info('Arguments:')
logger.info(dict(in_channels=3, image_size=32, patch_size=4, num_classes=10,
                 dim=768, depth=12, token_dim=196, channel_dim=3072))
logger.info(f'batch_size: {batch_size}')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
log_step = 50

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_step == 0 and i != 0:    # print every 2000 mini-batches
            logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / log_step:.3f}')
            running_loss = 0.0
    print(f'Epoch: {epoch}')
    eval(model, testloader, trainloader=trainloader, device=device, logger=logger)
    if epoch % 1 == 0 and epoch != 0:
        checkpoint = {
                    "model": model.module.state_dict(),
                    "epoch":epoch+1,
                    #"args": args,
                    "experiment_dir":logging_dir,
                    #"train_steps": train_steps,
                }
        checkpoint_dir = os.path.join(logging_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path_fin = os.path.join(checkpoint_dir, f'{epoch:07d}.pt')
        torch.save(checkpoint, checkpoint_path_fin)
        logger.info(f"Saved checkpoint to {checkpoint_path_fin}")

print('Finished Training')