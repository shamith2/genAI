# training FSRCNN model

# references from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

from datetime import datetime
import os

import torch
from torch.utils import tensorboard
import torchvision

from model import FSRCNN
from dataset import DIV2K


def train(upscale_factor: int = 4):
    """Function used for training FSRCNN"""

    # dataset
    root_dir = os.path.join(os.path.expanduser('~'), 'IPU', 'gen_ai', 'fsrcnn', 'datasets', 'div2k')
    training_set = DIV2K(root_dir=root_dir, upscale_factor=upscale_factor, train_mode='train', in_channels=3)
    validation_set = DIV2K(root_dir=root_dir, upscale_factor=upscale_factor, train_mode='val', in_channels=3)

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=2,
        shuffle=True,
        collate_fn=training_set.collate_fn
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=2,
        shuffle=False,
        collate_fn=validation_set.collate_fn
    )

    print("Size of Training Dataset: {} and Validation Dataset: {}\n".format(len(training_set), len(validation_set)))

    # model
    model = FSRCNN(d=56, s=12, m=4, upscale_factor=upscale_factor)

    # loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # optimizer
    params = []

    for name, layer_weights in model.named_parameters():
        if layer_weights.requires_grad:
            if "deconv" in name:
                params.append({'params': layer_weights, 'lr': 1e-4})

            else:
                params.append({'params': layer_weights})

    optimizer = torch.optim.RMSprop(params=params, lr=1e-3)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = tensorboard.SummaryWriter(os.path.join('runs', 'fsrcnn_training_{}'.format(timestamp)))

    num_epoch = 0
    total_epochs = 5

    # training data has to be >= batch_size * sub_batch_size
    sub_batch_size = 100

    best_val_loss = 1e+6

    for epoch in range(total_epochs):
        print('EPOCH {}:'.format(num_epoch + 1))

        # make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        batch_train_loss = 0.0
        avg_train_loss = 0.0
        running_val_loss = 0.0

        for i, (train_images, train_labels) in enumerate(training_loader):
            # zero the gradients for every batch
            optimizer.zero_grad()

            # predictions for training data batch
            train_outputs = model(train_images)

            # compute the loss and its gradients
            train_loss = loss_fn(train_outputs, train_labels)
            train_loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            batch_train_loss += train_loss.item()

            if i % sub_batch_size == sub_batch_size - 1:
                # loss per batch
                avg_train_loss += batch_train_loss / sub_batch_size

                print('Batch {} Loss: {}'.format(i + 1, avg_train_loss))

                tb_x = num_epoch * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/Train', avg_train_loss, tb_x)

                batch_train_loss = 0.0

        # set model to eval mode
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, (val_images, val_labels) in enumerate(validation_loader):
                # predictions for validation data batch
                val_outputs = model(val_images)

                # compute loss
                val_loss = loss_fn(val_outputs, val_labels)
                running_val_loss += val_loss

        avg_val_loss = running_val_loss / (i + 1)

        print('LOSS: train {} valid {}\n'.format(avg_train_loss, avg_val_loss))

        # Log the running loss averaged per batch
        # for both training and validation
        tb_writer.add_scalars('Training vs. Validation Loss',
                              {'Training': avg_train_loss, 'Validation': avg_val_loss},
                              num_epoch + 1)

        tb_writer.flush()

        # track the best performance, and save the model's state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = 'fsrcnn_{}_{}'.format(timestamp, num_epoch)
            torch.save(model.state_dict(), model_path)

        num_epoch += 1


def evaluate(upscale_factor=4):
    model = FSRCNN(d=56, s=12, m=4, upscale_factor=upscale_factor)
    model.load_state_dict(torch.load('fsrcnn_20240121_231840_3'))

    root_dir = os.path.join(os.path.expanduser('~'), 'IPU', 'gen_ai', 'fsrcnn', 'datasets', 'div2k')
    test_set = DIV2K(root_dir=root_dir, upscale_factor=upscale_factor, train_mode='val', in_channels=3)

    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            collate_fn=test_set.collate_fn
    )

    for i, (test_image, _) in enumerate(test_loader):
        hr_output = model(test_image)

        torchvision.utils.save_image(hr_output, fp='out.png')

        if i == 1:
            break


if __name__ == '__main__':
    # train()

    evaluate()
