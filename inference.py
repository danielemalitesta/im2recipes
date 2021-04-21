import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from custom_dataset import *
import numpy as np
import sys
from trijoint import im2recipe
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.manual_seed(opts.seed)

np.random.seed(opts.seed)

if not (torch.cuda.device_count()):
    device = torch.device(*('cpu', 0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda', 0))


def main():
    if not os.path.exists(opts.features_path):
        os.makedirs(opts.features_path)

    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device)

    print("=> loading checkpoint '{}'".format(opts.model_path))
    if device.type == 'cpu':
        checkpoint = torch.load(opts.model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(opts.model_path, encoding='latin1')
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # switch to evaluate mode
    model.eval()

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data = CustomDataset(root_dir=opts.img_path,
                         transform=transforms.Compose([
                             transforms.Resize(256),  # rescale the image keeping the original aspect ratio
                             transforms.CenterCrop(224),  # we get only the center of that rescaled
                             transforms.ToTensor(),
                             normalize,
                         ]))
    print('Test loader prepared.')

    print('Starting inference...\n')
    for i, d in enumerate(data):
        im, file = d
        output = np.squeeze(torch.nn.Sequential(list(model.children())[0])(im[None, ...].to(device)).data.cpu().numpy())
        np.save(opts.features_path + os.path.splitext(file)[0] + '.npy', output)

        if (i + 1) % 100 == 0:
            sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
            sys.stdout.flush()


if __name__ == '__main__':
    main()
