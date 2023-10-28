import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, input_dim, nChannel, nConv):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, nChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(nChannel),
            nn.ReLU(),
            *[
                nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(nChannel),
                nn.ReLU()
            ] * (nConv - 1),
            nn.Conv2d(nChannel, nChannel, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def main(args):
    # Load Image
    im = cv2.imread(args.input)

    if args.use_binary:
        # Convert the image to grayscale
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Convert the grayscale image to binary
        _, im_binary = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)
        # Save binary image
        cv2.imwrite("binary.png", im_binary)
        # Add an extra dimension for color channels
        im_binary = np.expand_dims(im_binary, 0)
        data = torch.from_numpy(np.array([im_binary.astype('float32') / 255.]))
    else:
        # Load Image in color format
        data = torch.from_numpy(np.array([im.transpose(2, 0, 1).astype('float32') / 255.]))

    # Training
    model = Classifier(data.size(1), args.nChannel, args.nConv)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_hpy = torch.nn.L1Loss(size_average=True)
    loss_hpz = torch.nn.L1Loss(size_average=True)
    HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    label_colours = np.random.randint(255, size=(100, 3))

    for batch_idx in range(args.maxIter):
        optimizer.zero_grad()
        output = model(data)[0].permute(1, 2, 0).view(-1, args.nChannel)
        
        # continuity loss
        outputHP = output.view(im.shape[0], im.shape[1], args.nChannel)
        lhpy = loss_hpy(outputHP[1:, :, :] - outputHP[:-1, :, :], HPy_target)
        lhpz = loss_hpz(outputHP[:, 1:, :] - outputHP[:, :-1, :], HPz_target)

        _, target = torch.max(output, 1)
        im_target = target.cpu().numpy()
        nLabels = len(np.unique(im_target))
        
        if args.visualize:
            im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target]).reshape(im.shape).astype(np.uint8)
            cv2.imshow("output", im_target_rgb)
            cv2.waitKey(10)

        # loss
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)        
        loss.backward()
        optimizer.step()

        print(batch_idx, '/', args.maxIter, '| label num :', nLabels, '| loss :', loss.item())

        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")

            output = model( data )[ 0 ]
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
            _, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            cv2.imwrite( "output.png", im_target_rgb )
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
    parser.add_argument('--nChannel', default=100, type=int, help='number of channels')
    parser.add_argument('--use_binary', default=True, type=bool, help='whether to use binary')
    parser.add_argument('--maxIter', default=1000, type=int, help='number of maximum iterations')
    parser.add_argument('--minLabels', default=3, type=int, help='minimum number of labels')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nConv', default=3, type=int, help='number of convolutional layers')
    parser.add_argument('--visualize', default=1, type=int, help='visualization flag')
    parser.add_argument('--input', help='input image file name', required=True)
    parser.add_argument('--stepsize_sim', default=1, type=float, help='step size for similarity loss')
    parser.add_argument('--stepsize_con', default=1, type=float, help='step size for continuity loss')
    parser.add_argument('--stepsize_scr', default=0.5, type=float, help='step size for scribble loss')
    
    args = parser.parse_args()
    
    main(args)