import os
import argparse
from dataset import *
from model_dino_5 import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
from multiprocessing import freeze_support
# Add this line to guard multiprocessing code
if __name__ == '__main__':
    freeze_support()  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument
    parser.add_argument('--model_dir', default='./tmp/model15_3.pth', type=str, help='where to save models' )
    parser.add_argument('--dino_path', type=str, default="./tmp/dino_deitsmall8_pretrain.pth")

    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hid_dim', default=384, type=int, help='hidden dimension size')
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.00000005, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=10000000, type=int, help='Number of steps for the learning rate decay.')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
    parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')

    opt = parser.parse_args()
    resolution = (256, 256)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train_set = CLEVR('train')
    train_set = PARTNET('train',resolution)

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim,dino_path=  opt.dino_path).to(device)
    # model.load_state_dict(torch.load('./tmp/model10.pth')['model_state_dict'])

    criterion = nn.MSELoss()

    params = [{'params': model.parameters()}]

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers)

    optimizer = optim.Adam(params, lr=opt.learning_rate)

    start = time.time()
    i = 0
    for epoch in range(opt.num_epochs):
        model.train()

        total_loss = 0

        for sample in tqdm(train_dataloader):
            i += 1

            if i < opt.warmup_steps:
                learning_rate = opt.learning_rate * (i / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (
                i / opt.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            image = sample['image'].to(device)
            # change to have dino OG
            recon_combined, recons, masks, slots,x_dino_OG = model(image)
            loss = criterion(recon_combined, x_dino_OG)
            total_loss += loss.item()

            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(train_dataloader)

        print ("Epoch: {}, Loss: {}, Time: {}, LR: {}".format(epoch, total_loss,
            datetime.timedelta(seconds=time.time() - start),learning_rate ))
        

        if not epoch % 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                }, opt.model_dir)
