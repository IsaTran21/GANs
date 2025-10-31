import torch
from generator import Generator
from discriminator import Discriminator
from loss import critic_loss_gp, gen_loss
from utils import visualize_imgs_test, get_alpha, get_tqdm, create_folder
tqdm = get_tqdm()
from torch.utils.tensorboard import SummaryWriter
import inspect # For fusing kernels
from datetime import datetime
from dataset import Dataset_transformed
from utils import save_model, pickup_training
import logging.config
from custom_logger import LOGGER_CONFIG
import os
def train_step(batch_sizes, epoch_dict, lr_gen, lr_disc, im_path, max_size_i, crop_size, save_path, save_after, pickup, pickup_epoch, gen_path, disc_path):
    """
    :param batch_sizes: a dict of batch size for each resolution, example: {4: 128, 8: 128, 16: 64, 32: 16, 64: 16}
    :param epoch_list: the list of epoch for training at each resolution. e.g., [8, 8, 6, 4, 4, 4, 4, 4]
    :param epochs: number of epochs being trained
    :param im_path: the path of the training dataset
    idx: the current index to use for the IDX_LIST_TOTAL, e.g., if we want to train the highest resolution at 64
    then we use idx=4 (= max_size_i - 2). We just this to slice the IDX_LIST_TOTAL list.
    :param max_size_i: the power of 2, such as: 2, 3, 4, 5, 6, 7, 8, 9, 10.
    In this project, only use max_size_i = 6 (resolution=64)
    :param crop_size: the crop size in the transformation of the images in the training set
    :param pickup: None or a specific resolution, such as 4, 8, 16, 32,...256
    :we don't return anything, the model will be saved to disk after each save_period, the default is 1
    which means that we will save after each epoch the generator and discriminator (although in this wgan loss, we
    usually use the "critic", but in this project, we will use them interchangeably.
    """
    # batch_sizes = {4: 128, 8: 128, 16: 64, 32: 16, 64: 16}
    IDX_LIST_TOTAL = [0, 1, 2, 3, 4, 5, 6, 7]
    idx = max_size_i - 1
    IDX_LIST = IDX_LIST_TOTAL[:idx]

    BATCH_ALL_KEY = sorted(list(batch_sizes.keys()))[:idx] #First, we get the keys as a list
    # Just to avoid dictionary which is not in order in Python :>
    # Then, get its value in order
    BATCH_ALL = [batch_sizes[key] for key in BATCH_ALL_KEY]
    ds_test = Dataset_transformed(im_path, batch_sizes, max_size_i, crop_size)
    TRAIN_DL_ALL = {i: ds_test.all_data[i] for i in BATCH_ALL_KEY}
    # lambda_val = 10
    gen, disc, device = None, None, None
    if pickup is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the generator and discriminator
        gen = Generator(512).to(device)
        disc = Discriminator().to(device)
    if pickup is not None:
        resolution_dict = {2**(IDX_LIST_TOTAL[i]+2):i for i in IDX_LIST_TOTAL}

        assert gen_path is not None and disc_path is not None, "If you want to pick up training, must provide the saved .pth models for the generator and discriminator"
        assert pickup in sorted(list(resolution_dict.keys())), "Pass valid resolution, e.g. 4, 8, 16,...256"
        gen, disc, device = pickup_training(gen_path, disc_path)
        current_resolution = resolution_dict.get(pickup)
        # If we stop training at the end of resolution say, 32, then, current_resolution=4
        # pickup now mean: load the pretrained model at resolution 32, and then started to train
        # the 64 resolution onwards to the target resolution at idx
        tqdm.write(f"Start training at {pickup}")
        IDX_LIST = IDX_LIST_TOTAL[current_resolution:idx]

        # After pickup training, we just continue on using it, no need to pickup for the next epoch
        pickup = None



    # The optimization for the generator and discriminator
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda"
    gen_optim = torch.optim.Adam(gen.parameters(), betas=(0.0,0.999), lr=lr_gen, fused=use_fused)
    disc_optim = torch.optim.Adam(disc.parameters(), betas=(0.0,0.999), lr=lr_disc, fused=use_fused)

    RESOLUTIONS = {0: 4, 1: 8, 2: 16, 3: 32, 4: 64, 5:128, 6: 256, 7:512}
    EPOCH_DICT = epoch_dict#[8, 8, 6, 4, 4, 4, 4, 4]

    gen.train()
    disc.train()
    log_dir = f"runs/progran_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(save_path, log_dir))
    current_step = 0

    # Create the models
    logging.config.dictConfig(LOGGER_CONFIG)
    save_logger = logging.getLogger("save_logger")
    save_model_root = create_folder(os.path.join(save_path, "checkpoint"))

    for i in tqdm(IDX_LIST, desc="At resolution training", position=0, colour="green"):  # Train at each resolution
        # i = IDX #
        TOTAL_EPOCH = EPOCH_DICT[RESOLUTIONS[i]]
        TRAIN_DL = TRAIN_DL_ALL[RESOLUTIONS[i]]
        TOTAL_BATCH = len(TRAIN_DL)
        CURRENT_BATCH_SIZE = BATCH_ALL[i]
        LEFT_EPOCHS = range(pickup_epoch, TOTAL_EPOCH)
        pickup_epoch = 0 # start from 0 for the next epoch
        for epoch in tqdm(LEFT_EPOCHS, desc="Epoch training", position=1, colour="blue", leave=False):
            gen_loss_val_total = 0
            disc_loss_val_total = 0

            for batch_val, x_true in enumerate(tqdm(TRAIN_DL, desc="Batch training", leave=False, position=2, colour="yellow")):
                current_step += 1

                x_true = x_true.to(device)
                alpha = get_alpha(epoch_val=epoch, batch_val=batch_val, total_epoch=TOTAL_EPOCH, total_batch=TOTAL_BATCH)

                noise = torch.randn((CURRENT_BATCH_SIZE, 512,)).to(device)
                fake = gen(noise, idx=i, alpha=alpha).to(device)
                # print(f'Fake shape: {fake.shape}')

                # Train the discriminator
                disc_optim.zero_grad()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss_disc_val = critic_loss_gp(x_fake=fake.detach(), x_true=x_true, critic=disc, idx=i, alpha=alpha)
                disc_loss_val_total += loss_disc_val.item()
                loss_disc_val.backward()  # create gradient
                disc_optim.step()  # Update gradients

                # Train the generator
                gen_optim.zero_grad()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    gen_loss_val = gen_loss(x_fake=fake, critic=disc, idx=i, alpha=alpha)
                gen_loss_val_total += gen_loss_val.item()
                gen_loss_val.backward()  # Calculate the gradient
                gen_optim.step()  # Update the gradient
                writer.add_scalars("Loss/train_batch", {
                    "discriminator": loss_disc_val.item(),
                    "generator": gen_loss_val.item(),
                }, current_step)

                # Visualization
                if batch_val % 500 == 0 or (batch_val + 1) % len(TRAIN_DL) == 0:

                    gen.eval()
                    with torch.no_grad():
                        fake_img = gen(noise, idx=i, alpha=alpha)
                        writer.add_image("Images/fake_image/", fake_img[1].detach(), current_step)
                        writer.add_image("Images/real_image/", x_true[1], current_step)
                    gen.train()

                if (batch_val == (len(TRAIN_DL)-1) and epoch % 20==0) or (batch_val == (len(TRAIN_DL)-1) and (epoch + 1) % TOTAL_EPOCH == 0):
                    gen.eval()
                    tqdm.write(
                        f'Resolution = {RESOLUTIONS[i]}, Epoch = {epoch + 1}/{TOTAL_EPOCH}, batch = {batch_val}/{len(TRAIN_DL)} - at resolution: {RESOLUTIONS[i]}, disc loss = {loss_disc_val.item():.4f}, gen loss = {gen_loss_val.item():.4f}')
                    with torch.no_grad():
                        fake_img = gen(noise, idx=i, alpha=alpha)
                        visualize_imgs_test(fake_img[1].detach(), x_true[1])

                    gen.train()

            if epoch % save_after == 0 or (epoch+1) == TOTAL_EPOCH:
                # Save each save_after epoch or the end of the epoch
                save_model(gen, save_model_root, f"generator_resolution_{RESOLUTIONS[i]}", epoch=epoch, optimizer=gen_optim,
                           loss=gen_loss_val_total / len(TRAIN_DL) , log=save_logger)
                save_model(disc, save_model_root, f"discriminator_resolution_{RESOLUTIONS[i]}", epoch=epoch, optimizer=disc_optim,
                           loss=disc_loss_val_total / len(TRAIN_DL), log=save_logger)


    writer.close()

