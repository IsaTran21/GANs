from config import (
    BATCH_SIZES,
    LR_GEN,
    LR_DISC,
    CHECKPOINT_FOLDER,
    IM_PATH,
    MAX_SIZE_I,
    CROP_SIZE,
    SAVE_AFTER,
    EPOCH_DICT)
from utils import get_argparge_type
from training import train_step

import argparse, ast



def int_or_none(value):
    if value.lower() == "none":
        return None
    return int(value)
if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="Training ProGAN")
    # This is for the batch size at each level
    resolution_list = [4, 8, 16, 32, 64, 128, 256, 512]
    for re in resolution_list:
        parser.add_argument(f"-b{re}", f"--batch_{re}", type=int, help=f"batch size for resolution {re}", default=BATCH_SIZES[re])

    # For each epoch
    for re in resolution_list:
        parser.add_argument(f"-e{re}", f"--epoch_{re}", type=int, help=f"epoch size for resolution {re}", default=EPOCH_DICT[re])

    parser.add_argument("-bs", "--batch_sizes", type=ast.literal_eval, help="The custom batch sizes. "
                                                                "e.g. {4: 128, 8: 128, 16: 64, 32: 16, 64: 16, 128: 16, 256: 16, 512: 16}", default=BATCH_SIZES)
    # parser.add_argument("-epl", "--epoch_list", type=ast.literal_eval, help="The list, e.g. [8, 8, 6, 4, 4, 4, 4, 4], each elment corresponds to the number of epochs for training at that resolution"
    #                                                             "we can count starts to count it from 4, 8, 16, 32, 64, 128, 256, 512", default=EPOCH_LIST)
    parser.add_argument("-ip", "--im_path", type=str, help="the path contains the training dataset, example: data/train/, remember to at the last slash", default=IM_PATH)
    parser.add_argument("-msi", "--max_size_i", type=int, help="the i power to get the resolution that we want, e.g., if we want to "
                                                               "generate the resolution 64, then the max_size_i = 6, because 2**6=64", default=MAX_SIZE_I)
    parser.add_argument("-cs", "--crop_size", type=int_or_none, help="The image height and width kept after removing outer regions.", default=CROP_SIZE)
    parser.add_argument("-sp", "--save_path", type=str, help="the paths for saving the models after training each save_after epochs", default=CHECKPOINT_FOLDER)
    parser.add_argument("-sa", "--save_after", type=int, help="The number of epoch to save the models.", default=SAVE_AFTER)
    parser.add_argument("-lg", "--lr_gen", type=float, help="Learning rate for the generator", default=LR_GEN)
    parser.add_argument("-ld", "--lr_disc", type=float, help="Learning rate for the discriminator/critic", default=LR_DISC)



    config_args = get_argparge_type(parser)
    batch_sizes = config_args.batch_sizes
    # epoch_list = config_args.epoch_list
    im_path = config_args.im_path
    max_size_i = config_args.max_size_i
    crop_size = config_args.crop_size
    save_path = config_args.save_path
    save_after = config_args.save_after
    lr_gen = config_args.lr_gen
    lr_disc = config_args.lr_disc
    updated_batch_sizes = {i: getattr(config_args, f"batch_{i}") for i in resolution_list}
    updated_epoch_sizes = {i: getattr(config_args, f"epoch_{i}") for i in resolution_list}

    train_step(batch_sizes=updated_batch_sizes,
               epoch_dict=updated_epoch_sizes,
               lr_gen=lr_gen,
               lr_disc=lr_disc,
               im_path=im_path,
               max_size_i=max_size_i,
               crop_size=crop_size,
               save_path=save_path,
               save_after=save_after)
