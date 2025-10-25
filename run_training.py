import logging
from custom_logger import LOGGER_CONFIG
import argparse
from utils import get_argparge_type
from training import training_step
from config import (EPOCHS,
                    BATCH_SIZE,
                    LR_GEN,
                    LR_DISC,
                    CHECKPOINT_FOLDER,
                    CYCLE,
                    IDEN,
                    SAVE_AFTER)

logging.config.dictConfig(LOGGER_CONFIG)
save_logger = logging.getLogger("save_logger")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training cycle gan")
    parser.add_argument("-bs", "--batch_size", type=int, help="The custom batch size.", default=BATCH_SIZE)
    parser.add_argument("-ep", "--epochs", type=int, help="The number of total epochs.", default=EPOCHS)
    parser.add_argument("-pa", "--pathA", type=str, help="The domain that we wants to translate, for example, if we want to translate from horses to zebra, then this is the horse.", default="data/imgA")
    parser.add_argument("-pb", "--pathB", type=str, help="The domain that we wants to translate, for example, if we want to translate from horses to zebra, then this is the zebra.", default="data/imgB")
    parser.add_argument("-uta", "--use_transformA", type=bool, help="use transform for domain image A, if we want to transform the horses to zebras, then it is the transformation of the horses.", default=True)
    parser.add_argument("-utb", "--use_transformB", type=bool, help="use transform for domain image A, if we want to transform the horses to zebras, then it is the transformation of the zebras.", default=True)
    parser.add_argument("-lrg", "--lr_gen", type=int, help="learning rate of the generators", default=LR_GEN)
    parser.add_argument("-lrd", "--lr_disc", type=int, help="learning rate of the discriminator", default=LR_DISC)
    parser.add_argument("-cy", "--cycle_weight", type=float, help="The weight of the cycle loss in the total generator loss", default=CYCLE)
    parser.add_argument("-iden", "--identity", type=float, help="The weight of the identity loss in the total generator loss", default=IDEN)
    parser.add_argument("-sa", "--save_after", type=float, help="Save models after number of save_after epochs", default=SAVE_AFTER)


    # Because notebook and normal script have different methods for this.
    config_args = get_argparge_type(parser)
    # Get the args
    batch_size = config_args.batch_size
    epochs = config_args.epochs
    realA_path = config_args.pathA
    realB_path = config_args.pathB
    use_transformA = config_args.use_transformA
    use_transformB = config_args.use_transformB
    lr_gen = config_args.lr_gen
    lr_disc = config_args.lr_disc
    cycle = config_args.cycle_weight
    iden = config_args.identity
    save_after = config_args.save_after

    training_step(
        batch_size=batch_size,
        epochs=epochs,
        realA_path=realA_path,
        realB_path=realB_path,
        use_transformA=use_transformA,
        use_transformB=use_transformB,
        save_path=CHECKPOINT_FOLDER,
        lr_gen=lr_gen,
        lr_disc=lr_disc,
        cycle=cycle,
        iden=iden,
        save_after=save_after)


