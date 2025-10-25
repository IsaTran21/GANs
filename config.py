import yaml

with open('config.yaml', 'r') as f:
    data = yaml.full_load(f)
gen_block_small = data["GEN_BLOCK_SMALL"]
gen_block_large = data["GEN_BLOCK_LARGE"]
BATCH_SIZE = data["TRAINING_PARAMS"]["BATCH_SIZE"]
EPOCHS = data["TRAINING_PARAMS"]["EPOCHS"]
LR_GEN = data["TRAINING_PARAMS"]["LR_GEN"]
LR_DISC = data["TRAINING_PARAMS"]["LR_DISC"]
CHECKPOINT_FOLDER = data["TRAINING_PARAMS"]["CHECKPOINT_FOLDER"]
GAN = data["TRAINING_PARAMS"]["LAMBDA_VAL"]["GAN"]
CYCLE = data["TRAINING_PARAMS"]["LAMBDA_VAL"]["CYCLE"]
IDEN = data["TRAINING_PARAMS"]["LAMBDA_VAL"]["IDEN"]