import torch
from gen import Generator
from discriminator import Discriminator


from dataset import get_dataloader
from loss import total_gen_loss, total_disc_loss
from utils import select_img, create_folder, save_model
from utils import get_tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging.config
from custom_logger import LOGGER_CONFIG

tqdm = get_tqdm()
def training_step(batch_size, epochs, realA_path, realB_path, use_transformA, use_transformB, save_path, lr_gen, lr_disc, cycle, iden, save_after):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the models
    logging.config.dictConfig(LOGGER_CONFIG)
    save_logger = logging.getLogger("save_logger")
    log_dir = f"runs/cycle_gan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    # The models
    genA = Generator(3).to(device)
    genB = Generator(3).to(device)
    discA = Discriminator(3, 64).to(device)
    discB = Discriminator(3, 64).to(device)

    # Get the train_loader
    train_loader = get_dataloader(realA_path, realB_path, batch_size, use_transformA=use_transformA,
                                  use_transformB=use_transformB)

    # Get the optimizers
    optimA_gen = torch.optim.Adam(genA.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    optimB_gen = torch.optim.Adam(genB.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    optimA_disc = torch.optim.Adam(discA.parameters(), lr=lr_disc, betas=(0.5, 0.999))
    optimB_disc = torch.optim.Adam(discB.parameters(), lr=lr_disc, betas=(0.5, 0.999))
    # Training test_discA
    pool_imgs = []
    current_step = 0
    for epoch in tqdm(range(epochs), desc="Epoch training", position=0, colour="green"):
        epoch_loss_disc = 0
        epoch_loss_gen = 0

        # for batch_idx, (imA, imB) in enumerate(train_loader):
        for batch_idx, (imA, imB) in enumerate(tqdm(train_loader, desc="Batch training", leave=True, position=1)):
            imA = imA.to(device)
            imB = imB.to(device)
            current_step += imB.shape[0]

            # ======================
            #  Train Discriminators
            # ======================
            optimA_disc.zero_grad()
            optimB_disc.zero_grad()

            # Generate fake images
            fakeB_x = genA(imA)  # genA is the G in img_a -> G -> faked_b-> F -> faked_a, genB is the F
            fakeA_x = genB(imB)

            # Use buffer for discriminator training
            fakeA_x_D, fakeB_x_D = select_img(pool_imgs, fakeA_x, fakeB_x, max_len=50)

            # Compute discriminator loss with detached fakes total_disc_loss(Dg, Df, imgA, imgB, fakeA, fakeB)
            # Use automatic mixed precision (AMP):
            # runs certain operations in bfloat16 for speed/memory efficiency,
            # while keeping others in float32 to avoid numerical issues.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                disc_loss_val = total_disc_loss(
                    discA, discB, imA, imB,
                    fakeA_x_D.detach(), fakeB_x_D.detach()
                )

            epoch_loss_disc += disc_loss_val.item()
            disc_loss_val.backward()
            optimA_disc.step()
            optimB_disc.step()

            # =================
            #  Train Generators
            # =================
            optimA_gen.zero_grad()
            optimB_gen.zero_grad()

            # Generator loss (reuse fake images) total_gen_loss(G, F, Dg, Df, imgA, imgB, lambda_cyc=10.0, lambda_idt=0.5)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                gen_loss_val = total_gen_loss(genA, genB, discA, discB, imA, imB, lambda_cyc=cycle, lambda_idt=iden)
            epoch_loss_gen += gen_loss_val.item()
            gen_loss_val.backward()
            optimA_gen.step()
            optimB_gen.step()

            # =================
            #  Logging & Vis
            # =================
            writer.add_scalars("Loss/train_batch", {
                "discriminator": disc_loss_val.item(),
                "generator": gen_loss_val.item(),
            }, current_step)
            writer.add_image("Images/fake_zebra/", fakeA_x.squeeze(0), current_step)
            writer.add_image("Images/fake_horse/", fakeB_x.squeeze(0), current_step)
            if batch_idx % 50 == 0:
                tqdm.write(
                    f"Epoch={epoch}/{epochs}, "
                    f"batch={(batch_idx + 1)}/{len(train_loader)} - "
                    f"Disc loss={disc_loss_val.item():.4f}, Gen loss={gen_loss_val.item():.4f}"
                )

            # Create the folder in the current working directory for saving the
            save_model_root = create_folder(save_path)
            break

            # if batch_idx % 50 == 0:
            #     with torch.no_grad():
            #         visualize_imgs(imA.squeeze(0), imB.squeeze(0), genA=fakeA_x.squeeze(0), genB=fakeB_x.squeeze(0), resize=None, figsize=(15,20))

        tqdm.write(
            f"Epoch={epoch}, "
            f"epoch_loss_disc={epoch_loss_disc / len(train_loader):.4f} - "
            f"epoch_loss_gen={epoch_loss_gen / len(train_loader):.4f}"
        )
        writer.add_scalars("Loss/train_epoch", {
            "discriminator": disc_loss_val.item(),
            "generator": gen_loss_val.item(),
        }, epoch)
        # Don't based on loss, only save after each 5 epochs
        if epoch % save_after == 0:
            save_model(genA, save_model_root, "genA", epoch=epoch, optimizer=optimA_gen,
                                       loss=gen_loss_val.item(), log=save_logger)
            save_model(genB, save_model_root, "genB", epoch=epoch, optimizer=optimA_gen,
                                       loss=gen_loss_val.item(), log=save_logger)
            save_model(discA, save_model_root, "discA", epoch=epoch, optimizer=optimA_gen,
                                        loss=disc_loss_val.item(), log=save_logger)
            save_model(discB, save_model_root, "discB", epoch=epoch, optimizer=optimA_gen,
                                        loss=disc_loss_val.item(), log=save_logger)
    writer.close()