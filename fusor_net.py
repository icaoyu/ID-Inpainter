import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model.CBAM import CBAMResNet
from model.networks import Generator, AttributesEncoder,Discriminator
from model.loss import GANLoss, F_Loss
from dataloader.dataset import *

class Fusor_Net(pl.LightningModule):
    def __init__(self, hp):
        super(Fusor_Net, self).__init__()
        self.hp = hp

        self.G = Generator(hp.sampler.vector_size)
        self.E = AttributesEncoder()
        self.D = Discriminator(3)
        self.Z = CBAMResNet(50, feature_dim=hp.sampler.vector_size, mode='ir')  # resnet50
        self.Z.load_state_dict(torch.load(hp.sampler.chkpt_path,map_location='cpu'))
        self.Loss_GAN = GANLoss()
        self.Loss_E_G = F_Loss()


    def forward(self, target_img, source_img):
        z_id = self.Z(source_img)
        z_id = F.normalize(z_id)
        z_id = z_id.detach()
        feature_map = self.E(target_img)
        output = self.G(z_id, feature_map)
        output_z_id = self.Z(output)
        output_z_id = F.normalize(output_z_id)
        output_feature_map = self.E(output)
        return output, z_id, output_z_id, feature_map, output_feature_map


    def training_step(self, batch, batch_idx, optimizer_idx):
        target_img, source_img, same,mask = batch
        if optimizer_idx == 0:
            output, z_id, output_z_id, feature_map, output_feature_map = self(target_img, source_img)

            self.generated_img = output

            output_multi_scale_val = self.D(output)
            loss_GAN = self.Loss_GAN(output_multi_scale_val, True, for_discriminator=False)
            loss_E_G, loss_att, loss_id, loss_rec = self.Loss_E_G(target_img, output, feature_map, output_feature_map, z_id,output_z_id, same)

            loss_G = loss_E_G + loss_GAN

            self.logger.experiment.add_scalar("Loss G", loss_G.item(), self.global_step)
            self.logger.experiment.add_scalar("Attribute Loss", loss_att.item(), self.global_step)
            self.logger.experiment.add_scalar("ID Loss", loss_id.item(), self.global_step)
            self.logger.experiment.add_scalar("Reconstruction Loss", loss_rec.item(), self.global_step)
            self.logger.experiment.add_scalar("GAN Loss", loss_GAN.item(), self.global_step)

            return loss_G

        else:
            multi_scale_val = self.D(target_img)
            output_multi_scale_val = self.D(self.generated_img.detach())

            loss_D_fake = self.Loss_GAN(multi_scale_val, True)
            loss_D_real = self.Loss_GAN(output_multi_scale_val, False)

            loss_D = loss_D_fake + loss_D_real

            self.logger.experiment.add_scalar("Loss D", loss_D.item(), self.global_step)
            return loss_D

    def validation_step(self, batch, batch_idx):
        target_img, source_img, same,mask = batch
        output, z_id, output_z_id, feature_map, output_feature_map = self(target_img, source_img)

        self.generated_img = output

        output_multi_scale_val = self.D(output)
        loss_GAN = self.Loss_GAN(output_multi_scale_val, True, for_discriminator=False)
        loss_E_G, loss_att, loss_id, loss_rec = self.Loss_E_G(target_img, output, feature_map, output_feature_map,
                                                              z_id, output_z_id, same)
        loss_G = loss_E_G + loss_GAN
        self.log('val_loss', loss_G)
        return {"loss": loss_G, 'target': target_img[0].cpu(), 'source': source_img[0].cpu(),  "output": output[0].cpu(), }

#     def validation_end(self, outputs):
#         loss = torch.stack([x["loss"] for x in outputs]).mean()
#         validation_image = []
#         for x in outputs:
#             validation_image = validation_image + [x['target'], x['source'], x["output"]]
#         validation_image = torchvision.utils.make_grid(validation_image, nrow=3)
#
#         self.logger.experiment.add_scalar("Validation Loss", loss.item(), self.global_step)
#         self.logger.experiment.add_image("Validation Image", validation_image, self.global_step)
#         self.log('val_loss', loss)
# #         self.save_checkpoint("last.ckpt")
#
#         return {"loss": loss, "image": validation_image, }


    def configure_optimizers(self):
        lr_g = self.hp.model.learning_rate_E_G
        lr_d = self.hp.model.learning_rate_D
        b1 = self.hp.model.beta1
        b2 = self.hp.model.beta2

        opt_g = torch.optim.Adam(list(self.G.parameters()) + list(self.E.parameters()), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform_list = [
            transforms.Resize((112, 112)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(transform_list)

        dataset = Train_Dataset(self.hp.model.dataset_dir, transform=transform)
        return DataLoader(dataset, batch_size=self.hp.model.batch_size, num_workers=self.hp.model.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        transform_list = [
            transforms.Resize((112, 112)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(transform_list)
        dataset = Val_Dataset(self.hp.model.valset_dir, transform=transform)
        # return DataLoader(dataset, batch_size=1, shuffle=False)

        return DataLoader(dataset, batch_size=self.hp.model.batch_size, num_workers=self.hp.model.num_workers, shuffle=False, drop_last=True)

