import os
from omegaconf import OmegaConf
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from fusor_net import Fusor_Net

'''
train strategy：
step1：set id_weight = 5 in loss.py,running:
python fusor_trainer.py -n ffhq -e 3 

step2：set id_weight = 7 in loss.py,running:
python fusor_trainer.py -n celeba -e 6 -p './chkpt/ffhq/ffhq_last_2.ckpt'

step2：set id_weight = 9 in loss.py,running:
python fusor_trainer.py -n vgg -e 10 -p './chkpt/celeba/celeba_last_5.ckpt'

'''
def main(args):
    hp = OmegaConf.load(args.config)
    model = Fusor_Net(hp)
    save_path = os.path.join(hp.log.chkpt_dir, args.name)
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(

        save_top_k=3,
        mode='min',
        monitor='val_loss',
        dirpath=save_path,
        filename='{epoch:02d}-{val_loss:.2f}',
        verbose=1,
        period=1
    )

    trainer = Trainer(
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        default_root_dir=save_path,
        gpus=-1 if args.gpus is None else args.gpus,
        distributed_backend='ddp',
        num_sanity_val_steps=0,
        resume_from_checkpoint=args.checkpoint_path,
        gradient_clip_val=hp.model.grad_clip,
        fast_dev_run=args.fast_dev_run,
        val_check_interval=args.val_interval,
        progress_bar_refresh_rate=1,
        max_epochs=args.max_epoch,
        limit_train_batches=0.3,
        limit_val_batches=0.2,
    )

    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(save_path, args.name + "_last_" + str(args.max_epoch - 1) + ".ckpt"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/train.yaml', required=False,
                        help="path of configuration yaml file")
    parser.add_argument('-g', '--gpus', type=str, default=None,
                        help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="Name of the run.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint for resuming")
    parser.add_argument('-s', '--save_top_k', type=int, default=-1,
                        help="save top k checkpoints, default(-1): save all")
    parser.add_argument('-f', '--fast_dev_run', type=bool, default=False,
                        help="fast run for debugging purpose")
    parser.add_argument('--val_interval', type=float, default=0.10,
                        help="run val loop every * training epochs")
    parser.add_argument('--trainmode', type=int, default=0, required=False,
                        help="0 :original input,1:with masked input,2:joinly train two model")
    parser.add_argument('-e', '--max_epoch', type=int, required=True,
                        help="max epochs to run")

    args = parser.parse_args()

    main(args)
