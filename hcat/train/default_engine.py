from typing import Union

import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import torch.optim.swa_utils
from torch.utils.data import DataLoader, Dataset
from yacs.config import CfgNode

"""
TODO: 
    - call this script (not multithreaded) using QThreads 
    - no mutliprocessing...
    - maybe allow for custom scripts if necessary otherwise
"""


def _valid_targets():
    raise NotImplementedError("test")

Dataset = Union[Dataset, DataLoader]

torch.manual_seed(101196)
torch.set_float32_matmul_precision('high')


def train(base_model: nn.Module, cfg: CfgNode):
    pass
    # device = f'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # base_model = base_model.to(device)
    # base_model = torch.nn.parallel.DistributedDataParallel(base_model)
    #
    # if int(torch.__version__[0]) >= 2:
    #     print('Comiled with Inductor')
    #     model = torch.compile(base_model)
    # else:
    #     model = torch.jit.script(base_model)
    #
    # # augmentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(transform_from_cfg, cfg=cfg,
    # #                                                                           device=device)
    # augmentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(merged_transform_2D, device=device)
    #
    # # INIT DATA ----------------------------
    # _datasets = []
    # for path, N in zip(cfg.TRAIN.TRAIN_DATA_DIR, cfg.TRAIN.TRAIN_SAMPLE_PER_IMAGE):
    #     _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else 'cpu'
    #     _datasets.append(dataset(files=path,
    #                              transforms=augmentations,
    #                              sample_per_image=N,
    #                              device=device,
    #                              ).to(_device))
    #
    # merged_train = MultiDataset(*_datasets)
    #
    # train_sampler = torch.utils.data.distributed.DistributedSampler(merged_train)
    # dataloader = DataLoader(merged_train, num_workers=0, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
    #                         sampler=train_sampler, collate_fn=colate)
    #
    # # Validation Dataset
    # _datasets = []
    # for path, N in zip(cfg.TRAIN.VALIDATION_DATA_DIR, cfg.TRAIN.VALIDATION_SAMPLE_PER_IMAGE):
    #     _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else 'cpu'
    #     _datasets.append(dataset(files=path,
    #                              transforms=augmentations,
    #                              sample_per_image=N,
    #                              device=device,
    #                              ).to(_device))
    #
    # merged_validation = MultiDataset(*_datasets)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(merged_validation)
    # if _datasets or cfg.TRAIN.VALIDATION_BATCH_SIZE >= 1:
    #     valdiation_dataloader = DataLoader(merged_validation, num_workers=0, batch_size=cfg.TRAIN.VALIDATION_BATCH_SIZE,
    #                                        sampler=test_sampler,
    #                                        collate_fn=colate)
    #
    # else:  # we might not want to run validation...
    #     valdiation_dataloader = None
    #
    # # INIT FROM CONFIG ----------------------------
    # torch.backends.cudnn.benchmark = cfg.TRAIN.CUDNN_BENCHMARK
    # torch.autograd.profiler.profile = cfg.TRAIN.AUTOGRAD_PROFILE
    # torch.autograd.profiler.emit_nvtx(enabled=cfg.TRAIN.AUTOGRAD_EMIT_NVTX)
    # torch.autograd.set_detect_anomaly(cfg.TRAIN.AUTOGRAD_DETECT_ANOMALY)
    #
    # epochs = cfg.TRAIN.NUM_EPOCHS
    #
    # writer = SummaryWriter() if cfg.TRAIN.TENSORBOARD else None
    # if writer:
    #     print('SUMMARY WRITER LOG DIR: ', writer.get_logdir())
    #
    #
    # optimizer = _valid_optimizers[cfg.TRAIN.OPTIMIZER](model.parameters(),
    #                                                    lr=cfg.TRAIN.LEARNING_RATE,
    #                                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # scheduler = _valid_lr_schedulers[cfg.TRAIN.SCHEDULER](optimizer, T_0=cfg.TRAIN.SCHEDULER_T0)
    # scaler = GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    #
    # swa_model = torch.optim.swa_utils.AveragedModel(model)
    # swa_start = 100
    #
    # # TRAIN LOOP ----------------------------
    # epoch_range = trange(epochs, desc=f'Loss = {1.0000000}')
    # avg_epoch_loss, avg_val_loss = [], []
    #
    # def prepare_loss_fn():
    #     raise NotImplementedError()
    #
    # for e in epoch_range:
    #     _loss = []
    #
    #     if cfg.TRAIN.DISTRIBUTED:
    #         train_sampler.set_epoch(e)
    #
    #     for images, data_dict in dataloader:
    #         optimizer.zero_grad(set_to_none=True)
    #
    #         with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
    #             loss: Dict[str, Tensor] = model(images, data_dict)
    #             loss = prepare_loss_fn(loss) # not implemented
    #
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #
    #         if e > swa_start:
    #             swa_model.update_parameters(model)
    #
    #         _loss.append(loss.item())
    #
    #     avg_epoch_loss.append(mean(_loss))
    #     scheduler.step()
    #
    #     # # Validation Step
    #     if e % 10 == 0 and valdiation_dataloader:
    #         _loss = []
    #         for images, data_dict in valdiation_dataloader:
    #
    #             with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
    #                 with torch.no_grad():
    #                     out: Tensor = model(images)
    #                     loss = prepare_loss_fn(loss)  # not implemented
    #
    #             scaler.scale(loss)
    #             _loss.append(loss.item())
    #
    #         avg_val_loss.append(mean(_loss))
    #
    #
    #     # now we write the loss to tqdm progress bar
    #     epoch_range.desc = f'lr={scheduler.get_last_lr()[-1]:.3e}, Loss (train | val): ' + f'{avg_epoch_loss[-1]:.5f} | {avg_val_loss[-1]:.5f}'
    #
    #     # Save a state dict every so often
    #     state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    #     if e % cfg.TRAIN.SAVE_INTERVAL == 0:
    #         torch.save(state_dict, cfg.TRAIN.SAVE_PATH + f'/test_{e}.trch')
    #
    # state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    # constants = {'cfg': cfg,
    #              'model_state_dict': state_dict,
    #              'avg_epoch_loss': avg_epoch_loss,
    #              'avg_val_loss': avg_epoch_loss,
    #              }
    #
    # return constants
    #
    # # try:
    # #     torch.save(constants, f'{cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch')
    # # except:
    # #     print(f'Could not save at: {cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch'
    # #           f'Saving at {os.getcwd()}/{os.path.split(writer.log_dir)[-1]}.trch instead')
    # #
    # #     torch.save(constants, f'{os.getcwd()}/{cfg.TRAIN.TARGET}_{os.path.split(writer.log_dir)[-1]}.trch', )
