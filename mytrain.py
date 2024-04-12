import argparse
import os
import time
from functools import partial

import torch
import pickle
import numpy as np
import logging
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

import model.utils as model_utils
import utils
from data.dataset import SeparationDataset
from data.musdb import get_musdb_folds
from data.utils import crop_targets, random_amplify
from test import evaluate, validate, validate_instruments
from model.waveunet import Waveunet

import librosa
import librosa.display
import matplotlib.pyplot as plt
# import h5py

def visualize_audio(audio, sr, filename):
    plt.figure(figsize=(12, 8))
    
    # 绘制波形图
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
    # 绘制频谱图
    plt.subplot(2, 1, 2)
    X = librosa.stft(audio)
    Xdb = librosa.amplitude_to_db(abs(X))

    # print("Xdb shape:", Xdb.shape)
    Xdb = Xdb.squeeze()
    # print("Xdb shape:", Xdb.shape)
    # print("Xdb type:", type(Xdb))

    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    # plt.show()
    
    plt.tight_layout()

    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    plt.savefig(filename)
    plt.close()

def visualize_dataset_distribution(train_data, filename):
    instrument_counts = {inst: 0 for inst in args.instruments}
    audio_lengths = []
    
    for _, targets in tqdm(train_data, desc="Processing audio"):
        # 检查targets字典是否表示一个无效项
        # if targets.get("invalid", False):
        #     continue  # 跳过这次循环
        # print(targets.shape)
        # print(type(targets))
        for inst in args.instruments:
            if inst in targets:
                instrument_counts[inst] += 1     
        audio_lengths.append(targets[args.instruments[0]].shape[1])
    
    print("finish")
    plt.figure(figsize=(12, 8))
    
    # 绘制乐器数量分布柱状图
    plt.subplot(2, 1, 1)
    plt.bar(args.instruments, instrument_counts.values())
    plt.title("Instrument Distribution")
    plt.xlabel("Instrument")
    plt.ylabel("Count")
    
    # 绘制音频长度分布直方图
    plt.subplot(2, 1, 2)
    plt.hist(audio_lengths, bins=20)
    plt.title("Audio Length Distribution")
    plt.xlabel("Length (samples)")
    plt.ylabel("Count")
    
    plt.tight_layout()

    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    plt.savefig(filename)
    plt.close()

def main(args):
    #torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5
    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    writer = SummaryWriter(args.log_dir)

    ### DATASET
    musdb = get_musdb_folds(args.dataset_dir)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop_targets, shapes=model.shapes)
    # Data augmentation function for training
    augment_func = partial(random_amplify, shapes=model.shapes, min=0.7, max=1.0)
    train_data = SeparationDataset(musdb, "train", args.instruments, args.sr, args.channels, model.shapes, True, args.hdf_dir, audio_transform=augment_func)
    val_data = SeparationDataset(musdb, "val", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)
    test_data = SeparationDataset(musdb, "test", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)

    # 在数据加载或预处理阶段调用可视化函数
    # for i, (audio, _) in enumerate(train_data):
    #     if i >= 10:  # 可视化前10个样本
    #         break
    #     # print("Original audio shape:", audio.shape)
    #     audio_mono = librosa.to_mono(audio)  # 混合通道
    #     # print("Mono audio shape:", audio_mono.shape)
    
    #     visualize_audio(audio_mono, args.sr, f"visualizations/train/audio_{i}.png")
    
    # visualize_dataset_distribution(train_data, "visualizations/train/dataset_distribution.png")    

    # 在数据加载或预处理阶段调用可视化函数
    # for i, (audio, _) in enumerate(val_data):
    #     if i >= 10:  # 可视化前10个样本
    #         break
    #     audio_mono = librosa.to_mono(audio)
    #     visualize_audio(audio_mono, args.sr, f"visualizations/val/audio_{i}.png")
    # # visualize_dataset_distribution(val_data, "visualizations/val/dataset_distribution.png")

    # # 在数据加载或预处理阶段调用可视化函数
    # for i, (audio, _) in enumerate(test_data):
    #     if i >= 10:  # 可视化前10个样本
    #         break
    #     audio_mono = librosa.to_mono(audio)
    #     visualize_audio(audio_mono, args.sr, f"visualizations/test/audio_{i}.png")        
    # visualize_dataset_distribution(test_data, "visualizations/test/dataset_distribution.png")

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)

    ##### TRAINING ####

    # Set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)

    # 在训练开始前定义最优和最新的checkpoint路径
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_model.pth")

    # 在训练开始前可视化模型结构
    # dummy_input = torch.randn(1, args.channels, model.shapes['input_frames'])
    # if args.cuda:
    #     dummy_input = dummy_input.cuda()
    # writer.add_graph(model, dummy_input)

    print('TRAINING START')
    while state["worse_epochs"] < args.patience:
        print("Training one epoch from iteration " + str(state["step"]))
        avg_time = 0.
        model.train()
        train_loss = 0
        train_loss_dict = {inst: 0.0 for inst in args.instruments}

        with tqdm(total=len(train_data) // args.batch_size) as pbar:
            np.random.seed()
            for example_num, (x, targets) in enumerate(dataloader):
                if args.cuda:
                    x = x.cuda()
                    for k in list(targets.keys()):
                        targets[k] = targets[k].cuda()

                t = time.time()

                # Set LR for this iteration
                utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                # writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                # Compute loss for each instrument/model
                optimizer.zero_grad()
                # outputs, avg_loss, avg_loss_dict = model_utils.compute_loss(model, x, targets, criterion, compute_grad=True)

                outputs, avg_loss = model_utils.compute_loss(model, x, targets, criterion, compute_grad=True)
                avg_loss_dict = model_utils.compute_avg_loss_dict(model, outputs, targets, criterion)

                optimizer.step()

                state["step"] += 1

                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                # writer.add_scalar("train_loss", avg_loss, state["step"])
                # 累积训练loss
                train_loss += avg_loss
                for inst in args.instruments:
                    train_loss_dict[inst] += avg_loss_dict[inst]

                if example_num % args.example_freq == 0:
                    input_centre = torch.mean(x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]], 0) # Stereo not supported for logs yet
                    writer.add_audio("input", input_centre, state["step"], sample_rate=args.sr)

                    for inst in outputs.keys():
                        writer.add_audio(inst + "_pred", torch.mean(outputs[inst][0], 0), state["step"], sample_rate=args.sr)
                        writer.add_audio(inst + "_target", torch.mean(targets[inst][0], 0), state["step"], sample_rate=args.sr)

                pbar.update(1)
        writer.add_scalar("lr", utils.get_lr(optimizer), state["epochs"])

        # 计算并记录平均训练loss
        avg_train_loss = train_loss / len(dataloader)
        writer.add_scalar("train_loss", avg_train_loss, state["epochs"])
        logger.info(f"Epoch {state['epochs']}: Train Loss = {avg_train_loss:.4f}")

        for inst in args.instruments:
            avg_train_loss_inst = train_loss_dict[inst] / len(dataloader)
            writer.add_scalar(f"train_loss/{inst}", avg_train_loss_inst, state["epochs"])
            logger.info(f"Epoch {state['epochs']}: Train Loss ({inst}) = {avg_train_loss_inst:.4f}")

        # VALIDATE
        val_loss = validate(args, model, criterion, val_data)
        print("VALIDATION FINISHED: LOSS: " + str(val_loss))
        writer.add_scalar("val_loss", val_loss, state["epochs"])
        logger.info(f"Epoch {state['epochs']}: Validation Loss = {val_loss:.4f}")

        val_loss_dict = validate_instruments(args, model, criterion, val_data)
        for inst in args.instruments:
            writer.add_scalar(f"val_loss/{inst}", val_loss_dict[inst], state["epochs"])
            logger.info(f"Epoch {state['epochs']}: Validation Loss ({inst}) = {val_loss_dict[inst]:.4f}")

        # EARLY STOPPING CHECK
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        if val_loss < state["best_loss"]:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = best_checkpoint_path
        
            # 保存最优的checkpoint
            print("Saving best model...")
            model_utils.save_model(model, optimizer, state, best_checkpoint_path)
            logger.info(f"Epoch {state['epochs']}: Saving best model with validation loss {val_loss:.4f}")
        else:
            state["worse_epochs"] += 1

        state["epochs"] += 1

        # CHECKPOINT
        print("Saving latest model...")
        model_utils.save_model(model, optimizer, state, latest_checkpoint_path)
        logger.info(f"Epoch {state['epochs']}: Saving latest model")

    #### TESTING ####
    # Test loss
    print("TESTING")
    logger.info("Starting testing...")

    # Load best model based on validation loss
    state = model_utils.load_model(model, None, state["best_checkpoint"], args.cuda)
    test_loss = validate(args, model, criterion, test_data)
    print("TEST FINISHED: LOSS: " + str(test_loss))
    logger.info(f"Test Loss: {test_loss:.4f}")
    writer.add_scalar("test_loss", test_loss, state["epochs"])

    test_loss_dict = validate_instruments(args, model, criterion, test_data)
    for inst in args.instruments:
        writer.add_scalar(f"test_loss/{inst}", test_loss_dict[inst], state["epochs"])

    # Mir_eval metrics
    test_metrics = evaluate(args, musdb["test"], model, args.instruments)

    # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(args.checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)

    # Write most important metrics into Tensorboard log
    # 计算平均指标
    avg_SDRs = {inst : np.mean([np.nanmean(song[inst]["SDR"]) for song in test_metrics]) for inst in args.instruments}
    avg_SIRs = {inst : np.mean([np.nanmean(song[inst]["SIR"]) for song in test_metrics]) for inst in args.instruments}
    avg_ISRs = {inst : np.mean([np.nanmean(song[inst]["ISR"]) for song in test_metrics]) for inst in args.instruments}
    avg_SARs = {inst : np.mean([np.nanmean(song[inst]["SAR"]) for song in test_metrics]) for inst in args.instruments}
    
    logger.info("Test Metrics:")
    for inst in args.instruments:
        writer.add_scalar("test_SDR_" + inst, avg_SDRs[inst], state["epochs"])
        writer.add_scalar("test_SIR_" + inst, avg_SIRs[inst], state["epochs"])
        writer.add_scalar("test_ISR_" + inst, avg_SDRs[inst], state["epochs"])
        writer.add_scalar("test_SAR_" + inst, avg_SIRs[inst], state["epochs"])

        logger.info(f"Instrument: {inst}")
        logger.info(f"  SDR: {avg_SDRs[inst]:.4f}")
        logger.info(f"  SIR: {avg_SIRs[inst]:.4f}")
        logger.info(f"  ISR: {avg_ISRs[inst]:.4f}")
        logger.info(f"  SAR: {avg_SARs[inst]:.4f}")
    overall_SDR = np.mean([v for v in avg_SDRs.values()])
    writer.add_scalar("test_SDR", overall_SDR, state["epochs"])
    print("SDR: " + str(overall_SDR))
    logger.info(f"Overall SDR: {overall_SDR:.4f}")

    writer.close()

if __name__ == '__main__':
    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--log_dir', type=str, default='logs/waveunet',
                        help='Folder to write logs into')
    parser.add_argument('--dataset_dir', type=str, default="/mnt/windaten/Datasets/MUSDB18HQ",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="hdf",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet',
                        help='Folder to write checkpoints into')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate in LR cycle (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate in LR cycle (default: 5e-5)')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of LR cycles per epoch')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=2,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--example_freq', type=int, default=200,
                        help="Write an audio summary into Tensorboard logs every X training iterations")
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    try:
        # 解析命令行参数
        args = parser.parse_args()

        # 在训练开始前配置日志记录器
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        # 创建日志文件的目录(如果不存在)
        log_dir = os.path.dirname(os.path.join(args.checkpoint_dir, "training.log"))
        os.makedirs(log_dir, exist_ok=True)

        # 创建一个文件处理器,将日志信息写入文件
        file_handler = logging.FileHandler(os.path.join(args.checkpoint_dir, "training.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        main(args)
    except Exception as e:
        # 捕获并记录任何异常
        logger.exception("An error occurred during training: %s", str(e))