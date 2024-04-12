import argparse
import os
import pickle

import torch
import museval
import numpy as np
from functools import partial
from test import evaluate
from model.waveunet import Waveunet
from data.musdb import get_musdb_folds
from data.dataset import SeparationDataset
from data.utils import crop_targets

def main(args):
    # 加载模型
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)
    
    if args.cuda:
        model = model.cuda()

    print('load model ' + str(args.model_path))
    state = torch.load(args.model_path)

    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        # 如果保存的模型没有使用'state_dict'键,则直接加载整个状态字典
        model.load_state_dict(state['model_state_dict'], strict=False)

    # 加载测试数据集
    musdb = get_musdb_folds(args.dataset_dir)
    crop_func = partial(crop_targets, shapes=model.shapes)
    test_data = SeparationDataset(musdb, "test", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)

    # 评估模型
    print('开始评估...')
    test_metrics = evaluate(args, musdb["test"], model, args.instruments)

    # 保存评估结果
    with open(os.path.join(args.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)

    # 计算平均指标
    avg_SDRs = {inst : np.mean([np.nanmean(song[inst]["SDR"]) for song in test_metrics]) for inst in args.instruments}
    avg_SIRs = {inst : np.mean([np.nanmean(song[inst]["SIR"]) for song in test_metrics]) for inst in args.instruments}
    avg_ISRs = {inst : np.mean([np.nanmean(song[inst]["ISR"]) for song in test_metrics]) for inst in args.instruments}
    avg_SARs = {inst : np.mean([np.nanmean(song[inst]["SAR"]) for song in test_metrics]) for inst in args.instruments}

    print("avg SDR:")
    for inst in args.instruments:
        print(f"{inst}: {avg_SDRs[inst]:.3f}")

    print("avg SIR:")
    for inst in args.instruments:
        print(f"{inst}: {avg_SIRs[inst]:.3f}")
    
    print("avg ISR:")
    for inst in args.instruments:
        print(f"{inst}: {avg_ISRs[inst]:.3f}")

    print("avg SAR:")
    for inst in args.instruments:
        print(f"{inst}: {avg_SARs[inst]:.3f}")

    overall_SDR = np.mean([v for v in avg_SDRs.values()])
    print(f"\n总体平均 SDR: {overall_SDR:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
    parser.add_argument('--cuda', action='store_true', help='使用CUDA (default: False)')
    parser.add_argument('--model_path', type=str, default='checkpoints/waveunet/model.pth', help='预训练模型路径')                        
    parser.add_argument('--dataset_dir', type=str, default="/path/to/musdb18", help='MUSDB18根目录')
    parser.add_argument('--hdf_dir', type=str, default="hdf", help='MUSDB18 hdf目录')
    parser.add_argument('--output_dir', type=str, default='results', help='导出评估结果的目录')
    parser.add_argument('--sr', type=int, default=44100, help="采样率")
    parser.add_argument('--channels', type=int, default=2, help="输入音频通道数")
    parser.add_argument('--output_size', type=float, default=2.0, help="输出音频长度(秒)")
    parser.add_argument('--features', type=int, default=32, help='每一层的特征通道数')
    parser.add_argument('--levels', type=int, default=6, help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1, help="每个block的卷积层数") 
    parser.add_argument('--strides', type=int, default=4, help="Waveunet中的跨度")
    parser.add_argument('--kernel_size', type=int, default=5, help="卷积核大小")
    parser.add_argument('--conv_type', type=str, default="gn", help="卷积类型 (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed", help="重采样策略: fixed基于sinc的低通滤波 或 learned卷积层: fixed/learned")
    parser.add_argument('--separate', type=int, default=1, help="为每个音源训练单独的模型 (1) 还是只训练一个 (0)")
    parser.add_argument('--feature_growth', type=str, default="double", help="每层特征数量的增长方式: 加 (add) 初始特征数量或乘以2 (double)")

    args = parser.parse_args()

    main(args)