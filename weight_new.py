import argparse
import torch
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='输入的权重.pt文件')
    parser.add_argument('--ToFP16', action='store_true', help='convert model from FP32 to FP16')
    opt = parser.parse_args()

    # Load pytorch model
    # torch.load读取的pytorch模型.pt文件是一个字典，包含五个key：(我们只需要best_fitness和model)
    # epoch——当前模型对应epoch数；
    # best_fitness——模型性能加权指标，是一个值；
    # training_results——每个epoch的mAP、loss等信息；
    # model——保存的模型
    # optimizer——优化信息，一般占.pt文件一半大小
    model = torch.load(opt.weights, map_location=torch.device('cpu'))['model']

    # 读取模型网络
    net = model['model']

    # 把模型从单精度FP32转为半精度FP16
    if opt.ToFP16:
        net.half()

    # 只保留有用信息
    # 预训练权重模型默认epoch=-1，不保存training_results，不保存optimizer，相当于只保存了模型和权重
    ckpt = {'epoch': -1,
            'best_fitness': model['best_fitness'],
            'training_results': None,
            'model': net,
            'optimizer': None}

    # 保存模型
    torch.save(ckpt, 'test-slim.pt')


