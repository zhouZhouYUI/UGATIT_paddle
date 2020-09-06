from UGATIT import UGATIT
import argparse
from utils import *

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    # 代码模式  train/test
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    # U-GAT-IT 全版本 或 轻量级版本 light
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    # 数据集
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name')

    # 训练迭代次数
    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    # 数据获取批量大小
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    # 图像打印输出频率
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    # 1000
    # 模型存储 频率
    parser.add_argument('--save_freq', type=int, default=2000, help='The number of model save freq')
    # 2000
    # 衰减标识
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    # 学习率
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    # 权重衰减
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    # Gan的权重
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    # Cycle的权重
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    # Identity权重
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    # CAM的权重
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')

    # 每个层 通道数量
    # parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--ch', type=int, default=16, help='base channel number per layer')
    # 残差块的数量
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    # 判别器层的维度
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    # 图像的大小
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    # 图像的通道数
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    # 存储结果的路径
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    # 使用gpu 或 cpu进行计算
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    # 基准标识？
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    # 加载模型重新训练
    parser.add_argument('--resume', type=str2bool, default=False)

    # 保存所有训练记录
    parser.add_argument('--keep_result', type=str2bool, default=True)

    # 是否使用数据增强
    parser.add_argument('--data_improve', type=str2bool, default=True)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test', 'single'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    device = args.device
    place = fluid.CUDAPlace(0) if device != 'cpu' else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # open session
        gan = UGATIT(args)

        # build graph
        gan.build_model()

        if args.phase == 'train' :
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

if __name__ == '__main__':
    main()
