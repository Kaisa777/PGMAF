import os
import argparse
import torch
import torch.nn as nn
import model
import data_process
import train_process
import test_process
# import dev_process

from util.write_file import WriteFile
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from datetime import datetime
from model import ModelParam

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-devices', type=int, default=0, help='gpu')
    parser.add_argument('-run_type', type=int, default=1, help='1: train, 2: test')
    parser.add_argument('-save_model_path', type=str, default='checkpoint', help='save the good model.pth path')
    parser.add_argument('-add_note', type=str, default='', help='Additional instructions when saving files')
    parser.add_argument('-gpu_num', type=str, default='0', help='gpu index')
    parser.add_argument('-gpu0_bsz', type=int, default=0, help='the first GPU batch size')
    parser.add_argument('-epoch', type=int, default=10, help='train epoch num')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size number')
    parser.add_argument('-acc_grad', type=int, default=1, help='Number of steps to accumulate gradient on (divide the batch_size and accumulate)')
    parser.add_argument('-lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('-min_lr', type=float, default=1e-9, help='the minimum lr')
    parser.add_argument('-warmup_step_epoch', type=float, default=2, help='warmup learning step')
    parser.add_argument('-num_workers', type=int, default=0, help='loader dataset thread number')
    parser.add_argument('-l_dropout', type=float, default=0.1, help='classify linear dropout')
    parser.add_argument('-train_log_file_name', type=str, default='train_correct_log.txt', help='save some train log')
    parser.add_argument('-dis_ip', type=str, default='tcp://localhost:23456', help='init_process_group ip and port')
    parser.add_argument('-optim_b1', type=float, default=0.9, help='torch.optim.Adam betas_1')
    parser.add_argument('-optim_b2', type=float, default=0.999, help='torch.optim.Adam betas_2')
    parser.add_argument('-data_path_name', type=str, default='10-flod-1', help='train, dev and test data path name')
    parser.add_argument('-data_type', type=str, default='MVSA-single', help='Train data type: MVSA-single and MVSA-multiple and HFM')
    parser.add_argument('-word_length', type=int, default=200, help='the sentence\'s word length')
    parser.add_argument('-save_acc', type=float, default=-1, help='The default ACC threshold')
    parser.add_argument('-save_F1', type=float, default=-1, help='The default F1 threshold')
    parser.add_argument('-text_model', type=str, default='bert-base', help='language model')
    parser.add_argument('-pl_loss', type=float, default=0.6, help='prototype learning loss')

    parser.add_argument('-loss_type', type=str, default='CE', help='Type of loss function')
    parser.add_argument('-optim', type=str, default='adam', help='Optimizer: adam, sgd, adamw')
    parser.add_argument('-activate_fun', type=str, default='gelu', help='Activation function')
    parser.add_argument('-image_model', type=str, default='resnet-50', help='Image model: resnet-18, resnet-34, resnet-50, resnet-101, resnet-152')

    parser.add_argument('-image_output_type', type=str, default='all', help='"all" represents the overall features and regional features of the picture, and "CLS" represents the overall features of the picture')
    parser.add_argument('-text_length_dynamic', type=int, default=0, help='1: Dynamic length; 0: fixed length')
    parser.add_argument('-fuse_type', type=str, default='max', help='att, ave, max')
    parser.add_argument('-temperature', type=float, default=0.07, help='Temperature used to calculate contrastive learning loss')
    parser.add_argument('-classes', type=int, default=3, help='label')
    parser.add_argument('-cuda', action='store_true', default=False, help='if True: use cuda. if False: use cpu')
    parser.add_argument('-fixed_image_model', action='store_true', default=False, help='是否固定图像模型的参数')
    parser.add_argument('-image_size', type=int, default=224, help='Image dim')
    
    parser.add_argument('--num_cluster', type=int, nargs='+', default=[500, 1000, 1500], help='Number of clusters')
    parser.add_argument('--Niter', type=int, default=20, help='Number of iterations for clustering')

    # Adding img_hidden_dim and txt_hidden_dim parameters
    parser.add_argument('-img_hidden_dim', type=int, nargs='+', default=[1024, 512, 256], help='Dimensions of image hidden layers')
    parser.add_argument('-txt_hidden_dim', type=int, nargs='+', default=[768, 512, 256], help='Dimensions of text hidden layers')

    parser.add_argument('-nbit', type=int, default=128, help='Number of bits for hashing')
    parser.add_argument('-alpha', type=float, default=0.5, help='Alpha value for feature fusion')  # 添加 alpha 参数
    parser.add_argument('-beta', type=float, default=0.5, help='Beta value for feature fusion')  # 添加 beta 参数

    parser.add_argument('-x', type=float, help='Description for x')
    parser.add_argument('-y', type=float, help='Description for y')
    parser.add_argument('-z', type=float, help='Description for z')

    parser.add_argument('-dropout', type=float, default=0.1, help='Dropout rate')  # 添加 dropout 参数

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_num)

    # 区分不同时间点的模型
    dt = datetime.now()
    opt.save_model_path = opt.save_model_path + '/' + dt.strftime('%Y-%m-%d-%H-%M-%S') + '-'
    if opt.add_note != '':
        opt.save_model_path += opt.add_note + '-'
    print('\n', opt.save_model_path, '\n')

    assert opt.batch_size % opt.acc_grad == 0
    opt.acc_batch_size = opt.batch_size // opt.acc_grad

    # CE（交叉熵），SCE（标签平滑的交叉熵），Focal（focal loss），Ghm（ghm loss）
    critertion = None
    if opt.loss_type == 'CE':
        critertion = nn.CrossEntropyLoss()

    # # 实例化 pllearning 类
    # pllearning_instance = model.pllearning(temperature=opt.temperature, devices=opt.cuda)

    pl_fuse_model = model.FuseModel(opt)
    print(torch.cuda.is_available())
    
    if opt.cuda is True:
        assert torch.cuda.is_available()
        if len(opt.gpu_num) > 1:
            if opt.gpu0_bsz > 0:
                pl_fuse_model = nn.DataParallel(pl_fuse_model).cuda()
            else:
                print('multi-gpu')
                """
                单机多卡的运行方式，nproc_per_node表示使用的GPU的数量
                python -m torch.distributed.launch --nproc_per_node=2 main.py
                """
                print('当前GPU编号：', opt.local_rank)
                torch.cuda.set_device(opt.local_rank) # 在进行其他操作之前必须先设置这个
                torch.distributed.init_process_group(backend='nccl')
                pl_fuse_model = pl_fuse_model.cuda()
                pl_fuse_model = nn.parallel.DistributedDataParallel(pl_fuse_model, find_unused_parameters=True)
        else:
            pl_fuse_model = pl_fuse_model.cuda()
        critertion = critertion.cuda()

    print('Init Data Process:')
    tokenizer = None
    abl_path = '../'
    if opt.text_model == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased/vocab.txt')

    # 数据加载
    if opt.data_type == 'HFM':
        data_path_root = abl_path + 'dataset/data/HFM/'
        train_data_path = data_path_root + 'train.json'
        dev_data_path = data_path_root + 'valid.json'
        test_data_path = data_path_root + 'test.json'
        photo_path = data_path_root + 'dataset_image'
        image_coordinate = None
        data_translation_path = data_path_root + '/HFM.json'
    else:
        data_path_root = abl_path + 'dataset/data/' + opt.data_type + '/' + opt.data_path_name + '/'
        train_data_path = data_path_root + 'train.json'
        dev_data_path = data_path_root + 'dev.json'
        test_data_path = data_path_root + 'test.json'
        photo_path = abl_path + 'dataset/data/' + opt.data_type + '/dataset_image'
        image_coordinate = None
        data_translation_path = abl_path + 'dataset/data/' + opt.data_type + '/' + opt.data_type + '_translation.json'

    # data_type:标识数据的类型，1是训练数据，2是开发集，3是测试数据
    train_loader, opt.train_data_len = data_process.data_process(opt, train_data_path, tokenizer, photo_path, data_type=1)
    dev_loader, opt.dev_data_len = data_process.data_process(opt, dev_data_path, tokenizer, photo_path, data_type=2)
    test_loader, opt.test_data_len = data_process.data_process(opt, test_data_path, tokenizer, photo_path, data_type=3)


    if opt.warmup_step_epoch > 0:
        opt.warmup_step = opt.warmup_step_epoch * len(train_loader)
        opt.warmup_num_lr = opt.lr / opt.warmup_step
    opt.scheduler_step_epoch = opt.epoch - opt.warmup_step_epoch

    if opt.scheduler_step_epoch > 0:
        opt.scheduler_step = opt.scheduler_step_epoch * len(train_loader)
        opt.scheduler_num_lr = opt.lr / opt.scheduler_step

    opt.save_model_path = WriteFile(opt.save_model_path, 'train_correct_log.txt', str(opt) + '\n\n', 'a+', change_file_name=True)
    log_summary_writer = SummaryWriter(log_dir=opt.save_model_path)
    log_summary_writer.add_text('Hyperparameter', str(opt), global_step=1)
    log_summary_writer.flush()

    # 训练
    if opt.run_type == 1:
        print('\nTraining Begin')
        train_process.train_process(opt, train_loader, dev_loader, test_loader, pl_fuse_model, critertion, opt.train_data_len, log_summary_writer)
        # train_process.train_process(opt, train_loader, dev_loader, test_loader, pl_fuse_model, critertion, pllearning_instance, opt.train_data_len, log_summary_writer)
    elif opt.run_type == 2:
        print('\nTest Begin')
        model_path = "checkpoint/best_model/best-model.pth"
        pl_fuse_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        test_process.test_process(opt, critertion, pl_fuse_model, test_loader, epoch=1)

    log_summary_writer.close()
