import json
import logging
import os
import sys
from argparse import ArgumentParser

import torch
#from torchvision import transforms
from nni.retiarii.fixed import fixed_arch

import datasets 
### Changjae Lee @ 2022-09-22  
# SearchMobileNet -> SearchTinyMLNet -> SearchTinyMLNet_div -> SearchTinyMLNet_e 
# -> ShuffleNetV2OneShot 
from model import ShuffleNetV2OneShot  
### Changjae Lee @ 2022-09-22 
# accuracy -> bin_accuracy -> bin_f_beta -> bin_accuracy 
from putils import LabelSmoothingLoss, bin_accuracy, get_parameters
from retrain import Retrain 
### Changjae Lee @ 2022-09-17 
import numpy as np 

logger = logging.getLogger('nni_proxylessnas')

if __name__ == "__main__":
    parser = ArgumentParser("proxylessnas")
    # configurations of the model 
    ### Changjae Lee @ 2022-09-22 
    # n_cell_stages -> stage_blocks 
    parser.add_argument("--stage_blocks", default='4,4,4,4,4,1', type=str)
    ### Changjae Lee @ 2022-09-17 
    # default='2,2,2,1,2,1' -> default='1,1,1,1,2,1' 
    ### Changjae Lee @ 2022-09-22 
    # stride_stages -> stage_strides 
    parser.add_argument("--stage_strides", default='1,1,1,1,2,1', type=str)
    ### Changjae Lee @ 2022-09-17 
    # default='24,40,80,96,192,320' -> default='8,8,16,16,24,40' 
    ### Changjae Lee @ 2022-09-22 
    # width_stages -> stage_channels 
    parser.add_argument("--stage_channels", default='8,8,16,16,24,40', type=str)
    #parser.add_argument("--bn_momentum", default=0.1, type=float)
    #parser.add_argument("--bn_eps", default=1e-3, type=float)
    #parser.add_argument("--dropout_rate", default=0, type=float)
    parser.add_argument("--no_decay_keys", default='bn', type=str, choices=[None, 'bn', 'bn#bias'])
    parser.add_argument('--grad_reg_loss_type', default='add#linear', type=str, choices=['add#linear', 'mul#log'])
    parser.add_argument('--grad_reg_loss_lambda', default=1e-1, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_alpha', default=0.2, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_beta',  default=0.3, type=float)  # grad_reg_loss_params
    parser.add_argument("--applied_hardware", default=None, type=str, help='the hardware to predict model latency')
    parser.add_argument("--reference_latency", default=None, type=float, help='the reference latency in specified hardware')
    # configurations of imagenet dataset
    ### Changjae Lee @ 2022-09-17 
    # default='/data/imagenet/' -> default='/data/tinyml/' 
    # default='/data/tinyml/' -> default='./data/tinyml/' 
    parser.add_argument("--data_path", default='./data/tinyml/', type=str)
    ### Changjae Lee @ 2022-09-17 
    # default=256 -> default=32 
    parser.add_argument("--train_batch_size", default=32, type=int)
    ### Changjae Lee @ 2022-09-17 
    # default=500 -> default=64 
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--n_worker", default=32, type=int)
    ### Changjae Lee @ 2022-09-17 
    # default=0.08 -> default=0.0 
    #parser.add_argument("--resize_scale", default=0.0, type=float)
    ### Changjae Lee @ 2022-09-17 
    # default='normal' -> default='None' 
    #parser.add_argument("--distort_color", default='None', type=str, choices=['normal', 'strong', 'None'])
    # configurations for training mode
    parser.add_argument("--train_mode", default='search', type=str, choices=['search', 'retrain'])
    # configurations for search
    ### Changjae Lee @ 2022-09-17 
    # default='./search_mobile_net.pt' -> default='./search_tinyml_net.pt' 
    # default='./search_tinyml_net.pt' -> default='./checkpoint/'
    parser.add_argument("--checkpoint_path", default='./checkpoint/', type=str)
    parser.add_argument("--arch_path", default='./arch_path.pt', type=str)
    ### Changjae Lee @ 2022-09-22 
    #parser.add_argument("--no-warmup", dest='warmup', action='store_false')
    # configurations for retrain
    parser.add_argument("--exported_arch_path", default=None, type=str)
    ### Changjae Lee @ 2022-09-17 
    parser.add_argument("--n_classes", default=2, type=int)
    parser.add_argument("--train_trainer_epochs", default=150, type=int) 
    parser.add_argument("--retrain_trainer_epochs", default=30, type=int) 
    ### Changjae Lee @ 2022-09-22 
    parser.add_argument("--first_conv_channels", default=8, type=int) 
    parser.add_argument("--last_conv_channels", default=16, type=int) 
    parser.add_argument("--pool", default='a', type=str) 

    args = parser.parse_args()
    if args.train_mode == 'retrain' and args.exported_arch_path is None:
        logger.error('When --train_mode is retrain, --exported_arch_path must be specified.')
        sys.exit(-1)

    if args.train_mode == 'retrain':
        assert os.path.isfile(args.exported_arch_path), \
            "exported_arch_path {} should be a file.".format(args.exported_arch_path)
        with fixed_arch(args.exported_arch_path):
            ### Changjae Lee @ 2022-09-17 
            # SearchMobileNet() -> SearchTinyMLNet()
            # n_classes=1000 -> n_classes=args.n_classes 
            # X -> output_size=args.output_size 
            # SearchTinyMLNet -> SearchTinyMLNet_div -> SearchTinyMLNet_e 
            # -> ShuffleNetV2OneShot 

            # X -> first_conv_channels 
            # output_size -> last_conv_channels 
            # width_stages -> stage_channels 
            # n_cell_stages -> stage_blocks 
            # stride_stages -> stage_strides 
            # dropout_rate -> X 
            # bn_param -> X 
            # X -> pool 
            model = ShuffleNetV2OneShot(first_conv_channels=args.first_conv_channels, 
                                        last_conv_channels=args.last_conv_channels, 
                                        n_classes=args.n_classes, 
                                        stage_channels=[int(i) for i in args.stage_channels.split(',')],
                                        stage_blocks=[int(i) for i in args.stage_blocks.split(',')],
                                        stage_strides=[int(i) for i in args.stage_strides.split(',')],
                                        pool=args.pool) 
    else:
        ### Changjae Lee @ 2022-09-17 
        # SearchMobileNet() -> SearchTinyMLNet() 
        # n_classes=1000 -> n_classes=args.n_classes 
        # X -> output_size=args.output_size 
        # SearchTinyMLNet -> SearchTinyMLNet_div -> SearchTinyMLNet_e 
        # -> ShuffleNetV2OneShot 
            
        # X -> first_conv_channels 
        # output_size -> last_conv_channels 
        # width_stages -> stage_channels 
        # n_cell_stages -> stage_blocks 
        # stride_stages -> stage_strides 
        # dropout_rate -> X 
        # bn_param -> X 
        # X -> pool 
        model = ShuffleNetV2OneShot(first_conv_channels=args.first_conv_channels, 
                                    last_conv_channels=args.last_conv_channels, 
                                    n_classes=args.n_classes, 
                                    stage_channels=[int(i) for i in args.stage_channels.split(',')],
                                    stage_blocks=[int(i) for i in args.stage_blocks.split(',')],
                                    stage_strides=[int(i) for i in args.stage_strides.split(',')],
                                    pool=args.pool) 
    
    ### Changjae Lee @ 2022-09-17 
    # SearchMobileNet -> SearchTinyMLNet -> ShuffleNetV2OneShot 
    # logger.info('ShuffleNetV2OneShot model create done')
    # model.init_model()
    # ### Changjae Lee @ 2022-09-17 
    # # SearchMobileNet -> SearchTinyMLNet -> ShuffleNetV2OneShot 
    # logger.info('ShuffleNetV2OneShot model init done')

    ### Changjae Lee @ 2022-09-17 
    # move network to GPU if available
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
    # https://stackoverflow.com/questions/65750273/how-to-write-nested-if-elif-else-condition-in-one-line 
    device = torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))    
        
    logger.info('Creating data provider...')
    ### Changjae Lee @ 2022-09-17 
    # ImagenetDataProvider() -> TinyMLDataProvider() 
    # resize_scale, distort_color -> X 
    data_provider = datasets.TinyMLDataProvider(save_path=args.data_path,
                                                  train_batch_size=args.train_batch_size,
                                                  test_batch_size=args.test_batch_size,
                                                  valid_size=None,
                                                  n_worker=args.n_worker)
    logger.info('Creating data provider done')

    if args.no_decay_keys:
        keys = args.no_decay_keys
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD([
            {'params': get_parameters(model, keys, mode='exclude'), 'weight_decay': 4e-5},
            {'params': get_parameters(model, keys, mode='include'), 'weight_decay': 0},
        ], lr=0.05, momentum=momentum, nesterov=nesterov)
    else:
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD(get_parameters(model), lr=0.05, momentum=momentum, nesterov=nesterov, weight_decay=4e-5)

    if args.grad_reg_loss_type == 'add#linear':
        grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
    elif args.grad_reg_loss_type == 'mul#log':
        grad_reg_loss_params = {
            'alpha': args.grad_reg_loss_alpha,
            'beta': args.grad_reg_loss_beta,
        }
    else:
        args.grad_reg_loss_params = None

    if args.train_mode == 'search':
        from nni.retiarii.oneshot.pytorch import ProxylessTrainer
        ### Changjae Lee @ 2022-09-17 
#         from torchvision.datasets import ImageNet
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#         dataset = ImageNet(args.data_path, transform=transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]))
        from help_proxylessnas import TinyML2NumpyDataset 
        dataset = TinyML2NumpyDataset(np.load(args.data_path + 'X_train_ft32.npy'), np.load(args.data_path + 'y_train_ft32.npy'))
        ### Changjae Lee @ 2022-09-17 
        # topk=(1, 5,) -> topk=(1,) 
        # loss=LabelSmoothingLoss() -> loss=torch.nn.CrossEntropyLoss()
        # dummy_input=(1, 3, 224, 224) -> dummpy_input=(1, 1250)
        # num_epochs=120 -> num_epochs=args.train_trainer_epochs 
        ### Changjae Lee @ 2022-09-19 
        # accuracy(output, target, topk=(1,)) -> bin_accuracy(output, target) -> bin_f_beta(output, target) 
        # -> bin_accuracy(output, target)
        trainer = ProxylessTrainer(model,
                                   loss=torch.nn.CrossEntropyLoss(),
                                   dataset=dataset,
                                   optimizer=optimizer,
                                   metrics=lambda output, target: bin_accuracy(output, target),
                                   num_epochs=args.train_trainer_epochs,
                                   log_frequency=10,
                                   grad_reg_loss_type=args.grad_reg_loss_type, 
                                   grad_reg_loss_params=grad_reg_loss_params, 
                                   applied_hardware=args.applied_hardware, dummy_input=(1, 1250),
                                   ref_latency=args.reference_latency)
        trainer.fit()
        print('Final architecture:', trainer.export())
        ### Changjae Lee @ 2022-09-17 
        #json.dump(trainer.export(), open('checkpoint.json', 'w'))
        json.dump(trainer.export(), open(args.checkpoint_path + 'checkpoint.json', 'w'))
    elif args.train_mode == 'retrain':
        # this is retrain
        ### Changjae Lee @ 2022-09-17 
        # X -> checkpoint_path=args.checkpoint_path 
        # n_epochs=300 -> n_epochs=args.retrain_trainer_epochs 
        trainer = Retrain(model, optimizer, device, data_provider, n_epochs=args.retrain_trainer_epochs, checkpoint_path=args.checkpoint_path)
        trainer.run()
        ### Changjae Lee @ 2022-09-17 
        #trainer.validate()