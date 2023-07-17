import time
import math
from datetime import timedelta
import torch
from torch import nn as nn
### Changjae Lee @ 2022-09-14 
#from nni.nas.pytorch.utils import AverageMeter 
# https://nni.readthedocs.io/en/stable/_modules/nni/retiarii/oneshot/pytorch/utils.html?highlight=averagemeter 
from nni.retiarii.oneshot.pytorch.utils import AverageMeter
### Changjae Lee @ 2022-09-17 
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html 
import torch.onnx 
import numpy as np 
import onnx 
import onnxruntime 
### Changjae Lee @ 2022-09-22 
from sklearn.metrics import confusion_matrix 
# https://torchmetrics.readthedocs.io/en/stable/classification/fbeta_score.html 
from torchmetrics import FBetaScore 

def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

### Changjae Lee @ 2022-09-19 
# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

### Changjae Lee @ 2022-09-19 
def bin_accuracy(output, target): 
    batch_size = target.size(0)
    correct = 0 
    
    # output (batch_size, 2) 
    # labels (batch_size, ) 
    
    _, pred = output.topk(1, 1, True, True)
    #pred = pred.t()
    #correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct += pred.eq(target.view_as(pred)).sum() 
    correct = correct.float().mul_(100.0 / batch_size)
    return [correct] 

### Changjae Lee @ 2022-09-22 
def bin_f_beta(output, target, device): 
    _, pred = output.topk(1, 1, True, True) 
    f_beta = FBetaScore(num_classes=2, beta=2).to(device) 
    F_beta_score = f_beta(pred.view(-1), target.view(-1))
    # https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html 
    # C = confusion_matrix(target.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy()) 
    # precision = C[1][1] / (C[1][1] + C[0][1])
    # sensitivity = C[1][1] / (C[1][1] + C[1][0]) 
    # F_beta_score = (1+2**2) * (precision * sensitivity) / ((2**2)*precision + sensitivity)
    # https://pytorch.org/docs/stable/generated/torch.from_numpy.html 
    #return [torch.from_numpy(F_beta_score).float().mul_(100.0)]
    return [F_beta_score.float().mul_(100.0)] 

### Changjae Lee @ 2022-09-17 
# class Retrain:
#     def __init__(self, model, optimizer, device, data_provider, n_epochs):
#         self.model = model
#         self.optimizer = optimizer
#         self.device = device
#         self.train_loader = data_provider.train
#         self.valid_loader = data_provider.valid
#         self.test_loader = data_provider.test
#         self.n_epochs = n_epochs
#         self.criterion = nn.CrossEntropyLoss()

#     def run(self):
#         self.model = torch.nn.DataParallel(self.model)
#         self.model.to(self.device)
#         # train
#         self.train()
#         # validate
#         self.validate(is_test=False)
#         # test
#         self.validate(is_test=True)

#     def train_one_epoch(self, adjust_lr_func, train_log_func, label_smoothing=0.1):
#         batch_time = AverageMeter('batch_time')
#         data_time = AverageMeter('data_time')
#         losses = AverageMeter('losses')
#         top1 = AverageMeter('top1')
#         top5 = AverageMeter('top5')
#         self.model.train()
#         end = time.time()
#         for i, (images, labels) in enumerate(self.train_loader):
#             data_time.update(time.time() - end)
#             new_lr = adjust_lr_func(i)
#             images, labels = images.to(self.device), labels.to(self.device)
#             output = self.model(images)
#             if label_smoothing > 0:
#                 loss = cross_entropy_with_label_smoothing(output, labels, label_smoothing)
#             else:
#                 loss = self.criterion(output, labels)
#             acc1, acc5 = accuracy(output, labels, topk=(1, 5))
#             losses.update(loss, images.size(0))
#             top1.update(acc1[0], images.size(0))
#             top5.update(acc5[0], images.size(0))

#             # compute gradient and do SGD step
#             self.model.zero_grad()  # or self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % 10 == 0 or i + 1 == len(self.train_loader):
#                 batch_log = train_log_func(i, batch_time, data_time, losses, top1, top5, new_lr)
#                 print(batch_log)
#         return top1, top5

#     def train(self, validation_frequency=1):
#         best_acc = 0
#         nBatch = len(self.train_loader)

#         def train_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
#                 batch_log = 'Train [{0}][{1}/{2}]\t' \
#                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
#                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
#                             'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
#                             'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
#                     format(epoch_ + 1, i, nBatch - 1,
#                         batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
#                 batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
#                 batch_log += '\tlr {lr:.5f}'.format(lr=lr)
#                 return batch_log
        
#         def adjust_learning_rate(n_epochs, optimizer, epoch, batch=0, nBatch=None):
#             """ adjust learning of a given optimizer and return the new learning rate """
#             # cosine
#             T_total = n_epochs * nBatch
#             T_cur = epoch * nBatch + batch
#             # init_lr = 0.05
#             new_lr = 0.5 * 0.05 * (1 + math.cos(math.pi * T_cur / T_total))
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = new_lr
#             return new_lr

#         for epoch in range(self.n_epochs):
#             print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
#             end = time.time()
#             train_top1, train_top5 = self.train_one_epoch(
#                 lambda i: adjust_learning_rate(self.n_epochs, self.optimizer, epoch, i, nBatch),
#                 lambda i, batch_time, data_time, losses, top1, top5, new_lr:
#                 train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
#             )
#             time_per_epoch = time.time() - end
#             seconds_left = int((self.n_epochs - epoch - 1) * time_per_epoch)
#             print('Time per epoch: %s, Est. complete in: %s' % (
#                 str(timedelta(seconds=time_per_epoch)),
#                 str(timedelta(seconds=seconds_left))))
            
#             if (epoch + 1) % validation_frequency == 0:
#                 val_loss, val_acc, val_acc5 = self.validate(is_test=False)
#                 is_best = val_acc > best_acc
#                 best_acc = max(best_acc, val_acc)
#                 val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'.\
#                     format(epoch + 1, self.n_epochs, val_loss, val_acc, best_acc)
#                 val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'.\
#                     format(val_acc5, top1=train_top1, top5=train_top5)
#                 print(val_log)
#             else:
#                 is_best = False

#     def validate(self, is_test=True):
#         if is_test:
#             data_loader = self.test_loader
#         else:
#             data_loader = self.valid_loader
#         self.model.eval()
#         batch_time = AverageMeter('batch_time')
#         losses = AverageMeter('losses')
#         top1 = AverageMeter('top1')
#         top5 = AverageMeter('top5')

#         end = time.time()
#         with torch.no_grad():
#             for i, (images, labels) in enumerate(data_loader):
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 # compute output
#                 output = self.model(images)
#                 loss = self.criterion(output, labels)
#                 # measure accuracy and record loss
#                 acc1, acc5 = accuracy(output, labels, topk=(1, 5))
#                 losses.update(loss, images.size(0))
#                 top1.update(acc1[0], images.size(0))
#                 top5.update(acc5[0], images.size(0))
#                 # measure elapsed time
#                 batch_time.update(time.time() - end)
#                 end = time.time()

#                 if i % 10 == 0 or i + 1 == len(data_loader):
#                     if is_test:
#                         prefix = 'Test'
#                     else:
#                         prefix = 'Valid'
#                     test_log = prefix + ': [{0}/{1}]\t'\
#                                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
#                                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
#                                         'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'.\
#                         format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
#                     test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
#                     print(test_log)
#         return losses.avg, top1.avg, top5.avg

### Changjae Lee @ 2022-09-17 
class Retrain: 
    ### Changjae Lee @ 2022-09-17 
    # X -> checkpoint_path 
    def __init__(self, model, optimizer, device, data_provider, n_epochs, checkpoint_path):
        self.model = model 
        self.optimizer = optimizer 
        self.device = device
        self.train_loader = data_provider.train
        self.valid_loader = data_provider.valid
        self.test_loader = data_provider.test
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        ### Changjae Lee @ 2022-09-17 
        self.checkpoint_path = checkpoint_path 
        ### Changjae Lee @ 2022-09-22 
        self.f_beta = FBetaScore(num_classes=2, beta=2).to(self.device)

    def run(self):
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        # train
        ### Changjae Lee @ 2022-09-17 
        self.train()
        #self.save(None)
        #self.onnx_test(True)
        # validate
        self.validate(is_test=False)
        # test
        self.validate(is_test=True)

    def train_one_epoch(self, adjust_lr_func, train_log_func, label_smoothing=0.1):
        batch_time = AverageMeter('batch_time')
        data_time = AverageMeter('data_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        ### Changjae Lee @ 2022-09-17 
        #top5 = AverageMeter('top5')
        self.model.train()
        ### Changjae Lee @ 2022-09-22 
        # https://www.freecodecamp.org/news/pytorch-tensor-methods/ 
        # predList = torch.Tensor([]).long().to(self.device)

        end = time.time()
        for i, (images, labels) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            if label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(output, labels, label_smoothing)
            else:
                loss = self.criterion(output, labels)
            
            ### Changjae Lee @ 2022-09-17 
            #acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            ### Changjae Lee @ 2022-09-19 
            #acc1 = accuracy(output, labels, topk=(1,))
            ### Changjae Lee @ 2022-09-22 
            acc1 = bin_accuracy(output, labels)
            #acc1 = bin_f_beta(output, labels)
            #acc1 = [self.f_beta(output, labels)] 
            # _, pred = output.topk(1, 1, True, True) 
            # predList = torch.cat((predList, pred.view(-1)), 0) 
            #F_beta_score = self.f_beta(labels.view(-1), pred.view(-1)) 
            #acc1 = [F_beta_score.float().mul_(100.0)] 

            losses.update(loss, images.size(0))
            top1.update(acc1[0], images.size(0))
            ### Changjae Lee @ 2022-09-17 
            #top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.model.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0 or i + 1 == len(self.train_loader):
                ### Changjae Lee @ 2022-09-17 
                batch_log = train_log_func(i, batch_time, data_time, losses, top1, new_lr)
                print(batch_log)
        
        ### Changjae Lee @ 2022-09-22 
        ### Changjae Lee @ 2022-09-23 
        # C = confusion_matrix(self.train_loader.dataset.y.detach().cpu().numpy(), predList.detach().cpu().numpy()) 
        # precision = C[1][1] / (C[1][1] + C[0][1])
        # sensitivity = C[1][1] / (C[1][1] + C[1][0]) 
        # F_beta_score = (1+2**2) * (precision * sensitivity) / ((2**2)*precision + sensitivity)
        
        ### Changjae Lee @ 2022-09-17 
        #return top1, top5 
        ### Changjae Lee @ 2022-09-22 
        return top1 
        # return top1, F_beta_score 

    def train(self, validation_frequency=1):
        ### Changjae Lee @ 2022-09-22 
        best_f_beta = 0
        ### Changjae Lee @ 2022-09-18 
        #best_acc_cls = 0 
        nBatch = len(self.train_loader)
        ### Changjae Lee @ 2022-09-17 
        # top5 -> X 
        def train_log_func(epoch_, i, batch_time, data_time, losses, top1, lr):
            ### Changjae Lee @ 2022-09-17 
            #print(f'{type(epoch_ + 1)} {type(i)} {type(nBatch - 1)}') # int int int 
            #print(f'{type(batch_time)} {type(data_time)} {type(losses)} {type(top1)}') # <class 'nni.retiarii.oneshot.pytorch.utils.AverageMeter'> 
            #print(f'{batch_time.val} {data_time.val} {losses.val} {top1.val.item()}')
            #print(f'{losses.val}')
            #print(f'{type(top1.val)} {top1.val.item()}')
            #print(f'{type(top1.avg)} {top1.avg.item()}')
            #print(f'{vars(top1)}')
            #print(f'{vars(dict(top1))}')
            
            #print('{top1.val:.3f} {top1.avg:.3f}'.format(top1=top1))
                            
#                 batch_log = 'Train [{0}][{1}/{2}]\t' \
#                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
#                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
#                             'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
#                             'Top-1 acc {top1.val.item():.3f} ({top1.avg.item():.3f})'. \
#                     format(epoch_ + 1, i, nBatch - 1,
#                         batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            ### Changjae Lee @ 2022-09-19 
            # Top-1 acc -> Acc 
            batch_log = f'Train [{epoch_ + 1}][{i}/{nBatch - 1}]\t' \
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                        f'Acc {top1.val.item():.3f} ({top1.avg.item():.3f})'
            #print(batch_log)
            ### Changjae Lee @ 2022-09-17 
#                 batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log
        
        def adjust_learning_rate(n_epochs, optimizer, epoch, batch=0, nBatch=None):
            """ adjust learning of a given optimizer and return the new learning rate """
            # cosine
            T_total = n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            # init_lr = 0.05
            new_lr = 0.5 * 0.05 * (1 + math.cos(math.pi * T_cur / T_total))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr

        for epoch in range(self.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            end = time.time()
            ### Changjae Lee @ 2022-09-17 
#             train_top1, train_top5 = self.train_one_epoch(
#                 lambda i: adjust_learning_rate(self.n_epochs, self.optimizer, epoch, i, nBatch),
#                 lambda i, batch_time, data_time, losses, top1, top5, new_lr:
#                 train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
#             )
            # train_top5, top5 -> X 
            ### Changjae Lee @ 2022-09-22 
            # X -> F_beta_score -> X 
            train_top1 = self.train_one_epoch(
                lambda i: adjust_learning_rate(self.n_epochs, self.optimizer, epoch, i, nBatch),
                lambda i, batch_time, data_time, losses, top1, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1, new_lr),
            )
            time_per_epoch = time.time() - end
            seconds_left = int((self.n_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))
            
            if (epoch + 1) % validation_frequency == 0:
                ### Changjae Lee @ 2022-09-17 
                # val_acc5 -> X 
                ### Changjae Lee @ 2022-09-22 
                val_loss, val_f_beta = self.validate(is_test=False)
                #print(f'\n\nreturn value of validate: {val_f_beta}\n\n')
                is_best = val_f_beta > best_f_beta
                best_f_beta = max(best_f_beta, val_f_beta) 
                ### Changjae Lee @ 2022-09-19 
                # top-1 acc -> Acc 
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\tF_beta {3:.3f} ({4:.3f})'.\
                    format(epoch + 1, self.n_epochs, val_loss, val_f_beta, best_f_beta)
                ### Changjae Lee @ 2022-09-17 
#                 val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'.\
#                     format(val_acc5, top1=train_top1, top5=train_top5)
                ### Changjae Lee @ 2022-09-17 
#                 val_log += '\tTrain top-1 {top1.avg:.3f}'.\
#                     format(top1=train_top1)
                ### Changjae Lee @ 2022-09-19 
                # top-1 -> Acc 
                val_log += f'\tTrain Acc {train_top1.avg.item():.3f}' 
                print(val_log)
            else:
                is_best = False
            
            ### Changjae Lee @ 2022-09-17 
            if is_best: 
                #print(best_f_beta)
                self.save(best_f_beta)
                ### Changjae Lee @ 2022-09-22 
                #self.onnx_test(best_acc, True) 
                self.onnx_test_f_beta(best_f_beta, True)

    def validate(self, is_test=True):
        if is_test:
            data_loader = self.test_loader
        else:
            data_loader = self.valid_loader
        self.model.eval()
        batch_time = AverageMeter('batch_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        ### Changjae Lee @ 2022-09-17 
        #top5 = AverageMeter('top5') 
        ### Changjae Lee @ 2022-09-22 
        predList = torch.Tensor([]).long().to(self.device)

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = self.model(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                ### Changjae Lee @ 2022-09-17 
                #acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                ### Changjae Lee @ 2022-09-19 
                #acc1 = accuracy(output, labels, topk=(1,))
                ### Changjae Lee @ 2022-09-22 
                acc1 = bin_accuracy(output, labels)
                #acc1 = bin_f_beta(output, labels)
                #acc1 = [self.f_beta(output, labels)] 
                _, pred = output.topk(1, 1, True, True) 
                predList = torch.cat((predList, pred.view(-1)), 0)
                #F_beta_score = self.f_beta(labels.view(-1), pred.view(-1)) 
                #acc1 = [F_beta_score.float().mul_(100.0)] 

                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                ### Changjae Lee @ 2022-09-17 
                #top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0 or i + 1 == len(data_loader):
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    ### Changjae Lee @ 2022-09-17 
#                     test_log = prefix + ': [{0}/{1}]\t'\
#                                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
#                                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
#                                         'Top-1 acc {top1.val.item():.3f} ({top1.avg.item():.3f})'.\
#                         format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    #print(f'{type(batch_time)} {type(loss)} {type(losses)} {type(top1)}')
                    ### Changjae Lee @ 2022-09-19 
                    # Top-1 acc -> Acc 
                    test_log = prefix + f': [{i}/{len(data_loader) - 1}]\t'\
                                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'\
                                        f'Acc {top1.val.item():.3f} ({top1.avg.item():.3f})'
                    ### Changjae Lee @ 2022-09-17 
                    #test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    print(test_log)
        ### Changjae Lee @ 2022-09-17 
        #return losses.avg, top1.avg, top5.avg
        #return losses.avg, top1.avg.item() 

        ### Changjae Lee @ 2022-09-23 
        C = confusion_matrix(data_loader.dataset.y.detach().cpu().numpy(), predList.detach().cpu().numpy()) 
        precision = C[1][1] / (C[1][1] + C[0][1])
        sensitivity = C[1][1] / (C[1][1] + C[1][0]) 
        F_beta_score = (1+2**2) * (precision * sensitivity) / ((2**2)*precision + sensitivity)
        #print(f'\n\nConfusion_matrix: {F_beta_score}\n\n')

        # F_beta_score = self.f_beta(predList.to(self.device), data_loader.dataset.y.to(self.device)).detach().cpu().numpy() 
        # print(f'\n\nself.f_beta: {F_beta_score}\n\n')

        return losses.avg, F_beta_score 
    
    ### Changjae Lee @ 2022-09-22 
    def save(self, best_f_beta): 
        # https://stackoverflow.com/questions/3257919/what-is-the-difference-between-is-none-and-none 
        if best_f_beta is None: 
            best_f_beta = 0.
        # with open(self.checkpoint_path + f'search_tinyml_net_f_beta_{best_f_beta*100:.3f}.txt', 'w') as f: 
        #     pass 
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference 
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models 
        #torch.save(self.model.module.state_dict(), self.checkpoint_path + 'search_tinyml_net.pt') 
        
        # https://towardsdatascience.com/best-practices-for-neural-network-exports-to-onnx-99f23006c1d5 
        if isinstance(self.model, torch.nn.DataParallel):  # extract the module from dataparallel models
            self.model = self.model.module
        self.model.cpu()
        self.model.eval()
        torch.save(self.model, self.checkpoint_path + 'search_tinyml_net.pt') 
        # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html 
        # Input to the model
        ### Changjae Lee @ 2022-09-22 
        # 1, 1250 -> 1, 1, 1250 
        self.temp_x = torch.randn(1, 1, 1250, requires_grad=True).float().cpu()
        self.temp_out = self.model(self.temp_x)
        #print(f'{self.temp_x} {self.temp_out}')
        #print(f'{self.model(self.temp_x)} {self.model(self.temp_x)}')
        #print(f'{ onnx.__version__ }')
        #print(f'{self.model}')
        #print(f'{self.model.module}')
        # https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch 
        temp_model = torch.load(self.checkpoint_path + 'search_tinyml_net.pt')
        #model.load_state_dict(state_dict)
        temp_model = temp_model.cpu() 
        temp_model.eval()
        #dummy_input = Variable(torch.randn(B, C, H, W))
        ### Changjae Lee @ 2022-09-22 
        # for STM32CubeMX 
        #torch.onnx.export(temp_model, self.temp_x, self.checkpoint_path + 'search_tinyml_net.onnx', export_params = True, opset_version=16)
        torch.onnx.export(temp_model, self.temp_x, self.checkpoint_path + 'search_tinyml_net.onnx', input_names=['input'], opset_version=13) 

        # Export the model
#         torch.onnx.export(self.model.cpu(),                # model being run
#                           self.temp_x.cpu(),                    # model input (or a tuple for multiple inputs)
#                           self.checkpoint_path + 'search_tinyml_net.onnx',  # where to save the model (can be a file or file-like object)
#                           export_params=True,        # store the trained parameter weights inside the model file
#                           opset_version=16,          # the ONNX version to export the model to # https://pytorch.org/docs/master/onnx.html 
#                           do_constant_folding=True,  # whether to execute constant folding for optimization
#                           input_names = ['input'],   # the model's input names
#                           output_names = ['output'], # the model's output names
#                           dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                         'output' : {0 : 'batch_size'}})

        #self.model.module.to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        
    ### Changjae Lee @ 2022-09-18 
    # best_acc, classification=False 
    ### Changjae Lee @ 2022-09-19 
    # best_acc, path_name, classification=False 
    def onnx_test(self, best_acc, classification=False): 
        onnx_model = onnx.load(self.checkpoint_path + 'search_tinyml_net.onnx')
        onnx.checker.check_model(onnx_model) 
        #print(onnx_model)
        
        ort_session = onnxruntime.InferenceSession(self.checkpoint_path + 'search_tinyml_net.onnx')

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy() 

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(self.temp_x)}
        ort_outs = ort_session.run(None, ort_inputs)
        #print(f'{ort_inputs} {ort_outs}') 
        
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(self.temp_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        if classification: 
            correct = 0 
            for X_test, y_test in self.test_loader:
                #print(X_test.shape)
                for x, y in zip(X_test, y_test): 
                    #print(x.shape)
                    ### Changjae Lee @ 2022-09-22 
                    # x.reshape(1, -1) -> x.reshape(1, 1, -1) 
                    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x.reshape(1, 1, -1))}
                    
                    ort_outs = ort_session.run(None, ort_inputs)
                    pred = ort_outs[0][0] 
                    pred = 0 if pred[0] > pred[1] else 1 
                    if (pred == y): 
                        correct = correct + 1  
                        
            ### Changjae Lee @ 2022-09-18 
            accuracy = correct / len(self.test_loader.dataset) 
            print('\nTest set: Accuracy: {}/{} ({:.3f}%)\n'.format(
                  correct, len(self.test_loader.dataset), accuracy))
            ### Changjae Lee @ 2022-09-18 
            with open(self.checkpoint_path + f'search_tinyml_net_acc_{best_acc*100:.3f}_cls_acc_{accuracy*100:.3f}.txt', 'w') as f: 
                pass 

    ### Changjae Lee @ 2022-09-18 
    # best_acc, classification=False 
    ### Changjae Lee @ 2022-09-19 
    # best_acc, path_name, classification=False 
    # best_acc -> best_f_beta 
    def onnx_test_f_beta(self, best_f_beta, classification=False): 
        onnx_model = onnx.load(self.checkpoint_path + 'search_tinyml_net.onnx')
        onnx.checker.check_model(onnx_model) 
        #print(onnx_model)
        
        ort_session = onnxruntime.InferenceSession(self.checkpoint_path + 'search_tinyml_net.onnx')

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy() 

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(self.temp_x)}
        ort_outs = ort_session.run(None, ort_inputs)
        #print(f'{ort_inputs} {ort_outs}') 
        
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(self.temp_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        if classification: 
            correct = 0 
            ### Changjae Lee @ 2022-09-22 
            resultList = [] 
            for X_test, y_test in self.test_loader:
                #print(X_test.shape)
                for x, y in zip(X_test, y_test): 
                    #print(x.shape)
                    ### Changjae Lee @ 2022-09-22 
                    # x.reshape(1, -1) -> x.reshape(1, 1, -1) 
                    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x.reshape(1, 1, -1))}
                    
                    ort_outs = ort_session.run(None, ort_inputs)
                    pred = ort_outs[0][0] 
                    pred = 0 if pred[0] > pred[1] else 1 
                    ### Changjae Lee @ 2022-09-22 
                    if (pred == y): 
                        correct = correct + 1  
                    resultList.append(pred)
                        
            ### Changjae Lee @ 2022-09-18 
            accuracy = correct / len(self.test_loader.dataset) 
            print('\nTest set: Accuracy: {}/{} ({:.3f}%)'.format(
                    correct, len(self.test_loader.dataset), accuracy))
            
            ### Changjae Lee @ 2022-09-22 
            C = confusion_matrix(self.test_loader.dataset.y.detach().cpu().numpy(), resultList) 
            precision = C[1][1] / (C[1][1] + C[0][1])
            sensitivity = C[1][1] / (C[1][1] + C[1][0]) 
            F_beta_score = (1+2**2) * (precision * sensitivity) / ((2**2)*precision + sensitivity)
    
            print('Test set: F_beta_score: {:.3f}%\n'.format(F_beta_score))
            
            ### Changjae Lee @ 2022-09-18 
            with open(self.checkpoint_path + f'search_tinyml_net_f_beta_{best_f_beta*100:.3f}_cls_acc_{accuracy*100:.3f}_f_beta_{F_beta_score*100:.3f}.txt', 'w') as f: 
                pass 
            