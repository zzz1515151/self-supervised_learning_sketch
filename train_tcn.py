from __future__ import print_function
import sys
sys.path.append("../")
import argparse
import os
import torch 
import time
import imp
import numpy as np
import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from utils.AverageMeter import AverageMeter
from utils.logger import logger 

from TCN import TCN
from dataloader.dataloader_stroke import RNNDataset, DataLoader

parser = argparse.ArgumentParser(description='traning TCN')
parser.add_argument("--exp", type = str, default = "", help = "experiment")
parser.add_argument("--num_workers", type = int, default = 4, help = "num_workers")
parser.add_argument("--checkpoint", type = int, default = 0, help = "load checkpoint")
parser.add_argument('--gpu', type = str, default = "0", help = 'choose GPU')
args = parser.parse_args()

exp_config = os.path.join(".", "config", args.exp + ".py")
exp_dir = os.path.join("./exp", "experiments", args.exp)
exp_log_dir = os.path.join(exp_dir, "log")
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)
exp_visual_dir = os.path.join(exp_dir, "visual")
if not os.path.exists(exp_visual_dir):
    os.makedirs(exp_visual_dir)


config = imp.load_source("", exp_config).config
#tensorboard && logger
now_str = datetime.datetime.now().__str__().replace(' ','_')
writer_path = os.path.join(exp_visual_dir, now_str)
writer = SummaryWriter(writer_path)

logger_path = os.path.join(exp_log_dir, now_str + ".log")
logger = logger(logger_path).get_logger()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

data_train_opt = config['data_train_opt']
data_test_opt = config['data_test_opt']

# init dataloader 
signal_type = config['signal_type']

logger.info("initing dataloader")

train_dataset = RNNDataset(root = data_train_opt["root"], split = data_train_opt["split"])
trainloader = DataLoader(dataset = train_dataset, 
                signal_type = signal_type,
                batch_size = data_train_opt["batch_size"], \
                num_workers = args.num_workers, shuffle = True)

test_dataset = RNNDataset(root = data_test_opt["root"], split = data_test_opt["split"])
testloader = DataLoader(dataset = test_dataset, 
                signal_type = signal_type,
                batch_size = data_test_opt["batch_size"], \
                num_workers = args.num_workers, 
                shuffle = False)
logger.info("dataloader OK!")


network = TCN(config["net_opt"])
if args.checkpoint > 0 :
    net_checkpoint_name = args.exp + "_net_epoch" + args.checkpoint
    net_checkpoint_path = os.path.join(exp_dir, net_checkpoint_name)
    assert(os.path.exists(net_checkpoint_path))
    try:        
        checkpoint = torch.load(net_checkpoint_path)
        network.load_state_dict(checkpoint["network"])
        logger.info("Load net checkpoint epoch {}".format(args.checkpoint))
    except:
        logger.info("Can not load checkpoint from {}".format(net_checkpoint_path))

network = network.cuda()

optim_opt = config["optim_opt"]
if optim_opt["name"] == "Adam":
    optimizer = optim.Adam(network.parameters(), lr = optim_opt["lr"])
elif optim_opt['name'] == 'SGD':
    optimizer = optim.SGD(network.parameters(), lr = optim_opt["lr"], \
                momentum = optim_opt["momentum"], \
                nesterov = optim_opt["nesterov"] if ('nesterov' in optim_opt) else False,\
                weight_decay=optim_opt['weight_decay'])
else:
    raise ValueError("Don't surport optimizer:{}".format(optim_opt["name"]))

if args.checkpoint > 0 :
    optim_checkpoint_name = args.exp + "_optim_epoch" + args.checkpoint
    optim_checkpoint_path = os.path.join(exp_dir, optim_checkpoint_name)
    assert(os.path.exists(optim_checkpoint_path))
    try:
        checkpoint = torch.load(optim_checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Load optimizer checkpoint epoch {}".format(args.checkpoint))
    except:
        logger.info("Can not load checkpoint from {}".format(optim_checkpoint_path))


lr_protocol = optim_opt["lr_protocol"]
loss = nn.CrossEntropyLoss()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

train_model_loss = AverageMeter()
train_acc = AverageMeter()
test_model_loss = AverageMeter()
test_acc = AverageMeter()


def train(epoch): 
    train_model_loss.reset()
    train_acc.reset()
    network.train()
    
    lr = next((lr for (max_epoch, lr) in lr_protocol if max_epoch>epoch), lr_protocol[-1][1]) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr     
    logger.info("Setting learning rate to: {}".format(lr))
    
    for idx, batch in enumerate(tqdm(trainloader(epoch))):
        #start = time.time()
        
        data = batch[0].cuda()
        target = batch[1].cuda()
        
        optimizer.zero_grad()
        
        output = network(data)
        #fc_output = network(data, ["fc_block"])
        #writer.add_histogram("train", fc_output.data, epoch * len(dloader_train) + idx)
        
        loss_total = loss(output, target)
        
        loss_total.backward()        
        
        optimizer.step()
        
        train_model_loss.update(loss_total.item())
        train_acc.update(accuracy(output, target, topk = (1,))[0][0])     
        if (idx+1) % config["display_step"] == 0:            
            logger.info("==> Iteration [{}][{}/{}]:".format(epoch+1, idx+1, len(trainloader)))
            logger.info("current loss:{}".format(loss_total.item()))
            #logger.info("grad norm:{}".format(total_grad_norm))
            logger.info("loss: {}  acc: {}".format(train_model_loss.avg, train_acc.avg))               
    logger.info("Begin Evaluating")    
    test(epoch)
    writer.add_scalars("loss", {
        "train_loss":train_model_loss.avg,
        "test_loss":test_model_loss.avg
        }, epoch+1)
    writer.add_scalars("acc", {
        "train_acc":train_acc.avg,
        "test_acc":test_acc.avg
        }, epoch+1)

def test(epoch):
    test_model_loss.reset()
    test_acc.reset()
    network.eval()
    for idx, batch in enumerate(tqdm(testloader(epoch))):
        data = batch[0].cuda()
        target = batch[1].cuda()
        output = network(data)
        #fc_output = network(data, ["fc_block"]) 
        #writer.add_histogram("test", fc_output.data, epoch * len(dloader_test) + idx)       
        loss_total = loss(output, target)
        test_model_loss.update(loss_total.item())
        test_acc.update(accuracy(output, target, topk = (1,))[0][0])    
    logger.info("==> Evaluation Result: ")
    logger.info("loss: {}  acc:{}".format(test_model_loss.avg, test_acc.avg))

best_test_acc = 0
best_epoch = None

logger.info("training Status: ") 
logger.info(config) 
assert(args.checkpoint < config['num_epochs'])
for epoch in range(args.checkpoint, config['num_epochs']):
    logger.info("Experiment:{}".format(args.exp))
    logger.info("Begin training epoch {}".format(epoch+1))
    train(epoch)        
    #save checkpoint
    net_checkpoint_name = args.exp + "_net_epoch" + str(epoch+1)
    net_checkpoint_path = os.path.join(exp_dir, net_checkpoint_name)
    net_state = {"epoch": epoch+1, "network": network.state_dict()}
    torch.save(net_state, net_checkpoint_path)

    optim_checkpoint_name = args.exp + "_optim_epoch" + str(epoch+1)
    optim_checkpoint_path = os.path.join(exp_dir, optim_checkpoint_name)
    optim_state = {"epoch": epoch+1, "optimizer": optimizer.state_dict()}
    torch.save(optim_state, optim_checkpoint_path)
    #delete previous checkpoint

    if not epoch == args.checkpoint:
    #net_checkpoint_name_del = args.exp + "_net_epoch" + str(epoch)
    #net_checkpoint_path_del = os.path.join(exp_dir, net_checkpoint_name_del)
        optim_checkpoint_name_del = args.exp + "_optim_epoch" + str(epoch)
        optim_checkpoint_path_del = os.path.join(exp_dir, optim_checkpoint_name_del)
        '''
        if os.path.exists(net_checkpoint_path_del):
            os.remove(net_checkpoint_path_del)
        '''
        if os.path.exists(optim_checkpoint_path_del):
            os.remove(optim_checkpoint_path_del)

    #save best model and delete previous best model
    if test_acc.avg > best_test_acc:
        if not best_epoch == None:
            net_best_name_del = args.exp + "_net_epoch" + str(best_epoch+1) + ".best"
            net_best_path_del = os.path.join(exp_dir, net_best_name_del)
            if os.path.exists(net_best_path_del):
                os.remove(net_best_path_del)
        best_epoch = epoch
        best_test_acc = test_acc.avg
        net_best_name = args.exp + "_net_epoch" + str(epoch+1) + ".best"
        net_best_path = os.path.join(exp_dir, net_best_name)
        net_best_state = {"epoch": epoch+1, "best_acc": best_test_acc, \
                           "network": network.state_dict()}
        torch.save(net_best_state, net_best_path)
        logger.info("Saving model best with acc:{}".format(best_test_acc))

















