_sketch_img_root = "./sketch_image_data"
_signal_type = "rotation"
train_batch_size = 256
test_batch_size = 256

config = {}

data_train_opt = {} 
data_train_opt['root'] = _sketch_img_root
data_train_opt['batch_size'] = train_batch_size
data_train_opt['unsupervised'] = True
data_train_opt['epoch_size'] = None
data_train_opt['resize'] = False
data_train_opt['dataset_name'] = 'quickdraw_224png_20190108'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['root'] = _sketch_img_root
data_test_opt['batch_size'] = test_batch_size
data_test_opt['unsupervised'] = True
data_test_opt['epoch_size'] = None
data_test_opt['resize'] = False
data_test_opt['dataset_name'] = 'quickdraw_224png_20190108'
data_test_opt['split'] = 'test'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['num_epochs'] = 10
config['signal_type'] = _signal_type

net_opt = {}
net_opt['num_classes'] = 4
config["net"] = net_opt

optim_opt = {}
optim_opt["name"] = "SGD"
optim_opt["lr"] = 0.1
optim_opt["momentum"] = 0.9
optim_opt["weight_decay"] = 5e-4
optim_opt["nesterov"] = True
optim_opt["lr_protocol"] = [(5, 0.01),(7, 0.001),(9, 0.0001),(10, 0.00001)]
config["optim"] = optim_opt
config["display_step"] = 1000







