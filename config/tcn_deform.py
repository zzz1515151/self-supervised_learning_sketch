_sketch_rnn_root = "./sketch_stroke_data"
_signal_type = "deformation"
train_batch_size = 512
test_batch_size = 256

config = {}

data_train_opt = {}
data_train_opt["root"] = _sketch_rnn_root
data_train_opt["batch_size"] = train_batch_size
data_train_opt["split"] = "train"

data_test_opt = {}
data_test_opt["root"] = _sketch_rnn_root
data_test_opt["batch_size"] = test_batch_size
data_test_opt["split"] = "test"

config["data_train_opt"] = data_train_opt
config["data_test_opt"] = data_test_opt
config["signal_type"] = _signal_type

net_opt = {}
net_opt["name"] = "TCN"
net_opt["num_classes"] = 2
net_opt["num_filters"] = 32
net_opt["window_sizes"] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
net_opt["is_feature_extractor"] = False

config["net_opt"] = net_opt

optim_opt = {}
optim_opt["name"] = "Adam"
optim_opt["lr"] = 1e-2
optim_opt["lr_protocol"] = [(5, 0.001),(10, 0.0001)]
config["optim_opt"] = optim_opt


config["num_epochs"] = 10
config["display_step"] = 1000

