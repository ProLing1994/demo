import argparse
import sys 
import time
from tqdm import tqdm

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/KWS')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo')
sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.logging_helpers import setup_logger
from common.common.utils.python.plotly_tools import plot_loss, plot_loss2d

def test(cfg, net, loss_func, epoch_idx, batch_idx, logger, test_data_loader, mode='eval'):
  """
  :param cfg:                config contain data set information
  :param net:                net
  :param epoch_idx:          epoch index
  :param batch_idx:          batch index
  :param logger:             log for save testing result
  :param test_data_loader:   testing data loader
  :param mode:               evaluate either training set or testing set
  :return:                   None
  """
  net.eval()

  scores = []
  labels = []
  losses = []
  for _, (x, label, index) in tqdm(enumerate(test_data_loader)):
    x, label = x.cuda(), label.cuda()
    score = net(x)
    loss = loss_func(score, label)

    scores.append(torch.max(score, 1)[1].cpu().data.numpy())
    labels.append(label.cpu().data.numpy())
    losses.append(loss.item())

  # caltulate accuracy
  accuracy = float((np.array(scores) == np.array(labels)).astype(int).sum()) / float(len(labels))
  loss = np.array(losses).sum()/float(len(labels))

  msg = 'epoch: {}, batch: {}, {}_accuracy: {:.4f}, {}_loss: {:.4f}'.format(epoch_idx, batch_idx, mode, accuracy, mode, loss)
  logger.info(msg)

def train(config_file):
  """ training engine
  :param config_file:   the input configuration file
  :return:              None
  """
  # record time
  begin_t = time.time() 

  # load configuration file
  cfg = load_cfg_file(config_file)

  # clean the existing folder if the user want to train from scratch
  setup_workshop(cfg)

  # control randomness during training
  init_torch_and_numpy(cfg)

  # enable logging
  log_file = os.path.join(cfg.general.save_dir, 'logging', 'train_log.txt')
  logger = setup_logger(log_file, 'kws_train')

  # define network
  net = import_network(cfg)

  # define loss function
  loss_func = define_loss_function(cfg)

  # set training optimizer, learning rate scheduler
  optimizer = set_optimizer(cfg, net)

  # load checkpoint if finetune_on == True or resume epoch > 0
  if cfg.general.finetune_on == True:
    # fintune, Load model, reset learning rate
    last_save_epoch, start_batch = load_checkpoint(cfg.general.finetune_epoch, net,  
                                                    cfg.general.finetune_model_dir)
    start_epoch, last_save_epoch, start_batch = 0, 0, 0
  elif cfg.general.resume_epoch >= 0:
    # resume, Load the model, continue the previous learning rate
    last_save_epoch, start_batch = load_checkpoint_resume(cfg.general.resume_epoch, net, optimizer, 
                                                          cfg.general.save_dir)
    start_epoch = last_save_epoch
  else:
    start_epoch, last_save_epoch, start_batch = 0, 0, 0

  # get training data set and test data set
  train_dataloader, len_dataset = generate_dataset(cfg, 'training')
  if cfg.general.is_test:
    eval_validation_dataloader = generate_test_dataset(cfg, 'validation')
    # eval_train_dataloader = generate_test_dataset(cfg, 'training')

  msg = 'Training dataset number: {}'.format(len_dataset)
  logger.info(msg)

  msg = 'Init Time: {}'.format((time.time() - begin_t) * 1.0)
  logger.info(msg)

  batch_number = len(train_dataloader)
  data_iter = iter(train_dataloader)
  batch_idx = start_batch

  # loop over batches
  for i in range(batch_number):

    net.train()
    begin_t = time.time() 
    optimizer.zero_grad()

    epoch_idx = start_epoch + i * cfg.train.batch_size // len_dataset
    batch_idx += 1

    inputs, labels, indexs = data_iter.next()

    # save training images for visualization
    if cfg.debug.save_inputs:
        save_intermediate_results(cfg, "training", epoch_idx, inputs, labels, indexs)

    inputs, labels = inputs.cuda(), labels.cuda()
    scores = net(inputs)
    loss = loss_func(scores, labels)
    loss.backward()
    optimizer.step()

    # caltulate accuracy
    pred_y = torch.max(scores, 1)[1].cpu().data.numpy()
    accuracy = float((pred_y == labels.cpu().data.numpy()).astype(int).sum()) / float(labels.size(0))

    # print training information
    sample_duration = (time.time() - begin_t) * 1.0 / cfg.train.batch_size
    msg = 'epoch: {}, batch: {}, train_accuracy: {:.4f}, train_loss: {:.4f}, time: {:.4f} s/vol' \
        .format(epoch_idx, batch_idx, accuracy, loss.item(), sample_duration)
    logger.info(msg)

    if (batch_idx % cfg.train.plot_snapshot) == 0:
      if cfg.general.is_test:
        train_loss_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
        plot_loss2d(log_file, train_loss_file, name=['train_loss', 'eval_loss'],
                  display='Training/Validation Loss ({})'.format(cfg.loss.name)) 
        train_accuracy_file = os.path.join(cfg.general.save_dir, 'train_accuracy.html')
        plot_loss2d(log_file, train_accuracy_file, name=['train_accuracy', 'eval_accuracy'],
                  display='Training/Validation Accuracy ({})'.format(cfg.loss.name)) 
      else:
        train_loss_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
        plot_loss(log_file, train_loss_file, name='train_loss',
                  display='Training Loss ({})'.format(cfg.loss.name))
        train_accuracy_file = os.path.join(cfg.general.save_dir, 'train_accuracy.html')
        plot_loss(log_file, train_accuracy_file, name='train_accuracy',
                  display='Training Accuracy ({})'.format(cfg.loss.name))

    if epoch_idx % cfg.train.save_epochs == 0 or epoch_idx == cfg.train.num_epochs - 1:
      if last_save_epoch != epoch_idx:
        last_save_epoch = epoch_idx

        # save training model
        save_checkpoint(net, optimizer, epoch_idx, batch_idx, cfg, config_file)

        if cfg.general.is_test:
          test(cfg, net, loss_func, epoch_idx, batch_idx, logger, eval_validation_dataloader, mode='eval')
          # test(cfg, net, loss_func, epoch_idx, batch_idx, logger, eval_train_dataloader, mode='eval')

def main():
  parser = argparse.ArgumentParser(description='Streamax KWS Training Engine')
  # parser.add_argument('-i', '--input', type=str, default="/home/engineers/yh_rmai/code/demo/Speech/KWS/config/kws/kws_config.py", nargs='?', help='config file')
  # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config.py", nargs='?', help='config file')
  # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", nargs='?', help='config file')
  # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu_2.py", nargs='?', help='config file')
  parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py", nargs='?', help='config file')
  # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu_2_label.py", nargs='?', help='config file')
  args = parser.parse_args()
  train(args.input)

if __name__ == "__main__":
  main()