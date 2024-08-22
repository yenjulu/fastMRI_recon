import argparse
import yaml, os, time
import h5py
import torchvision.transforms as transforms
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from get_instances import *
from utils import *

def setup(args):
    config_path = args.config
    with open(config_path, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)
    device = 'cuda'

    #read configs =================================
    n_layers = configs['n_layers']
    k_iters = configs['k_iters']
    dataset_name = configs['dataset_name']
    dataset_params = configs['dataset_params']
    batch_size = configs['batch_size'] if args.batch_size is None else args.batch_size
    model_name = configs['model_name']
    model_params = configs.get('model_params', {})
    model_params['n_layers'] = n_layers
    model_params['k_iters'] = k_iters
    score_names = configs['score_names']
    config_name = configs['config_name']

    workspace = os.path.join(args.workspace, config_name) #workspace/config_name
    checkpoints_dir, log_dir = get_dirs(workspace) #workspace/config_name/checkpoints ; workspace/config_name/log.txt
    tensorboard_dir = os.path.join(args.tensorboard_dir, configs['config_name']) #runs/config_name
    logger = Logger(log_dir)
    writer = get_writers(tensorboard_dir, ['test'])['test']
    dataloader = get_loaders(dataset_name, dataset_params, batch_size, ['test'])['test']
    model = get_model(model_name, model_params, device)
    score_fs = get_score_fs(score_names)

    #restore
    saver = CheckpointSaver(checkpoints_dir)
    prefix = 'best' if configs['val_data'] else 'final'
    checkpoint_path = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.startswith(prefix)][0]
    model = saver.load_model(checkpoint_path, model)

    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

    return configs, device, workspace, logger, writer, dataloader, model, score_fs, tensorboard_dir

def main(args):
    configs, device, workspace, logger, writer, dataloader, model, score_fs, tensorboard_dir = setup(args)
    
    logger.write('\n')
    logger.write('test start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    start = time.time()  
    if args.seed:
        set_seeds(args.seed)
        
    running_score = defaultdict(int)

    model.eval()
    
    for i, (y, ref_kspace_tensor, nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor) in enumerate(tqdm(dataloader)):       
        y, ref_kspace_tensor, x, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor = y.to(device), ref_kspace_tensor.to(device), nw_input_tensor.to(device), sens_maps_tensor.to(device), trn_mask_tensor.to(device), loss_mask_tensor.to(device)
        
        with torch.no_grad():
            y_pred, nw_output_kspace, y_resnet = model(x, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor)


        center_crop = transforms.CenterCrop((384, 384))
        x = center_crop(x)
        y = center_crop(y)
        y_pred = center_crop(y_pred)
        y_resnet = center_crop(y_resnet)

        y = np.abs(y.detach().cpu().numpy())
        y_pred = np.abs(y_pred.detach().cpu().numpy())
        y_resnet = np.abs(y_resnet.detach().cpu().numpy())

        # y = img_normalize(y)
        # y_pred = img_normalize(y_pred)
        # y_resnet = img_normalize(y_resnet)

        f = h5py.File(tensorboard_dir + '/test/recon_' + str(i).zfill(3) + '.h5', 'w')
        f.create_dataset('recon', data=y_pred)
        f.create_dataset('gt', data=y)
        f.close()

        for score_name, score_f in score_fs.items():
            running_score[score_name] += score_f(y, y_pred) * y_pred.shape[0]
                
        if args.write_image > 0 and (i % args.write_image == 0):
            writer.add_figure('img', display_img(np.abs(r2c(x[-1].detach().cpu().numpy())), trn_mask_tensor[-1].detach().cpu().numpy(), loss_mask_tensor[-1].detach().cpu().numpy(), \
                y[-1], y_pred[-1], y_resnet[-1], psnr(y[-1], y_pred[-1])), i)
        
    epoch_score = {score_name: score / len(dataloader.dataset) for score_name, score in running_score.items()}
    for score_name, score in epoch_score.items():
        writer.add_scalar(score_name, score, 0)
        logger.write('test {} score: {:.4f}'.format(score_name, score))

    writer.close()
    logger.write('-----------------------')
    logger.write('total test time: {:.2f} min'.format((time.time()-start)/60))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=False, default="configs/base_modl,k=1.yaml",
                        help="config file path")
    parser.add_argument("--workspace", type=str, default='./workspace')
    parser.add_argument("--final_report", type=str, default='./final_report')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--write_image", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    main(args)