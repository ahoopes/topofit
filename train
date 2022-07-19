#!/usr/bin/env python

"""
Script to train a TopoFit model with custom data. If this code is
useful to you, please cite:

TopoFit: Rapid Reconstruction of Topologically-Correct Cortical Surfaces
Andrew Hoopes, Juan Eugenio Iglesias, Bruce Fischl, Douglas Greve, Adrian Dalca
Medical Imaging with Deep Learning. 2022.
"""

import os
import time
import argparse
import numpy as np
import torch
import topofit


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outdir', required=True, help='directory for model and logging output')
parser.add_argument('-t', '--training-subjs', required=True, help='text file with complete paths to preprocessed training subjects')
parser.add_argument('-v', '--validation-subjs', required=True, help='text file with complete paths to preprocessed validation subjects')
parser.add_argument('--hemi', required=True, help='hemisphere to train with (`lr` or `rh`)')
parser.add_argument('--reg-weight', type=float, default=0.5, help='mesh regularization weight')
parser.add_argument('--load-epoch', type=int, help='epoch number of model checkpoint to load from outdir')
parser.add_argument('--gpu', default='0', help='GPU device ID')
parser.add_argument('--skip-low-res', action='store_true', help='skip the initial low-resolution training')
parser.add_argument('--vol', help='Input volume (norm.mgz)',default='norm.mgz')
parser.add_argument('--xhemi', action='store_true', help='Xhemi')
args = parser.parse_args()

# sanity check on inputs
if args.hemi not in ('lh', 'rh'):
    print("error: hemi must be 'lh' or 'rh'")
    exit(1)

# necessary for speed gains
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

print(f'Input volume is {args.vol}');
print(f'Xhemi {args.xhemi}');

# configure GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda')
topofit.utils.set_device(device)

# get subjects
training_subjs = topofit.utils.read_file_list(args.training_subjs)
validation_subjs = topofit.utils.read_file_list(args.validation_subjs)

# configure output paths
os.makedirs(args.outdir, exist_ok=True)
validation_history_file = os.path.join(args.outdir, 'history.csv')
epoch_checkpoint_name = os.path.join(args.outdir, '{epoch:04d}.pt')

# configure model
print('Configuring model')
model = topofit.model.SurfNet().to(device)

# optimizer
print('Configuring optimizer')
init_learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

if args.load_epoch is None:
    # initialize model weights
    initial_epoch = 0
    model.initialize_weights()
else:
    # load checkpoint
    load_checkpoint = epoch_checkpoint_name.format(epoch=args.load_epoch)
    print(f'Loading checkpoint from {load_checkpoint}')
    checkpoint = torch.load(load_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state = checkpoint.get('optimizer_state_dict')
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    initial_epoch = args.load_epoch

# enable training mode
model.train()

# training with a ico-order 7 output mesh is 2x slower than a 6-order
# mesh, so let's start off in a 'low-res training mode' to speed things up
model.low_res_training = not args.skip_low_res

# set learning rate
print(f'Setting learning rate to {init_learning_rate:.2e}')
for group in optimizer.param_groups:
    group['lr'] = init_learning_rate

# training settings
epochs = 4000
steps_per_epoch = 100
checkpoint_save_epochs = 25
validation_epochs = 25
lr_decay_factor = 0.5
lr_decay_patience = 100
min_lr = 1e-7
print(f'Starting training at epoch {initial_epoch} / {epochs}')

# init tracking and training parameters
# validation metric used to compute learning rate decay
best_decay_metric = 1e10
best_decay_last_epoch = initial_epoch

# configure dataset sampler
data_iterator = iter(topofit.io.get_data_loader(args.hemi, training_subjs, model.low_res_training, 8,args.vol, args.xhemi))

# start training loop
for epoch in range(initial_epoch, epochs):

    # keep track of stuff for each epoch
    epoch_log = {}
    epoch_losses = {}
    epoch_step_time = []

    # utility function for caching losses
    def cache_loss(name, loss):
        if epoch_losses.get(name) is None:
            epoch_losses[name] = []
        epoch_losses[name].append(loss.item())

    for step in range(steps_per_epoch):

        # time step
        step_start_time = time.perf_counter()

        # reset optimizer
        optimizer.zero_grad(set_to_none=True)

        # sample and move training data to the GPU
        sample = next(data_iterator)
        for key, value in sample.data.items():
            sample.data[key] = value.to(device)

        # predict surface
        result, topology = model(sample.data['input_image'], sample.data['input_vertices'])
        
        # get true and predicted surfaces
        pred_white = result['pred_vertices']
        true_white = sample.data['true_vertices']

        # compute mesh similarity loss
        distance_loss = model.guided_chamfer_loss(true_white, pred_white)
        cache_loss('dist', distance_loss)
        loss = distance_loss

        # mesh regularization loss
        if args.reg_weight != 0:
            reg_loss = model.hinge_spring_loss(pred_white, topology)
            cache_loss('reg', reg_loss)
            loss = loss + reg_loss * args.reg_weight

        # total loss
        cache_loss('total', loss)

        # backpropagate and optimize
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.perf_counter() - step_start_time)

    # gather some loss info
    epoch_info = [
        'Epoch %d/%d' % (epoch + 1, epochs),
        '%.2f min' % (np.sum(epoch_step_time) / 60),
        '%.2f sec/step' % np.mean(epoch_step_time),
    ]
    epoch_info.extend(['loss-{n}: {v:.4f}'.format(n=n, v=np.mean(v)) for n, v in epoch_losses.items()])

    # get learning rate
    learning_rate = optimizer.param_groups[0]['lr']

    # run validation step
    if epoch % validation_epochs == 0 and epoch != initial_epoch:

        # validate
        model.train(mode=False)
        with torch.no_grad():
            for subj in validation_subjs:
                data = topofit.io.load_subject_data(subj, args.hemi, ground_truth=True, low_res=model.low_res_training, vol=args.vol, xhemi=args.xhemi)
                input_image = data['input_image'].to(device)
                input_vertices = data['input_vertices'].to(device)
                true_white = data['true_vertices'].to(device)
                result, topology = model(input_image, input_vertices)
                pred_white = result['pred_vertices']
                validation_dist = model.guided_chamfer_loss(true_white, pred_white).cpu().numpy()
        model.train(mode=True)

        # log validation
        metrics = {
            'dist': validation_dist,
            'lr': learning_rate,
            'loss': np.mean(epoch_losses['total']),
        }

        # write header
        if not os.path.isfile(validation_history_file):
            with open(validation_history_file, 'w') as file:
                file.write(', '.join(['epoch'] + [k for k in metrics.keys()]) + '\n') 

        # write validation metrics
        with open(validation_history_file, 'a') as file:
            file.write(', '.join([str(epoch)] + [str(v) for v in metrics.values()]) + '\n')

        # check if the validation results have plateaued
        if (validation_dist + 1e-3) < best_decay_metric:
            # if not, reset decay patience and best scores
            best_decay_last_epoch = epoch
            best_decay_metric = validation_dist
        elif epoch - best_decay_last_epoch >= lr_decay_patience:
            # if yes, cut learning rate and reset
            best_decay_last_epoch = epoch
            learning_rate *= lr_decay_factor
            epoch_info.append('Updating learning rate to %s' % str(learning_rate))
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

    # save standard epoch checkpoint
    if epoch % checkpoint_save_epochs == 0 and epoch != initial_epoch:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, epoch_checkpoint_name.format(epoch=epoch))

    # print epoch info
    print(' - '.join(epoch_info), flush=True)

    # stopping criteria
    if model.low_res_training and learning_rate < (init_learning_rate / 3):
        print('\nLow-res stop-criteria hit, switching to high-res')
        print(f'model. Resetting the learning rate to {init_learning_rate}.\n')
        for group in optimizer.param_groups:
            group['lr'] = init_learning_rate
        model.low_res_training = False
        data_iterator = iter(topofit.io.get_data_loader(args.hemi, training_subjs, model.low_res_training,args.xhemi))
    elif learning_rate < min_lr:
        print('Surpassed minimum learning rate - stopping training')
        break
