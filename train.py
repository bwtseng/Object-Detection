import utility as utl 
import torchnet.meter as tnt 
import performance_tracker as pt 
import checkpoint as ckpt 
import summary
import sensitivity_analysis as sa 
import collector
import logging
import config
import math 
import time 
import random
from collections import OrderedDict
from torch.autograd import Variable
from apex import amp
from functools import partial
import logging 
import torch 
import numpy as np 


msglogger = logging.getLogger()
# Currently, YOLOv1 only follow the regression loss, the class meter thus remove at this stage:
def _log_training_progress(args, losses, epoch, steps_completed, steps_per_epoch, batch_time, optimizer, loggers):
    # Log some statistics
    errs = OrderedDict()
    
    #if not early_exit_mode(args):
    """
    Early exist model may be incorporated in the future.
    else:
        # For Early Exit case, the Top1 and Top5 stats are computed for each exit.
        for exitnum in range(args.num_exits):
            errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
            errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)
    """
    stats_dict = OrderedDict()
    for loss_name, meter in losses.items():
        stats_dict[loss_name] = meter.mean

    stats_dict.update(errs)
    stats_dict['LR'] = optimizer.param_groups[0]['lr']
    stats_dict['Time'] = batch_time.mean
    stats = ('Performance/Training/', stats_dict)

    params = model.named_parameters() if args.log_params_histograms else None
    utl.log_training_progress(stats, params, epoch, steps_completed,
                              steps_per_epoch, args.print_freq, loggers)

def light_train_with_distiller(args, model, optimizer, compress_scheduler, device, dataloaders, dataset_sizes, loggers, epoch=1):

    """
    Training-with-compression loop for one epoch. 
    IMPORTANT INFORMATION:

    For each training step in epoch:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)
    """    
    total_samples = dataset_sizes['train']
    batch_size = dataloaders["train"].batch_size

    steps_per_epoch = math.ceil(total_samples / batch_size)    
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    OVERALL_LOSS_KEY = 'Overall Loss'
    #OBJECTIVE_LOSS_KEY = 'Objective Loss' # Only compute loss from the cost function, but we revise its name in Natural loss.
    NAT_LOSS_KEY = "Natural Loss"
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                         (NAT_LOSS_KEY, tnt.AverageValueMeter())])    

    model.train()
    end = time.time()
    for train_step, data in enumerate(dataloaders["train"], 0):
        inputs = Variable(data[0]).float().to(device)
        labels = Variable(data[1]).float().to(device)

        if compress_scheduler:
            compress_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)
        
        nat_output = model(inputs, labels) # Tuple (loss, pred_tensor)
        losses[NAT_LOSS_KEY].add(nat_output[0].item())

        """
        ****
        Drop the early exist mode in this first version
        ****
        if not early_exit_mode(args):
        loss = criterion(output, target)
        # Measure accuracy
        classerr.add(output.detach(), target)
        acc_stats.append([classerr.value(1), classerr.value(5)])
        else:
        # Measure accuracy and record loss
        classerr.add(output[args.num_exits-1].detach(), target) # add the last exit (original exit)
        loss = earlyexit_loss(output, target, criterion, args)
        """

        if compress_scheduler: 
            # Should be revised if using adversarial robustness training.    
            agg_loss =  compress_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, nat_output[0],
                                                                optimizer=optimizer, return_loss_components=True)            
            # Should be modified, this may be incorporated in the future.
            loss = agg_loss.overall_loss
            # if admm loss is zero, following line may raise error.
            losses[OVERALL_LOSS_KEY].add(loss.item())
            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                try:
                    losses[lc.name].add(lc.value.item())
                except:
                    # This is a constant case which may be raised by our ADMM implementation.
                    losses[lc.name].add(lc.value)
        else: 
            losses[OVERALL_LOSS_KEY].add(nat_loss.item())
            loss = nat_loss

        optimizer.zero_grad()

        # ******************
        # Try apex AMP package:
        # ******************
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        if compress_scheduler:
            # Applied zero gradient here.
            compress_scheduler.before_parameter_optimization(epoch, train_step, steps_per_epoch, optimizer)
        
        optimizer.step()
        if compress_scheduler:
            # Sometime this on "minibatch_end" function will not do anything.
            compress_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)
        
        batch_time.add(time.time() - end)
        steps_completed = (train_step + 1)

        if steps_completed % args.print_freq == 0 :
            # arguments for this function: losses, epoch, steps_completed, steps_per_epoch, batch_time, optimizer, loggers
            _log_training_progress(args, losses, epoch, steps_completed, steps_per_epoch, batch_time, optimizer, loggers)
        end = time.time()

    utl.log_weights_sparsity(model, epoch, loggers)
    return losses[NAT_LOSS_KEY], losses[OVERALL_LOSS_KEY]

def _log_valiation_progress(args, losses, epoch, steps_completed, steps_per_epoch, loggers):
    #if not _is_earlyexit(args):
    """
    Early exist model use following code:
    else:
        stats_dict = OrderedDict()
        for exitnum in range(args.num_exits):
            la_string = 'LossAvg' + str(exitnum)
            stats_dict[la_string] = args.losses_exits[exitnum].mean
            # Because of the nature of ClassErrorMeter, if an exit is never taken during the batch,
            # then accessing the value(k) will cause a divide by zero. So we'll build the OrderedDict
            # accordingly and we will not print for an exit error when that exit is never taken.
            if args.exit_taken[exitnum]:
                t1 = 'Top1_exit' + str(exitnum)
                t5 = 'Top5_exit' + str(exitnum)
                stats_dict[t1] = args.exiterrors[exitnum].value(1)
                stats_dict[t5] = args.exiterrors[exitnum].value(5)
    """
    
    stats_dict = OrderedDict(('Nat_Loss', losses['natural_loss'].mean))
    stats = ('Performance/Validation/', stats_dict)
    utl.log_training_progress(stats, None, epoch, steps_completed,
                                   steps_per_epoch, args.print_freq, loggers)

def _validate(args, dataloaders, data_group, model, device, loggers, epoch=-1, noise_factor=0):
    if epoch != -1:
        msglogger.info("----Validate (epoch=%d)----", epoch)

    losses = {'natural_loss': tnt.AverageValueMeter()}
    """
    Drop early exist model so far.
    if _is_earlyexit(args):
        # for Early Exit, we have a list of errors and losses for each of the exits.
        args.exiterrors = []
        args.losses_exits = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exit_taken = [0] * args.num_exits
    """
    batch_time = tnt.AverageValueMeter()
    total_samples = len(dataloaders[data_group].sampler)
    batch_size = dataloaders[data_group].batch_size
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # For robustness testing, 20% of the testing data will be steralized.
    add_noise_steps =  int(total_steps * 0.2)
    end_range = int(total_steps) - add_noise_steps
    start_step = random.randint(0, end_range)
    start_step = 60
    end_step = start_step + add_noise_steps
    # Turn into evaluation mode.
    model.eval()
    end = time.time()
    # Starting primiary testing code here.
    with torch.no_grad():
        for validation_step, data in enumerate(dataloaders[data_group]):
            inputs = Variable(data[0]).float().to(device)
            labels = Variable(data[1]).float().to(device)

            if validation_step >= start_step and validation_step <= end_step and noise_factor:
                msglogger.info('%f is the factor multiplied in epoch %d', noise_factor, epoch)
                #print(validation_step)
                shape = inputs.shape
                # If using imagenet dataset:
                mean = np.array([0.0, 0.0, 0.0])
                std = np.array([1.0, 1.0, 1.0])
                min_val = (np.array([0,0,0]) - mean) / std
                max_val = (np.array([1,1,1]) - mean) / std
                noise = noise_factor * torch.randn(shape).to(device)
                
                """
                # for Plotting
                count = 0
                for fac in [0, 0.5, 1, 1.5, 2, 2.5, 3]:
                    temp_noise = fac * torch.randn(shape).to(device)
                    temp_inputs = inputs + temp_noise
                    for i in range(3):
                        temp_inputs[:, i, :, :] = torch.clamp(temp_inputs[:, i, :, :], min_val[i], max_val[i])    
                    #temp_inputs = temp_inputs.numpy()
                    temp_inputs = temp_inputs[9, :, :, :].cpu().detach().numpy()
                    
                    for i in range(3):
                        temp_inputs[i, :, :] = (temp_inputs[i, :, :] * std[i]) + mean[i]
                    temp_inputs = np.transpose(temp_inputs, (1, 2, 0))
                    def plot(img):
                        #x = x - np.min(x)
                        #x /= np.max(x)
                        img *= 255  
                        img = img.astype(np.uint8)
                        img = img.reshape(224, 224, 3)
                        return img
                    plt.imsave(os.path.join(msglogger.logdir, str(count)+'.png'), plot(temp_inputs))
                    count+=1
                assert 1 == 2  
                """

                inputs += noise
                for i in range(3):
                    inputs[:, i, :, :] = torch.clamp(inputs[:, i, :, :], min_val[i], max_val[i])
                                
            nat_output = model(inputs, labels) # Tuple (loss, pred_tensor)

            # Early exist mode will incorporate in the near future.
            '''
            if not _is_earlyexit(args):
                # compute loss
                loss = criterion(output, target)
                # measure accuracy and record loss
                losses['objective_loss'].add(loss.item())
                classerr.add(output.detach(), target)
                if args.display_confusion:
                    confusion.add(output.detach(), target)
            else:
                earlyexit_validate_loss(output, target, criterion, args)
            '''

            losses['natural_loss'].add(nat_output[0].item())
            steps_completed = (validation_step + 1)
            batch_time.add(time.time() - end)
            end = time.time()  
            steps_completed = (validation_step + 1)
            # "\033[0;37;40m\tExample\033[0m"
            if steps_completed % (args.print_freq) == 0 :   
                _log_valiation_progress(args, losses, epoch, steps_completed, total_steps, [loggers[1]])

    
    stats_tmp = OrderedDict()
    stats_tmp['Nat_Loss'] = losses['natural_loss'].mean
    stats = ('Performance/Validation/', stats_tmp)
    
    utl.log_training_progress(stats, None, epoch, steps_completed=0,
                                    total_steps=1, log_freq=1, loggers=[loggers[0]])

    msglogger.info('==>  Nat_Loss: {:.5f} \n.'.format(losses['natural_loss'].mean))
    return losses['natural_loss'].mean


def trian_validate_with_scheduling(args, net, optimizer, compress_scheduler, device, dataloaders, dataset_sizes, 
                                   loggers, tracker, epoch=1, validate=True, verbose=True):
    # Whtat's collectors_context
    # At first, we need to specify the model name, and its learning progress:
    #if not os.path.isdir(args.output_dir):
    #    os.mkdir(args.output_dir)
    #os.mkdir(args.output_dir, exist_ok=True)
    
    name = args.name
    if args.name == '':
        name = args.arch + "_" + args.dataset
    # Must exist pruning mode.
    # Reset learning rate and momentum buffer in the optimizer for next learning stage! 
    # Should know whether the learning rate decay is based on epochs or steps 
    # Or more, the meaning of last epoch argument indicates current epoch.
    #****
    # This line may raise the problems that the epoch doesn't exist any policy....
    #****
    if compress_scheduler: 
        if compress_scheduler.prune_mechanism:
            if epoch == (compress_scheduler.pruner_info['max_epoch']):
                # Reset optimizer and learning rate in retrain phase.
                # NOTE: We should specify the true 
                for index in range(len(compress_scheduler.policies[epoch])):
                    policy_name = compress_scheduler.policies[epoch][index].__class__.__name__.split("Policy")[0]
                    if policy_name == "LR":
                        compress_scheduler.policies[epoch][index].lr_scheduler.optimizer.param_groups[0]['lr'] = args.lr_retrain
                        compress_scheduler.policies[epoch][index].lr_scheduler.base_lrs = [args.lr_retrain]
                        compress_scheduler.policies[epoch][index].lr_scheduler.optimizer.param_groups[0]['momentum'] = 0.9
                        compress_scheduler.policies[epoch][index].lr_scheduler.optimizer.param_groups[0]['initial_lr'] = args.lr_retrain
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if 'momentum_buffer' in optimizer.state[p]:
                                    optimizer.state[p].pop('momentum_buffer', None)
                        break 
        
            if epoch == (compress_scheduler.pruner_info['min_epoch']):
                # *****
                # NOTE If not using ADMM pruner, do we need to reset lr scheduler in this loop?
                # *****
                # Reset learning rate and momentum buffer for pruning stage! 
                policy_name = compress_scheduler.policies[epoch][0].__class__.__name__.split("Policy")[0]            
                #if policy_name != "ADMM":
                #    compress_scheduler.policies[epoch][0].lr_scheduler.optimizer.param_groups[0]['lr'] = args.lr_prune
                #    compress_scheduler.policies[epoch][0].lr_scheduler.base_lrs = [args.lr_prune]
                #    compress_scheduler.policies[epoch][0].lr_scheduler.optimizer.param_groups[0]['momentum'] = 0.9
                #    compress_scheduler.policies[epoch][0].lr_scheduler.optimizer.param_groups[0]['initial_lr'] = args.lr_prune
                for group in optimizer.param_groups:
                    group['lr'] = args.lr_prune    
                    group['initial_lr'] = args.lr_prune
                    # for group in optimizer.param_groups:
                    for p in group['params']:
                        if 'momentum_buffer' in optimizer.state[p]:
                            optimizer.state[p].pop('momentum_buffer', None)
        
            if epoch >= compress_scheduler.pruner_info['max_epoch']:
                name  += "_retrain"
            
            elif epoch < compress_scheduler.pruner_info['min_epoch']:
                name += "_pretrain"

            else:
                name += "_prune" 
    else: 
        # Only proceed with pre-train or re-train phase model. 
        name = name + "_" + args.stage
    
    if compress_scheduler:
        #dataset_name = 'val' if args.split_ratio != 0 else 'test'
        dataset_name = 'test'
        #data_group, model, criterion, device, num_classes, loggers, epoch=-1, noise_factor=0)
        
        # These two partial function is created for the channel pruning methods (Prof. Han):
        forward_fn = partial(_validate, args=args, dataloaders=dataloaders, data_group=dataset_name, device=device, loggers=loggers,   epoch=-1, noise_factor=0)

        #light_train_with_distiller(model, criterion, optimizer, compress_scheduler, device, num_classes, 
        #                       dataset_sizes, loggers, epoch=1)

        train_fn = partial(light_train_with_distiller, args=args, optimizer=optimizer, compress_scheduler=compress_scheduler, 
                               device=device, dataloaders =dataloaders, dataset_sizes=dataset_sizes, loggers=loggers, epoch=0)
        
        compress_scheduler.on_epoch_begin(epoch, optimizer, forward_fn=forward_fn, train_fn=train_fn)


    
    nat_loss, overall_loss = light_train_with_distiller(args, net, optimizer, compress_scheduler, 
                                                          device, dataloaders, dataset_sizes, loggers, epoch)  
                                                          
    if validate: 
        nat_loss = _validate(args, dataloaders, 'test', net, device, loggers, epoch=epoch)  
        
    if compress_scheduler:
        # Or we can compute IoU here?
        loss = nat_loss
        #top1 = nat_loss
        compress_scheduler.on_epoch_end(epoch, optimizer, metrics={'min':loss}) #, 'max':top1})
    
    is_best, checkpoint_extras = _finalize_epoch(args, net, tracker, epoch, top1=nat_loss)             
    # Check whether the out direcotry is already built.
    ckpt.save_checkpoint(epoch, args.arch, net, optimizer=optimizer,
                         scheduler=compress_scheduler, extras=checkpoint_extras,
                         is_best=is_best, name=name, dir=msglogger.logdir)

    #return top1, top5, loss, tracker
    return nat_loss, tracker


def _log_best_scores(args, performance_tracker, logger, how_many=-1):
    """
    Utility to log the best scores (minimum loss value).
    This function is currently written for pruning use-cases, but can be generalized.
    """
    assert isinstance(performance_tracker, (pt.SparsityAccuracyTracker))
    if how_many < 1:
        how_many = performance_tracker.max_len

    how_many = min(how_many, performance_tracker.max_len)
    best_scores = performance_tracker.best_scores(how_many)
    for score in best_scores:
        logger.info('==> Best [Nat_Loss: %.3f Sparsity:%.2f  NNZ-Params: %d on epoch: %d]',
                    -1*score.top1, score.sparsity, -score.params_nnz_cnt, score.epoch)


def _finalize_epoch(args, net, tracker, epoch, **kwargs):
    # ****
    # We mulitply -1 to the loss value so that the performance tracker can work well without any modifications.
    # ****
    loss_multiply = -1 * kwargs['top1']
    tracker.step(net, epoch, top1=loss_multiply, adv_train=False)    
    _log_best_scores(args, tracker, msglogger)
    best_score = tracker.best_scores()[0]
    is_best = epoch == best_score.epoch
    checkpoint_extras = {'current_top1': kwargs['top1'],
                         'best_top1': -1 * best_score.top1,
                         'best_epoch': best_score.epoch}
    return is_best, checkpoint_extras

