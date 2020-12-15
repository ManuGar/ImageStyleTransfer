import os
import shutil
from random import sample
from pathlib import Path
import sys
import subprocess

os.system('pip install git+https://github.com/tmabraham/UPIT.git')

def set_gpu(gpu_id):
    import torch
    torch.cuda.set_device(gpu_id)

def generate_images(algo, dataset_name, output_path):
    if algo == 'upit':
        from upit.models.cyclegan import CycleGAN
        from upit.data.unpaired import get_dls
        from upit.inference.cyclegan import cycle_learner
        from fastai.vision.all import partial
        from upit.train.cyclegan import cycle_learner, fit_flat_lin, combined_flat_anneal, ShowCycleGANImgsCallback, \
            CycleGANTrainer, CycleGANLoss
        import torch

        trainA_path = Path('datasets/'+dataset_name +'/trainA')
        trainB_path = Path('datasets/'+dataset_name +'/trainB')
        set_gpu(0)
        print(f"There are {len(trainA_path.ls())} photos to tranfer to the original style")
        print(f"There are {len(trainB_path.ls())} Original style")
        dls = get_dls(trainA_path, trainB_path,load_size=256,crop_size=256,bs=4)
        # cycle_gan = CycleGAN(3,3,64,gen_blocks=3)
        # learn = cycle_learner(dls, cycle_gan,opt_func=partial(Adam,mom=0.5,sqr_mom=0.999),show_img_interval=8)
        cycle_gan = CycleGAN(3, 3, 64, gen_blocks=3)

        cycle_gan.G_B.load_state_dict(torch.load('upitModel.pth'))
        learn = cycle_learner(dls, cycle_gan)
        # model = torch.load('models/upitModel.pth')
        learn.lr_find()
        learn.fit_flat_lin(7,7,2e-4)
        export_generator(learn, generator_name='upitModel')
        assert os.path.exists('upitModel.pth')
        #pred_path = '../GenerateDatasetCycleGAN/'
        pred_path = output_path
        cycle_gan.G_B.load_state_dict(torch.load('models/upitModel.pth'))
        learn = cycle_learner(dls, cycle_gan)
        get_preds_cyclegan(learn,'datasets/'+dataset_name +'/trainA',pred_path,suffix='jpg',bs=1)

    elif algo == 'nst':
        import neural_style_transfer
        image_style = sample(os.listdir('datasets/'+dataset_name+'/trainB'),k=1)
        neural_style_transfer.neural_style_transfer('datasets/'+dataset_name+'/trainA','datasets/'+dataset_name+'/trainB/'+image_style[0], output_path)

    elif algo=='strotss':
        os.system('git clone https://github.com/joheras/STROTSS')
        abs_path_strotss = os.path.abspath('STROTSS')
        sys.path.insert(0, abs_path_strotss)

        images = os.listdir('datasets/'+dataset_name+'/trainA')
        image_style = sample(os.listdir('datasets/'+dataset_name+'/trainB'),k=1)
        for image in images:
            os.system('cd STROTSS; python3 styleTransfer.py ' + image + ' datasets/'+dataset_name+'/trainB/'+image_style[0] + ' 1.0 5')
            shutil.move('STROTSS/output.png',output_path+'/'+image)

    elif algo == 'forkGAN':
        # set_gpu(0)
        os.system('git clone https://github.com/zhengziqiang/ForkGAN')
        shutil.copytree('datasets/' + dataset_name, 'ForkGAN/datasets/'+dataset_name)
        shutil.move('ForkGAN/datasets/' + dataset_name + '/trainA','ForkGAN/datasets/' + dataset_name + '/testA')
        shutil.move('ForkGAN/datasets/' + dataset_name + '/trainB','ForkGAN/datasets/' + dataset_name + '/testB')
        shutil.copytree('ForkGAN/datasets/' + dataset_name + '/testB', 'ForkGAN/datasets/' + dataset_name + '/trainB')
        shutil.copytree('ForkGAN/datasets/' + dataset_name + '/testA', 'ForkGAN/datasets/' + dataset_name + '/trainA')

        # os.system('cd ForkGAN')
        os.system('cd ForkGAN;python main.py --phase train --dataset_dir '+ dataset_name+ ' --epoch 20 --gpu 1 --n_d 2 --n_scale 2 --checkpoint_dir ./check/'+dataset_name+' --sample_dir ./check/'+dataset_name + '/sample --L1_lambda 10')
        os.system('cd ForkGAN;python main.py --phase test --dataset_dir '+ dataset_name + ' --gpu 1 --n_d 2 --n_scale 2 --checkpoint_dir ./check/'+dataset_name + ' --test_dir ./check/'+ dataset_name +'/testa2b --which_direction AtoB')
        shutil.move('ForkGAN/check/'+ dataset_name+ '/testa2b',output_path)
        shutil.rmtree('ForkGAN/datasets/'+dataset_name)
        # os.system('cd ..')

    elif algo == 'ganilla':
        set_gpu(0)
        os.system('git clone https://github.com/giddyyupp/ganilla.git')
        shutil.copytree('datasets/' + dataset_name, 'ganilla/datasets/' + dataset_name)
        shutil.move('ganilla/datasets/' + dataset_name + '/trainA', 'ganilla/datasets/' + dataset_name + '/testA')
        shutil.move('ganilla/datasets/' + dataset_name + '/trainB', 'ganilla/datasets/' + dataset_name + '/testB')
        shutil.copytree('ganilla/datasets/' + dataset_name + '/testB', 'ganilla/datasets/' + dataset_name + '/trainB')
        shutil.copytree('ganilla/datasets/' + dataset_name + '/testA', 'ganilla/datasets/' + dataset_name + '/trainA')
        # os.system('cd ganilla')
        # os.system('pip install ganilla/requirements.txt')
        os.system('cd ganilla;python train.py --dataroot ./datasets/'+ dataset_name + ' --name '+dataset_name+'_cyclegan --model cycle_gan --netG resnet_fpn')
        os.system('cd ganilla;python test.py --dataroot ./datasets/'+dataset_name+' --name '+ dataset_name+'_cyclegan --model cycle_gan --netG resnet_fpn')
        shutil.move('ganilla/results/'+ dataset_name + '_cyclegan/test_100/images',output_path)
        shutil.rmtree('ganilla/datasets/'+dataset_name)
        # os.system('cd ..')

    elif algo == 'dualGAN':
        # set_gpu(0)
        os.system('git clone https://github.com/duxingren14/DualGAN.git')
        # os.makedirs('DualGAN/datasets/' + dataset_name+ '/train',exist_ok=True)
        # os.makedirs('DualGAN/datasets/' + dataset_name+ '/test',exist_ok=True)
        # os.makedirs('DualGAN/datasets/' + dataset_name+ '/val',exist_ok=True)

        shutil.copytree('datasets/' + dataset_name+'/', 'DualGAN/datasets/' + dataset_name + '/train')
        shutil.move('DualGAN/datasets/' + dataset_name + '/train/trainA','DualGAN/datasets/' + dataset_name + '/train/A')
        shutil.move('DualGAN/datasets/' + dataset_name + '/train/trainB','DualGAN/datasets/' + dataset_name + '/train/B')
        shutil.copytree('datasets/' + dataset_name, 'DualGAN/datasets/' + dataset_name + '/test')
        shutil.move('DualGAN/datasets/' + dataset_name + '/test/trainA', 'DualGAN/datasets/' + dataset_name + '/test/A')
        shutil.move('DualGAN/datasets/' + dataset_name + '/test/trainB', 'DualGAN/datasets/' + dataset_name + '/test/B')
        shutil.copytree('datasets/' + dataset_name, 'DualGAN/datasets/' + dataset_name + '/val')
        shutil.move('DualGAN/datasets/' + dataset_name + '/val/trainA', 'DualGAN/datasets/' + dataset_name + '/val/A')
        shutil.move('DualGAN/datasets/' + dataset_name + '/val/trainB', 'DualGAN/datasets/' + dataset_name + '/val/B')
        os.system('cd DualGAN; python main.py --phase train --dataset_name '+ dataset_name + ' --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100')
        os.system('cd DualGAN; python main.py --phase test --dataset_name '+ dataset_name + ' --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100')
        shutil.move('DualGAN/sample/'+ dataset_name+ '-img_sz_256-fltr_dim_64-L1-lambda_AB_1000.0_1000.0',output_path)
        shutil.rmtree('DualGAN/datasets/'+dataset_name)
        # os.system('cd ..')

    elif algo == 'CUT':
        set_gpu(0)
        os.system('git clone https://github.com/taesungp/contrastive-unpaired-translation CUT')
        shutil.copytree('datasets/' + dataset_name, 'CUT/datasets/' + dataset_name)
        shutil.move('CUT/datasets/' + dataset_name + '/trainA', 'CUT/datasets/' + dataset_name + '/testA')
        shutil.move('CUT/datasets/' + dataset_name + '/trainB', 'CUT/datasets/' + dataset_name + '/testB')
        shutil.copytree('CUT/datasets/' + dataset_name + '/testB', 'CUT/datasets/' + dataset_name + '/trainB')
        shutil.copytree('CUT/datasets/' + dataset_name + '/testA', 'CUT/datasets/' + dataset_name + '/trainA')
        # os.system('cd CUT')
        os.system('cd CUT;python CUT/train.py --dataroot ./datasets/' + dataset_name + ' --name '+ dataset_name + '_CUT --CUT_mode CUT')
        os.system('cd CUT;python CUT/test.py --dataroot ./datasets/' + dataset_name + ' --name ' + dataset_name + '_CUT --CUT_mode CUT --phase train')
        shutil.move('CUT/results/'+ dataset_name + '_CUT/fake_B',output_path)
        shutil.rmtree('datasets/'+dataset_name)
        # os.system('cd ..')

    elif algo == 'fastCUT':
        set_gpu(0)
        os.system('git clone https://github.com/taesungp/contrastive-unpaired-translation CUT')
        shutil.copytree('datasets/' + dataset_name, 'CUT/datasets/' + dataset_name)
        shutil.move('CUT/datasets/' + dataset_name + '/trainA', 'CUT/datasets/' + dataset_name + '/testA')
        shutil.move('CUT/datasets/' + dataset_name + '/trainB', 'CUT/datasets/' + dataset_name + '/testB')
        shutil.copytree('CUT/datasets/' + dataset_name + '/testB', 'CUT/datasets/' + dataset_name + '/trainB')
        shutil.copytree('CUT/datasets/' + dataset_name + '/testA', 'CUT/datasets/' + dataset_name + '/trainA')
        os.system('cd CUT;python CUT/train.py --dataroot ' + dataset_name + ' --name '+dataset_name+ '_FastCUT --CUT_mode FastCUT')
        os.system('cd CUT;python CUT/test.py --dataroot ' + dataset_name + ' --name '+ dataset_name+ '_FastCUT --CUT_mode FastCUT --phase train')
        shutil.move('CUT/results/'+ dataset_name + '_FastCUT/fake_B',output_path)
        shutil.rmtree('datasets/'+dataset_name)
        # os.system('cd ..')

    else:
        print('error')

def export_generator(learn, generator_name='generator',path=Path('.'),convert_to='B'):
    import torch
    if convert_to=='B':
        model = learn.model.G_B
    elif convert_to=='A':
        model = learn.model.G_A
    else:
        raise ValueError("convert_to must be 'A' or 'B' (generator that converts either from A to B or B to A)")
    torch.save(model.state_dict(),path/(generator_name+'.pth'))

def get_preds_cyclegan(learn,test_path,pred_path,bs=4,num_workers=16,suffix='tif'):
    """
    A prediction function that takes the Learner object `learn` with the trained model, the `test_path` folder with the images to perform
    batch inference on, and the output folder `pred_path` where the predictions will be saved, with a batch size `bs`, `num_workers`,
    and suffix of the prediction images `suffix` (default='png').
    """
    from upit.inference.cyclegan import cycle_learner,load_dataset
    from fastprogress.fastprogress import progress_bar
    import torchvision
    assert os.path.exists(test_path)

    if not os.path.exists(pred_path):
        os.mkdir(pred_path)

    test_dl = load_dataset(test_path,bs,num_workers)
    model = learn.model.G_B
    model = model.cpu()
    for i, xb in progress_bar(enumerate(test_dl),total=len(test_dl)):
        fn, im = xb
        preds = (model(im)/2 + 0.5)
        for i in range(len(fn)):
            new_fn = os.path.join(pred_path,'.'.join([os.path.basename(fn[i]).split('.')[0]+'_fakeB',suffix]))
            print(new_fn)
            torchvision.utils.save_image(preds[i],new_fn)


