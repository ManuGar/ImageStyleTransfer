import os
import shutil
from upit.data.unpaired import *
from upit.models.cyclegan import *
from upit.train.cyclegan import *
from upit.inference.cyclegan import *
from fastai.vision.all import *
import torch

torch.cuda.set_device(1)
def generate_images(algo, dataset_name, output_path):
    if algo == 'upit':
        os.system('pip install git+https://github.com/tmabraham/upit.git >/dev/null')


        trainA_path = Path('datasets/'+dataset_name +'trainA')
        trainB_path = Path('datasets/'+dataset_name +'trainB')
        print(f"There are {len(trainA_path.ls())} photos to tranfer to the original style")
        print(f"There are {len(trainB_path.ls())} Original style")

        dls = get_dls(trainA_path, trainB_path,load_size=256,crop_size=256,bs=4)
        cycle_gan = CycleGAN(3,3,64,gen_blocks=3)
        learn = cycle_learner(dls, cycle_gan,opt_func=partial(Adam,mom=0.5,sqr_mom=0.999),show_img_interval=8)
        learn.lr_find()
        learn.fit_flat_lin(7,7,2e-4)
        if os.path.exists('models/upitModel.pth'): os.remove('models/upitModel.pth')
        export_generator(learn, generator_name='models/upitModel')
        assert os.path.exists('models/upitModel.pth')

        #pred_path = '../GenerateDatasetCycleGAN/'
        pred_path = output_path

        cycle_gan.G_B.load_state_dict(torch.load('models/upitModel.pth'))
        learn = cycle_learner(dls, cycle_gan)
        get_preds_cyclegan(learn,trainA_path,pred_path,suffix='jpg',bs=1)

    elif algo == 'forkGAN':
        os.system('git clone https://github.com/zhengziqiang/ForkGAN')
        os.system('cd ForkGAN')
        os.system('python main.py --phase train --dataset_dir '+dataset_name+ ' --epoch 20 --gpu 1 --n_d 2 --n_scale 2 --checkpoint_dir ./check/'+dataset_name+' --sample_dir ./check/'+dataset_name + '/sample --L1_lambda 10')
        os.system('python main.py --phase test --dataset_dir '+ datset_name+ ' --gpu 1 --n_d 2 --n_scale 2 --checkpoint_dir ./check/'+dataset_name + ' --test_dir ./check/'+ dataset_name +'/testa2b --which_direction AtoB')
        os.system('cd ..')
        shutil.move('ForkGAN/check/'+ dataset_name+ '/testa2b',output_path)

    elif algo == 'ganilla':
        os.system('git clone https://github.com/giddyyupp/ganilla.git')
        os.system('cd ganilla')
        os.system('pip install requirements.txt')
        os.system('python train.py --dataroot ./datasets/'+ dataset_name + ' --name '+dataset_name+'_cyclegan --model cycle_gan --netG resnet_fpn')
        os.system('python test.py --dataroot ./datasets/'+dataset_name+' --name '+ dataset_name+'_cyclegan --model cycle_gan --netG resnet_fpn')
        os.system('cd ..')
        shutil.move('ganilla/results/'+ dataset_name + '_cyclegan/test_100/images',output_path)

    elif algo == 'dualGAN':
        os.system('git clone https://github.com/duxingren14/DualGAN.git')
        os.system('cd DualGAN')
        os.system('python main.py --phase train --dataset_name '+ dataset_name + ' --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100')
        os.system('python main.py --phase test --dataset_name '+ dataset_name + ' --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100')
        os.system('cd ..')
        shutil.move('DualGAN/test/'+ dataset_name+ '-img_sz_256-fltr_dim_64-L1-lambda_AB_1000.0_1000.0',output_path)

    elif algo == 'CUT':
        os.system('git clone https://github.com/taesungp/contrastive-unpaired-translation CUT')
        os.system('cd CUT')
        os.system('python train.py --dataroot ./datasets/' + dataset_name + ' --name '+ dataset_name + '_CUT --CUT_mode CUT')
        os.system('python test.py --dataroot ./datasets/' + dataset_name + ' --name ' + dataset_name + '_CUT --CUT_mode CUT --phase train')
        os.system('cd ..')
        shutil.move('CUT/results/'+ dataset_name + '_CUT/fake_B',output_path)

    elif algo == 'fastCUT':
        os.system('git clone https://github.com/taesungp/contrastive-unpaired-translation CUT')
        os.system('cd CUT')
        os.system('python train.py --dataroot ./datasets/' + dataset_name + ' --name '+dataset_name+ '_FastCUT --CUT_mode FastCUT')
        os.system('python test.py --dataroot ./datasets/' + dataset_name + ' --name '+ dataset_name+ '_FastCUT --CUT_mode FastCUT --phase train')
        os.system('cd ..')
        shutil.move('CUT/results/'+ dataset_name + '_FastCUT/fake_B',output_path)


    elif algo == 'sinCUT':
        os.system('git clone https://github.com/taesungp/contrastive-unpaired-translation CUT')
        os.system('cd CUT')
        os.system('python train.py --model sincut --name '+ dataset_name + ' --dataroot ./datasets/' + dataset_name)
        os.system('python test.py --model sincut --name ' + dataset_name + ' --dataroot ./datasets/' + dataset_name)
        os.system('cd ..')
        shutil.move('CUT/results/'+ dataset_name + '_SinCUT/fake_B',output_path)

    else:
        print('error')


    def export_generator(learn, generator_name='generator',path=Path('.'),convert_to='B'):
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

        assert os.path.exists(test_path)

        if not os.path.exists(pred_path):
            os.mkdir(pred_path)

        test_dl = load_dataset(test_path,bs,num_workers)
        model = learn.model.G_B
        for i, xb in progress_bar(enumerate(test_dl),total=len(test_dl)):
            fn, im = xb
            preds = (model(im)/2 + 0.5)
            for i in range(len(fn)):
                new_fn = os.path.join(pred_path,'.'.join([os.path.basename(fn[i]).split('.')[0]+'_fakeB',suffix]))
                print(new_fn)
                torchvision.utils.save_image(preds[i],new_fn)


