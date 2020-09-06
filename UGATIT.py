import time, itertools
from dataset import Selfie2AnimeDataReader
from networks import *
from utils import *
from glob import glob
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
import paddle.fluid.layers as layers
import paddle.fluid.io
import os

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        self.place = fluid.CUDAPlace(0) if self.device != 'cpu' else fluid.CPUPlace()

        self.keep_result = args.keep_result
        self.data_improve = args.data_improve

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# device : ", self.device)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        """ DataLoader """
        # # 读取数据集
        dataset = os.path.join('dataset', self.dataset)
        # 真实人物图片
        self.trainA_dr = Selfie2AnimeDataReader(os.path.join(dataset, 'trainA'), batch_size=self.batch_size, data_improve=self.data_improve)
        # 动漫人物图片
        self.trainB_dr = Selfie2AnimeDataReader(os.path.join(dataset, 'trainB'), batch_size=self.batch_size, data_improve=self.data_improve)

        self.testA_dr = Selfie2AnimeDataReader(os.path.join(dataset, 'testA'), data_improve=False)
        self.testB_dr = Selfie2AnimeDataReader(os.path.join(dataset, 'testB'), data_improve=False)

        # self.trainA_loader = Selfie2AnimeDataReader(os.path.join(dataset, 'trainA')).create_reader()
        # self.trainB_loader = Selfie2AnimeDataReader(os.path.join(dataset, 'trainB')).create_reader()
        self.testA_loader = self.testA_dr.create_reader(is_shuffle=False)
        self.testB_loader = self.testB_dr.create_reader(is_shuffle=False)

        """ Define Generator, Discriminator 
            定义生成器 和 判别器
        """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        """ Define Loss 
            定义损失
        """
        self.L1_loss = dygraph.L1Loss()
        self.MSE_loss = dygraph.MSELoss()
        self.BCE_loss = dygraph.BCELoss()


        """ Trainer 
            定义训练器
        """

        # 学习率指数衰减
        self.lr_exponential_decay = layers.exponential_decay(learning_rate=self.lr, decay_steps=2000,
                                decay_rate=(1 / (self.iteration // 2)),
                                staircase=True)


        self.G_optim = paddle.fluid.optimizer.Adam(parameter_list=self.genA2B.parameters() + self.genB2A.parameters(),
                                                   learning_rate= self.lr
                                                   , beta1=0.5, beta2= 0.999,
                                                   # , weight_decay=self.weight_decay
                                                   regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay)
                                                   )
        self.D_optim = paddle.fluid.optimizer.Adam(parameter_list= self.disGA.parameters() + self.disGB.parameters() + self.disLA.parameters() + self.disLB.parameters(),
                                                   learning_rate=self.lr
                                                   , beta1=0.5, beta2= 0.999,
                                                   # , weight_decay=self.weight_decay
                                                   regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay)
                                                   )

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN
        """
        # self.Rho_clipper = RhoClipper(0, 1)

    def clip_rho(self, net, vmin=0, vmax=1):
        for name, param in net.named_parameters():
            if 'rho' in name:
                param.set_value(fluid.layers.clip(param, vmin, vmax))

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt.*'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])

                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                start_iter = start_iter + 1
                print(" [*] Load SUCCESS")
                # TODO 固定迭代次数之后修改学习率
                # if self.decay_flag and start_iter > (self.iteration // 2):
                #     self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                #     self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            # TODO 固定迭代次数之后修改学习率
            # if self.decay_flag and step > (self.iteration // 2):
            #     self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
            #     self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A = next(trainA_iter)[0]
            except:
                trainA_iter = iter(self.trainA_dr.create_reader())
                real_A = next(trainA_iter)[0]

            try:
                real_B = next(trainB_iter)[0]
            except:
                trainB_iter = iter(self.trainB_dr.create_reader())
                real_B = next(trainB_iter)[0]

            # Update D
            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)
            
            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, layers.ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit, layers.zeros_like(fake_GA_logit))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, layers.ones_like(real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit, layers.zeros_like(fake_GA_cam_logit))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, layers.ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit, layers.zeros_like(fake_LA_logit))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, layers.ones_like(real_LA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, layers.zeros_like(fake_LA_cam_logit))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, layers.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit, layers.zeros_like(fake_GB_logit))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, layers.ones_like(real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit, layers.zeros_like(fake_GB_cam_logit))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, layers.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit, layers.zeros_like(fake_LB_logit))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, layers.ones_like(real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, layers.zeros_like(fake_LB_cam_logit))
            
            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.minimize(Discriminator_loss)
            self.D_optim.clear_gradients()

            # Update G
            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, layers.ones_like(fake_GA_logit))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, layers.ones_like(fake_GA_cam_logit))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, layers.ones_like(fake_LA_logit))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, layers.ones_like(fake_LA_cam_logit))

            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, layers.ones_like(fake_GB_logit))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, layers.ones_like(fake_GB_cam_logit))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, layers.ones_like(fake_LB_logit))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, layers.ones_like(fake_LB_cam_logit))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, layers.ones_like(fake_B2A_cam_logit)) + self.BCE_loss(fake_A2A_cam_logit, layers.zeros_like(fake_A2A_cam_logit))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, layers.ones_like(fake_A2B_cam_logit)) + self.BCE_loss(fake_B2B_cam_logit, layers.zeros_like(fake_B2B_cam_logit))
            
            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            
            self.G_optim.minimize(Generator_loss)
            self.G_optim.clear_gradients()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            # self.Rho_clipper(self.genA2B)
            # self.Rho_clipper(self.genB2A)
            self.clip_rho(self.genA2B)
            self.clip_rho(self.genB2A)
            
            timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
            print("%s [%5d/%5d] cost_time: %4.4f d_loss: %.8f, g_loss: %.8f" % (timestring, step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for _ in range(train_sample_num):
                    try:
                        real_A = next(trainA_iter)[0]
                    except:
                        trainA_iter = iter(self.trainA_dr.create_reader())
                        real_A = next(trainA_iter)[0]

                    try:
                        real_B = next(trainB_iter)[0]
                    except:
                        trainB_iter = iter(self.trainB_dr.create_reader())
                        real_B = next(trainB_iter)[0]

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
 
                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A= next(testA_iter)[0]
                    except:
                        testA_iter = iter(self.testA_dr.create_reader())
                        real_A = next(testA_iter)[0]

                    try:
                        real_B = next(testB_iter)[0]
                    except:
                        testB_iter = iter(self.testB_dr.create_reader())
                        real_B = next(testB_iter)[0]

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % 5000 == 0:
                fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(self.result_dir, self.dataset + '_params_genA2B_latest.pt'))
                fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(self.result_dir, self.dataset + '_params_genB2A_latest.pt'))
                
                fluid.save_dygraph(self.disGA.state_dict(), os.path.join(self.result_dir, self.dataset + '_params_disGA_latest.pt'))
                fluid.save_dygraph(self.disGB.state_dict(), os.path.join(self.result_dir, self.dataset + '_params_disGB_latest.pt'))
                fluid.save_dygraph(self.disLA.state_dict(), os.path.join(self.result_dir, self.dataset + '_params_disLA_latest.pt'))
                fluid.save_dygraph(self.disLB.state_dict(), os.path.join(self.result_dir, self.dataset + '_params_disLB_latest.pt'))

                fluid.save_dygraph(self.G_optim.state_dict(), os.path.join(self.result_dir, self.dataset + '_params_genA2B_latest.pt'))
                fluid.save_dygraph(self.D_optim.state_dict(), os.path.join(self.result_dir, self.dataset + '_params_disGA_latest.pt'))
                
    def save(self, dir, step):
        fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(dir, self.dataset + '_params_genA2B_%07d.pt' % step))
        fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(dir, self.dataset + '_params_genB2A_%07d.pt' % step))
        
        fluid.save_dygraph(self.disGA.state_dict(), os.path.join(dir, self.dataset + '_params_disGA_%07d.pt' % step))
        fluid.save_dygraph(self.disGB.state_dict(), os.path.join(dir, self.dataset + '_params_disGB_%07d.pt' % step))
        fluid.save_dygraph(self.disLA.state_dict(), os.path.join(dir, self.dataset + '_params_disLA_%07d.pt' % step))
        fluid.save_dygraph(self.disLB.state_dict(), os.path.join(dir, self.dataset + '_params_disLB_%07d.pt' % step))

        fluid.save_dygraph(self.G_optim.state_dict(), os.path.join(dir, self.dataset + '_params_genA2B_%07d.pt' % step))
        fluid.save_dygraph(self.D_optim.state_dict(), os.path.join(dir, self.dataset + '_params_disGA_%07d.pt' % step))

        # 删除旧模型, 前3个模型，避免存储不够
        if not self.keep_result:
            del_step = step - 3*self.save_freq
            if del_step > 0:
                cmd_tmp = os.popen('rm -r %s*_%07d.pt*' %  (os.path.join(dir, self.dataset), del_step)).readlines()
                print('删除历史模型记录: rm -r %s*_%07d.pt*' %  (os.path.join(dir, self.dataset), del_step), cmd_tmp)
        

    def load(self, dir, step):
        params_genA2B, g_opt = fluid.load_dygraph(os.path.join(dir, self.dataset + '_params_genA2B_%07d.pt' % step))
        params_genB2A, _ = fluid.load_dygraph(os.path.join(dir, self.dataset + '_params_genB2A_%07d.pt' % step))
        params_disGA, d_opt = fluid.load_dygraph(os.path.join(dir, self.dataset + '_params_disGA_%07d.pt' % step))
        params_disGB, _ = fluid.load_dygraph(os.path.join(dir, self.dataset + '_params_disGB_%07d.pt' % step))
        params_disLA, _ = fluid.load_dygraph(os.path.join(dir, self.dataset + '_params_disLA_%07d.pt' % step))
        params_disLB, _ = fluid.load_dygraph(os.path.join(dir, self.dataset + '_params_disLB_%07d.pt' % step))

        self.genA2B.load_dict(params_genA2B)
        self.genB2A.load_dict(params_genB2A)
        self.disGA.load_dict(params_disGA)
        self.disGB.load_dict(params_disGB)
        self.disLA.load_dict(params_disLA)
        self.disLB.load_dict(params_disLB)

        self.G_optim.set_dict(g_opt)
        self.D_optim.set_dict(d_opt)

        # 设置学习率 修改
        # if self.lr != self.G_optim.current_step_lr():
        #     print('set self.G_optim.current_step_lr():{} to {}'.format(self.G_optim.current_step_lr(), self.lr))
        #     self.G_optim._learning_rate = self.lr
        # else:
        #     print('self.G_optim.current_step_lr():{}'.format(self.G_optim.current_step_lr()))
        # if self.lr != self.D_optim.current_step_lr():
        #     print('set self.D_optim.current_step_lr():{} to {}'.format(self.D_optim.current_step_lr(), self.lr))
        #     self.D_optim._learning_rate = self.lr
        # else:
        #     print('self.D_optim.current_step_lr():{}'.format(self.D_optim.current_step_lr()))

        # self.G_optim._learning_rate = self.lr_exponential_decay 
        # self.D_optim._learning_rate = self.lr_exponential_decay 

        print('self.G_optim.current_step_lr():{}'.format(self.G_optim.current_step_lr()))
        print('self.D_optim.current_step_lr():{}'.format(self.D_optim.current_step_lr()))
        print('===== 加载模型完毕！======')

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt.*'))

        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()

        n = 0
        for real_A in self.testA_loader:
            real_A = real_A[0]

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'single', 'A2B_%d.png' % (n + 1)), RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0)
            n = n + 1

        n = 0
        for real_B in self.testB_loader:
            real_B = real_B[0]
            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)
            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),   
                                  cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'single', 'B2A_%d.png' % (n + 1)), RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))) * 255.0)
            n = n + 1
