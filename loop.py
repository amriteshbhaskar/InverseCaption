import torch
from torchvision import transforms
import torchvision
from cfg import d
from torch import autograd
import os
from generator import generator
from discriminator import discriminator
from torch.utils.data import DataLoader
from CUBDataset import CUBDataset
from torch.autograd import Variable
import numpy as np


class Helper(object):
    def save_model(self, netD, netG, dir_path, epoch):
        path = os.path(dir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/Discriminator_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/Generator_{1}.pth'.format(path, epoch))

    def initializeWieights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Logger(object):
    def __init__(self, vis_screen):
        self.viz = VisdomPlotter(env_name=vis_screen)
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    def log_iteration_wgan(self, epoch, gen_iteration, d_loss, g_loss, real_loss, fake_loss):
        print("Epoch: %d, Gen_iteration: %d, d_loss= %f, g_loss= %f, real_loss= %f, fake_loss = %f" %
              (epoch, gen_iteration, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_loss, fake_loss))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())

    def log_iteration_gan(self, epoch, d_loss, g_loss, real_score, fake_score):
        print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
            epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
            fake_score.data.cpu().mean()))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())
        self.hist_Dx.append(real_score.data.cpu().mean())
        self.hist_DGx.append(fake_score.data.cpu().mean())

    def plot_epoch(self, epoch):
        self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        self.hist_D = []
        self.hist_G = []

    def plot_epoch_w_scores(self, epoch):
        self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        self.viz.plot('D(X)', 'train', epoch, np.array(self.hist_Dx).mean())
        self.viz.plot('D(G(X))', 'train', epoch, np.array(self.hist_DGx).mean())
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    def draw(self, right_images, fake_images):
        self.viz.draw('generated images', fake_images.data.cpu().numpy()[:64] * 128 + 128)
        self.viz.draw('real images', right_images.data.cpu().numpy()[:64] * 128 + 128)


class Trainer(object):
    def __init__(self, data_dir, batch_size, epochs, save_path, learning_rate, split):
        dev = d()
        if dev.cuda == False:
            self.gen = generator()
            self.disc = discriminator()
        else:
            self.gen = torch.nn.DataParallel(generator().cuda())
            self.disc = torch.nn.DataParallel(discriminator.cuda())

        self.dataset = CUBDataset(data_dir, split=split)
        self.dataLoader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        self.optimD = torch.optim.Adam(self.disc.parameters(), lr=learning_rate, betas=(0.5, 0.99), amsgrad=True)
        self.optimG = torch.optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(0.5, 0.99), amsgrad=True)

        self.model_dir = save_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.train()

    def train(self):
        print("Training Started")
        bce = torch.nn.BCELoss()
        mse = torch.nn.MSELoss()
        l1 = torch.nn.L1Loss()

        for epoch in range(self.epochs):
            iter = 0
            # print(type(self.dataset))
            # print(self.dataset)
            # i = 0
            for point in enumerate(self.dataLoader):
                # print(len(point))
                correct_image = point['correct_image']
                incorrect_image = point['incorrect_image']
                correct_embed = point['correct_embed']
                #---------------------------------------------------------------------

                #For discriminator
                correct_image = Variable(correct_image.float()).cuda()
                incorrect_image = Variable(incorrect_image.float()).cuda()
                correct_embed = Variable(correct_embed.float()).cuda()

                incorrect_labels = Variable(np.zeroes(self.batch_size)).cuda()
                # One Sided Label Smoothing
                correct_labels = torch.FloatTensor(np.ones(self.batch_size) + -1)
                correct_labels  = Variable(correct_labels).cuda()

                self.disc.zero_grad()
                # Right images and right caption
                output, activations = self.disc(correct_image, correct_labels)
                correct_loss = bce(output, correct_labels)
                # Wrong image and right caption
                output, activations = self.disc(incorrect_image, correct_labels)
                incorrect_loss = bce(output, incorrect_labels)

                #Generated image and right captions
                noise = Variable(torch.random(self.batch_size, 100)).cuda()
                noise = noise.view(self.batch_size, 100, 1, 1)
                # Feeding it to the discriminator
                generated_images = Variable(self.gen(noise, correct_labels)).cuda()
                output, activations = self.disc(generated_images, correct_labels)
                generated_loss = torch.mean(output)
                # Calculating the net loss
                net_loss = generated_loss + correct_loss + incorrect_loss
                net_loss.backward()
                # Taking one more step towards convergence
                self.optimD.step()
                # ----------------------------------------------------------------------------
                #For generator
                self.gen.zero_grad()
                noise = Variable(torch.random(self.batch_size, 100)).cuda()
                noise = noise.view(self.batch_size, 100, 1, 1)

                generated_images = Variable(self.gen(noise, correct_labels)).cuda()
                output, generated = self.disc(generated_images, correct_labels)
                output, real = self.disc(correct_image, correct_labels)

                generated = torch.mean(generated, 0)
                real = torch.mean(real, 0)

                net_loss = bce(output, correct_labels) + mse(generated, real)*100 + 50*l1(generated_images, correct_image)
                net_loss.backward()
                self.optimG.step()
