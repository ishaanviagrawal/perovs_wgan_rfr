import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Input
from numpy import mean
from numpy import ones
from numpy.random import randn
from keras import backend
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from keras.models import load_model
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")


class WGAN:

    def __init__(self, len, latent_dim):
        self.len = len
        self.latent_dim = latent_dim

    def gen_noise_vector(self, n_samples):
        '''
        Generating the random noise vector/latent vector that is given as input to the generator
        '''
        x = randn(self.latent_dim * n_samples)
        x = x.reshape(n_samples, self.latent_dim)
        return x

    def gen_fake_vector(self, generator, n_samples):
        '''
        Returns the fake samples created by the generator
        '''
        x = self.gen_noise_vector(n_samples)
        X = generator.predict(x)
        y = np.ones((n_samples, 1))
        return X, y

    def gen_real_vector(self, n, data):
        '''
        Randomly sampling real samples from the training data
        '''
        X = data.sample(n)
        y = -np.ones((n, 1))
        return X, y
    
    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)
    
    def critic(self):
        weight_initial = RandomNormal(stddev=0.02)
        
        model = Sequential()
        model.add(Input(shape=(self.len)))
        model.add(Dense(32, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        model.compile(loss = self.wasserstein_loss, optimizer = RMSprop(lr=0.00001))
        return model
    
    def generator(self):
        weight_initial = RandomNormal(stddev=0.02)
        
        model = Sequential()
        model.add(Input(shape=(self.latent_dim,)))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(self.len))
        return model
    
    def wgan(self, gen, critic):
        for layer in critic.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        model = Sequential()
        model.add(gen)
        model.add(critic)

        optimizer = RMSprop(lr=0.00001)
        model.compile(loss = self.wasserstein_loss, optimizer = optimizer)
        return model

    def training_loop(self, generator, critic, wgan, data, epochs=10, n_batch=1000, n_critic=5):
        '''
        Training loop for the WGAN, for every time the generator is updated, the critic is updated 5 times
        '''
        epoch_batch_num = int(data.shape[0] / n_batch)
        n_steps = epoch_batch_num * epochs
        half_batch = int(n_batch / 2)
        c1 = []
        c2 = []
        g = []

        for i in range(n_steps):
            c1_temp = []
            c2_temp = []

            for _ in range(n_critic):
                #Generating the real and fake samples to give to the critic
                X_real, y_real = self.gen_real_vector(half_batch, data)
                X_fake, y_fake = self.gen_fake_vector(generator, half_batch)

                #Critic is trained on the real and fake samples
                c1_loss = critic.train_on_batch(X_real, y_real)
                c1_temp.append(c1_loss)
                c2_loss = critic.train_on_batch(X_fake, y_fake)
                c2_temp.append(c2_loss)
            
            c1.append(mean(c1_temp))
            c2.append(mean(c2_temp))

            X_gan = self.gen_noise_vector(n_batch)
            y_gan = -ones((n_batch, 1))

            #training the combined generator and discriminator
            g_loss = wgan.train_on_batch(X_gan, y_gan)
            g.append(g_loss)

        return generator
    
    def train_wgan(self, data, path, len_data=15):
        '''
        Function to train and save a new model
        data: training dataset
        path: file name for the model to be saved
        len_data: length of the composition vector (14-dim composition vector + phase)
        '''
        critic = self.critic()
        generator = self.generator()
        gan_model = self.wgan(generator, critic)
        if len_data < 5000:
            model = self.training_loop(generator, critic, gan_model, data, n_batch = 64)
        else:
            model = self.training_loop(generator, critic, gan_model, data, n_batch = 3000)
        model.save(path)
        return model
    
    def gen_phase(self, data_gen, n_gen):
        '''
        Calculates the final phase for all the compounds
        data_gen: data generated by the model
        n_gen: total number of generated compounds

        '''
        for i in range(n_gen):
            data_gen.loc[i, 'Phase'] = math.floor(data_gen.loc[i, 'Phase'])
            if data_gen.loc[i, 'Phase'] < 1:
                data_gen.loc[i, 'Phase'] = 1

        data_gen['Phase'] = data_gen['Phase'].astype(int)
        return data_gen

    
    def clean_comp(self, data_gen, n_gen, f_el):
        '''
        n_gen: number of generated samples
        ensures that the sum of composition of all A, B, and X site species is 1
        '''
        for i in range(n_gen):
            for a in f_el:
                if data_gen.loc[i, a] < 0:
                    data_gen.loc[i, a] = abs(data_gen.loc[i, a])

            sum_A = 0
            sum_B = 0
            sum_X = 0
            for a in range(7):
                sum_A += data_gen.loc[i, f_el[a]]

            for a in range(8):
                sum_B += data_gen.loc[i, f_el[a+7]]

            for a in range(4):
                sum_X += data_gen.loc[i, f_el[a+15]]


            for k in f_el[0:7]:
                data_gen.loc[i, k] = data_gen.loc[i, k]/sum_A
            
            for k in f_el[7:15]:
                data_gen.loc[i, k] = data_gen.loc[i, k]/sum_B
            
            for k in f_el[15:20]:
                data_gen.loc[i, k] = data_gen.loc[i, k]/sum_X

        return data_gen


    def calculate_properties(self, data_gen, n_gen):
        '''
        composition_vector: 15 dim generated samples with composition, phase, Decomp, Gap, and SLME
        calulates the properties of A/B/X site species according to composition and gives 54 dim vector

        '''
        A_list = ['A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period']
        B_list = ['B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 'B_period']
        X_list = ['X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period']
        sp_properties = el_prop
        sp_properties.set_axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], axis='columns', inplace=True)

        for i in range(n_gen):

            for k in range(len(A_list)):
                data_gen.loc[i, A_list[k]] = 0
                for a in range(7):
                     data_gen.loc[i, A_list[k]] += data_gen.loc[i, f_el[a]]*sp_properties.loc[f_el[a], k]

            for k in range(len(B_list)):
                data_gen.loc[i, B_list[k]] = 0
                for a in range(8):
                     data_gen.loc[i, B_list[k]] += data_gen.loc[i, f_el[a+6]]*sp_properties.loc[f_el[a+7], k]

            for k in range(len(X_list)):
                data_gen.loc[i, X_list[k]] = 0
                for a in range(4):
                     data_gen.loc[i, X_list[k]] += data_gen.loc[i, f_el[a+14]]*sp_properties.loc[f_el[a+15], k] 

        return data_gen
    
    def gen_novel_comps(self, train_data, gen, n_gen=1000):
        '''
        Function to generate novel compositions with calculated properties given a trained model
        train_data: training dataset
        n_gen: total number of compositions to be generated
        '''
        noise_vector = self.gen_noise_vector(n_gen)
        X = gen.predict(noise_vector)
        comp_vector = pd.DataFrame(data=X,  columns=train_data.columns)
        comp_vector = self.gen_phase(comp_vector, n_gen)
        comp_vector = self.clean_comp(comp_vector, n_gen)
        comp_vector = self.calculate_properties(comp_vector, n_gen)
        return comp_vector