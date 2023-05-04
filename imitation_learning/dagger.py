from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import os
import sys
import airsimdroneracingvae
import tensorflow as tf

# imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)
import racing_models
import racing_utils

@tf.function
def compute_loss(labels, predictions):
    recon_loss = tf.losses.mean_squared_error(labels, predictions)
    return recon_loss

class DAgger():
    def __init__(self, learner_encoder, learner_model,
                 expert_type, expert_weights_path,
                 expert_feature_weights_path=None,
                 expert_latent_space_constraints=True,
                 expert_use_capsnet=False,
                 batch_size = 32,
                 learning_rate = 1e-3):
        
        self.expert_type = expert_type

        # learner is BC-img; hardcoded
        self.learner_encoder = learner_encoder
        self.learner_model = learner_model

        self.images = []
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.batch_size = batch_size

        # create models
        if self.expert_type == 'full':
            self.expert_model = racing_models.bc_full.BcFull()
            self.expert_model.load_weights(expert_weights_path).expect_partial()
        elif self.expert_type == 'latent':
            # create model
            if expert_latent_space_constraints:
                self.cmvae_model = racing_models.cmvae.CmvaeDirect(n_z=10, gate_dim=4, res=64,
                                                                   trainable_model=False,
                                                                   capsule_network=expert_use_capsnet)
            else:
                self.cmvae_model = racing_models.cmvae.Cmvae(n_z=10, gate_dim=4, res=64, trainable_model=False,
                                                             capsule_network=expert_use_capsnet)
            self.cmvae_model.load_weights(expert_feature_weights_path).expect_partial()
            self.expert_model = racing_models.bc_latent.BcLatent()
            self.expert_model.load_weights(expert_weights_path).expect_partial()
        elif self.expert_type == 'reg':
            self.expert_reg = racing_models.dronet.Dronet(num_outputs=4, include_top=True)
            self.expert_reg.load_weights(expert_feature_weights_path).expect_partial()
            self.expert_model = racing_models.bc_latent.BcLatent()
            self.expert_model.load_weights(expert_weights_path).expect_partial()

    def predict_velocities(self, img, p_o_b, bc):
        img = (img / 255.0) * 2 - 1.0
        self.images.append(img)
        if bc:
            predictions = self.get_expert_predictions(img)
        else:
            z, _, _ = self.learner_encoder.encode(img)
            predictions = self.learner_model(z).numpy()
        predictions = racing_utils.dataset_utils.de_normalize_v(predictions)
        # print('Predicted body vel: \n {}'.format(predictions[0]))
        v_xyz_world = racing_utils.geom_utils.convert_t_body_2_world(airsimdroneracingvae.Vector3r(predictions[0,0], predictions[0,1], predictions[0,2]), p_o_b.orientation)
        return np.array([v_xyz_world.x_val, v_xyz_world.y_val, v_xyz_world.z_val, predictions[0,3]])
    
    def get_expert_predictions(self, img):
        if self.expert_type == 'full':
            expert_predictions = self.expert_model(img)
        elif self.expert_type == 'latent':
            z, _, _ = self.cmvae_model.encode(img)
            expert_predictions = self.expert_model(z)
        elif self.expert_type == 'reg':
            z = self.expert_reg(img)
            expert_predictions = self.expert_model(z)
        expert_predictions = expert_predictions.numpy()
        return expert_predictions

    def train_learner(self):
        if len(self.images) > 20000:
            # take the last 1000 examples
            self.images = self.images[-20000:]
        losses = []
        for i in range(0, len(self.images), self.batch_size):
            start = i
            end = min(i + self.batch_size, len(self.images))
            if end - start < 2:
                continue
            images = np.concatenate(self.images[start:end], axis=0)
            expert_predictions = self.get_expert_predictions(images)
            with tf.GradientTape() as tape:
                z, _, _ = self.learner_encoder.encode(images)
                predictions = self.learner_model(z)
                recon_loss = tf.reduce_mean(compute_loss(expert_predictions, predictions))
                losses.append(float(recon_loss))
            gradients = tape.gradient(recon_loss, self.learner_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.learner_model.trainable_variables))
        return np.mean(losses)
    

