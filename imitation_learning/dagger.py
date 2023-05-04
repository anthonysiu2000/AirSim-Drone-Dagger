from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import os
import sys
import airsimdroneracingvae

# imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)
import racing_models
import racing_utils


class DAgger():
    def __init__(self, learner_weights_path,
                 expert_type, expert_weights_path,
                 expert_feature_weights_path=None,
                 expert_latent_space_constraints=True):
        
        # learner is BC-full; hardcoded
        self.learner_model = racing_models.bc_full.BcFull()
        self.learner_model.load_weights(learner_weights_path)

        self.expert_type = expert_type

        # create models
        if self.expert_type == 'full':
            self.expert_model = racing_models.bc_full.BcFull()
            self.expert_model.load_weights(expert_weights_path)
        elif self.expert_type == 'latent':
            # create model
            if expert_latent_space_constraints:
                self.cmvae_model = racing_models.cmvae.CmvaeDirect(n_z=10, gate_dim=4, res=64, trainable_model=False)
            else:
                self.cmvae_model = racing_models.cmvae.Cmvae(n_z=10, gate_dim=4, res=64, trainable_model=False)
            self.cmvae_model.load_weights(expert_feature_weights_path)
            self.expert_model = racing_models.bc_latent.BcLatent()
            self.expert_model.load_weights(expert_weights_path)
        elif self.expert_type == 'reg':
            self.expert_reg = racing_models.dronet.Dronet(num_outputs=4, include_top=True)
            self.expert_reg.load_weights(expert_feature_weights_path)
            self.expert_model = racing_models.bc_latent.BcLatent()
            self.expert_model.load_weights(expert_weights_path)

    def predict_velocities(self, img, p_o_b):
        img = (img / 255.0) * 2 - 1.0
        if self.regressor_type == 'full':
            predictions = self.expert_model(img)
        elif self.regressor_type == 'latent':
            z, _, _ = self.cmvae_model.encode(img)
            predictions = self.expert_model(z)
        elif self.regressor_type == 'reg':
            z = self.expert_reg(img)
            predictions = self.expert_model(z)
        predictions = predictions.numpy()
        predictions = racing_utils.dataset_utils.de_normalize_v(predictions)
        # print('Predicted body vel: \n {}'.format(predictions[0]))
        v_xyz_world = racing_utils.geom_utils.convert_t_body_2_world(airsimdroneracingvae.Vector3r(predictions[0,0], predictions[0,1], predictions[0,2]), p_o_b.orientation)
        return np.array([v_xyz_world.x_val, v_xyz_world.y_val, v_xyz_world.z_val, predictions[0,3]])
    

