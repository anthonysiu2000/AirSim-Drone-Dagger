import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)
import racing_models.cmvae
import racing_utils

# DEFINE TESTING META PARAMETERS
data_dir = 'airsim_datasets/soccer_50k'
read_table = True
latent_space_constraints = True
capsule_network = True
weights_path = 'model_outputs/cmvae_capsule_con/cmvae_model_55.ckpt'

n_z = 10
img_res = 64
num_imgs_display = 50
columns = 10
rows = 10

num_interp_z = 10
idx_close = 0  #7
idx_far = 1  #39

z_range_mural = [-0.02, 0.02]
z_num_mural = 11

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load test dataset
train_ds, test_ds = racing_utils.dataset_utils.create_dataset_csv(data_dir, 16, img_res)
images_np, raw_table = racing_utils.dataset_utils.create_test_dataset_csv(data_dir, img_res, read_table=read_table)
print('Done with dataset')

def compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode):
    # compute reconstruction loss
    if mode == 0:
        img_loss = tf.losses.mean_squared_error(img_gt, img_recon)
        # img_loss = tf.losses.mean_absolute_error(img_gt, img_recon)
        gate_loss = tf.losses.mean_squared_error(gate_gt, gate_recon)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum((1 + stddev - tf.math.pow(means, 2) - tf.math.exp(stddev)), axis=1))
    # elif mode == 1:
    #     # labels = tf.reshape(labels, predictions.shape)
    #     # recon_loss = tf.losses.mean_squared_error(labels, predictions)
    #     # recon_loss = loss_object(labels, predictions)
    # print('Predictions: {}'.format(predictions))
    # print('Labels: {}'.format(labels))
    # print('Lrec: {}'.format(recon_loss))
    # copute KL loss: D_KL(Q(z|X,y) || P(z|X))
    return img_loss, gate_loss, kl_loss

def test(img_gt, gate_gt, mode):
    img_recon, gate_recon, means, stddev, z = model(img_gt, mode)
    img_loss, gate_loss, kl_loss = compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode)
    img_loss = tf.reduce_mean(img_loss)
    gate_loss = tf.reduce_mean(gate_loss)
    if mode == 0:
        test_loss_rec_img.update_state(img_loss)
        test_loss_rec_gate.update_state(gate_loss)
        test_loss_kl.update_state(kl_loss)

def train(img_gt, gate_gt, mode):
    img_recon, gate_recon, means, stddev, z = model(img_gt, mode)
    img_loss, gate_loss, kl_loss = compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode)
    img_loss = tf.reduce_mean(img_loss)
    gate_loss = tf.reduce_mean(gate_loss)
    if mode == 0:
        train_loss_rec_img.update_state(img_loss)
        train_loss_rec_gate.update_state(gate_loss)
        train_loss_kl.update_state(kl_loss)

images_np = images_np[:1000,:]
if read_table is True:
    raw_table = raw_table[:1000,:]

# create model
if latent_space_constraints is True:
    model = racing_models.cmvae.CmvaeDirect(n_z=n_z, gate_dim=4, res=img_res, trainable_model=True, capsule_network=capsule_network)
else:
    model = racing_models.cmvae.Cmvae(n_z=n_z, gate_dim=4, res=img_res, trainable_model=True, capsule_network=capsule_network)

model.load_weights(weights_path)



# define metrics
train_loss_rec_img = tf.keras.metrics.Mean(name='train_loss_rec_img')
train_loss_rec_gate = tf.keras.metrics.Mean(name='train_loss_rec_gate')
train_loss_kl = tf.keras.metrics.Mean(name='train_loss_kl')
test_loss_rec_img = tf.keras.metrics.Mean(name='test_loss_rec_img')
test_loss_rec_gate = tf.keras.metrics.Mean(name='test_loss_rec_gate')
test_loss_kl = tf.keras.metrics.Mean(name='test_loss_kl')

mode = 0
flag = True
# print('MODE NOW: {}'.format(mode))
for train_images, train_labels in train_ds:
    train(train_images, train_labels, mode)
for test_images, test_labels in test_ds:
    test(test_images, test_labels, mode)

if mode == 0:
    epoch = 0
    print('Epoch {} | TRAIN: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {} | TEST: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {}'
            .format(0, train_loss_rec_img.result(), train_loss_rec_gate.result(), train_loss_kl.result(),
                    train_loss_rec_img.result()+train_loss_rec_gate.result()+train_loss_kl.result(),
                    test_loss_rec_img.result(), test_loss_rec_gate.result(), test_loss_kl.result(),
                    test_loss_rec_img.result() + test_loss_rec_gate.result() + test_loss_kl.result()
                    ))


img_recon, gate_recon, means, stddev, z = model(images_np, mode=0)
img_recon = img_recon.numpy()
gate_recon = gate_recon.numpy()
z = z.numpy()

# de-normalization of gates and images
images_np = ((images_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
img_recon = ((img_recon + 1.0) / 2.0 * 255.0).astype(np.uint8)
gate_recon = racing_utils.dataset_utils.de_normalize_gate(gate_recon)

# if not read_table:
#     sys.exit()

# get stats for gate reconstruction
if read_table is True:
    racing_utils.stats_utils.calculate_gate_stats(gate_recon, raw_table)

# show some reconstruction figures
fig = plt.figure(figsize=(20, 20))
for i in range(1, num_imgs_display+1):
    idx_orig = (i-1)*2+1
    fig.add_subplot(rows, columns, idx_orig)
    img_display = racing_utils.dataset_utils.convert_bgr2rgb(images_np[i - 1, :])
    plt.axis('off')
    plt.imshow(img_display)
    fig.add_subplot(rows, columns, idx_orig+1)
    img_display = racing_utils.dataset_utils.convert_bgr2rgb(img_recon[i-1, :])
    plt.axis('off')
    plt.imshow(img_display)
fig.savefig('reconstruction_results.png')
plt.show()

# show interpolation btw two images in latent space
z_close = z[idx_close, :]
z_far = z[idx_far, :]
z_interp = racing_utils.geom_utils.interp_vector(z_close, z_far, num_interp_z)


# get the image predictions
img_recon_interp, gate_recon_interp = model.decode(z_interp, mode=0)
img_recon_interp = img_recon_interp.numpy()
gate_recon_interp = gate_recon_interp.numpy()

# de-normalization of gates and images
img_recon_interp = ((img_recon_interp + 1.0) / 2.0 * 255.0).astype(np.uint8)
gate_recon_interp = racing_utils.dataset_utils.de_normalize_gate(gate_recon_interp)

# join predictions with array and print
indices = np.array([np.arange(num_interp_z)]).transpose()
results = np.concatenate((indices, gate_recon_interp), axis=1)
print('Img index | Predictions: = \n{}'.format(results))


fig, axs = plt.subplots(1, 4, tight_layout=True)
axs[0].plot(np.arange(gate_recon_interp.shape[0]), gate_recon_interp[:, 0], 'b-', label='r')
axs[1].plot(np.arange(gate_recon_interp.shape[0]), gate_recon_interp[:, 1]*180/np.pi, 'b-', label=r'$\theta$')
axs[2].plot(np.arange(gate_recon_interp.shape[0]), gate_recon_interp[:, 2]*180/np.pi, 'b-', label=r'$\phi$')
axs[3].plot(np.arange(gate_recon_interp.shape[0]), gate_recon_interp[:, 3]*180/np.pi, 'b-', label=r'$\psi$')

for idx in range(4):
    # axs[idx].grid()
    y_ticks_array = gate_recon_interp[:, idx][np.array([0, gate_recon_interp[:, idx].shape[0]-1])]
    y_ticks_array = np.around(y_ticks_array, decimals=1)
    if idx > 0:
        y_ticks_array = y_ticks_array * 180 / np.pi
    axs[idx].set_yticks(y_ticks_array)
    axs[idx].set_xticks(np.array([0, 9]))
    axs[idx].set_xticklabels((r'$I_a$', r'$I_b$'))

axs[0].set_title(r'$r$')
axs[1].set_title(r'$\theta$')
axs[2].set_title(r'$\phi$')
axs[3].set_title(r'$\psi$')

axs[0].set_ylabel('[meter]')
axs[1].set_ylabel(r'[deg]')
axs[2].set_ylabel(r'[deg]')
axs[3].set_ylabel(r'[deg]')

# plot the interpolated images
fig2 = plt.figure(figsize=(96, 96))
columns = num_interp_z + 2
rows = 1
fig2.add_subplot(rows, columns, 1)
img_display = racing_utils.dataset_utils.convert_bgr2rgb(images_np[idx_close, :])
plt.axis('off')
plt.imshow(img_display)
for i in range(1, num_interp_z + 1):
    fig2.add_subplot(rows, columns, i+1)
    img_display = racing_utils.dataset_utils.convert_bgr2rgb(img_recon_interp[i - 1, :])
    plt.axis('off')
    plt.imshow(img_display)
fig2.add_subplot(rows, columns, num_interp_z + 2)
img_display = racing_utils.dataset_utils.convert_bgr2rgb(images_np[idx_far, :])
plt.axis('off')
plt.imshow(img_display)
fig2.savefig('reconstruction_interpolation_results.png')
plt.show()

# new plot traveling through latent space
fig3 = plt.figure(figsize=(96, 96))
columns = z_num_mural
rows = n_z
z_values = racing_utils.geom_utils.interp_vector(z_range_mural[0], z_range_mural[1], z_num_mural)
for i in range(1, z_num_mural*n_z + 1):
    fig3.add_subplot(rows, columns, i)
    z = np.zeros((1, n_z)).astype(np.float32)
    z[0, int((i-1)//columns)] = z_values[i%columns-1]
    # print (z)
    img_recon_interp, gate_recon_interp = model.decode(z, mode=0)
    img_recon_interp = img_recon_interp.numpy()
    img_recon_interp = ((img_recon_interp[0, :] + 1.0) / 2.0 * 255.0).astype(np.uint8)
    img_display = racing_utils.dataset_utils.convert_bgr2rgb(img_recon_interp)
    plt.axis('off')
    plt.imshow(img_display)
fig3.savefig('z_mural.png')
plt.show()
