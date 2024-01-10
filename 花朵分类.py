import math, re, os, random, gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import applications as tf_applications
from tensorflow.keras.optimizers.legacy import Adam, Nadam, Adamax
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D
from kaggle_datasets import KaggleDatasets
import warnings
import sys
import ctypes


def clean_memory():
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    gc.collect()


def seed_everything(seed=36):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def accelerator():
    tf.config.optimizer.set_jit(True)
    print(tf.config.optimizer.get_jit())

    tf.config.optimizer.set_experimental_options(
        {'disable_model_pruning': True, 'scoped_allocator_optimization': True, 'implementation_selector': True,
         'auto_parallel': True, 'constant_folding': True, 'shape_optimization': True, 'remapping': True,
         'arithmetic_optimization': True, 'dependency_optimization': True, 'function_optimization': True,
         'loop_optimization': True})
    print(tf.config.optimizer.get_experimental_options())


def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except ValueError:
        strategy = tf.distribute.get_strategy()
    return strategy


accelerator()
clean_memory()
seed_everything(36)
warnings.filterwarnings('ignore')

strategy = get_strategy()
AUTO = tf.data.AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE

CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
    'swin_tiny_331': dict(input_size=(331, 331), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_base_331': dict(input_size=(331, 331), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    'swin_large_331': dict(input_size=(331, 331), window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
}


class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=f'{prefix}/mlp/fc1')
        self.fc2 = Dense(out_features, name=f'{prefix}/mlp/fc2')
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size,
                   window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size,
                   W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., prefix=''):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix

        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name=f'{self.prefix}/attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',
                                                            shape=(
                                                                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False, name=f'{self.prefix}/attn/relative_position_index')
        self.built = True

    def call(self, x, mask=None):
        B_, N, C = x.get_shape().as_list()
        qkv = tf.transpose(tf.reshape(self.qkv(
            x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(
            self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
                                            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs
    keep_prob = 1.0 - drop_prob
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.prefix = prefix

        self.norm1 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm1')
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, prefix=self.prefix)
        self.drop_path = DropPath(
            drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop, prefix=self.prefix)

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name=f'{self.prefix}/attn_mask')
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=[-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = tf.reshape(
            attn_windows, shape=[-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[
                        self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,
                               name=f'{prefix}/downsample/reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = tf.keras.Sequential([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (
                                               i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path_prob=drop_path_prob[i] if isinstance(
                                               drop_path_prob, list) else drop_path_prob,
                                           norm_layer=norm_layer,
                                           prefix=f'{prefix}/blocks{i}') for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix)
        else:
            self.downsample = None

    def call(self, x):
        x = self.blocks(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__(name='patch_embed')
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, name='proj')
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = x.get_shape().as_list()
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(
            x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerModel(tf.keras.Model):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', include_top=False,
                 img_size=(224, 224), patch_size=(4, 4), in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__(name=model_name)

        self.include_top = include_top

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                      shape=(
                                                          1, num_patches, embed_dim),
                                                      initializer=tf.initializers.Zeros())

        self.pos_drop = Dropout(drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build layers
        self.basic_layers = tf.keras.Sequential([BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                  patches_resolution[1] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                    depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers{i_layer}') for i_layer in range(self.num_layers)])
        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = GlobalAveragePooling1D()
        if self.include_top:
            self.head = Dense(num_classes, name='head')
        else:
            self.head = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.basic_layers(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        if self.include_top:
            x = self.head(x)
        return x


def SwinTransformer(model_name='swin_tiny_224', num_classes=1000, include_top=True, pretrained=True, use_tpu=False, cfgs=CFGS):
    cfg = cfgs[model_name]
    net = SwinTransformerModel(
        model_name=model_name, include_top=include_top, num_classes=num_classes, img_size=cfg['input_size'], window_size=cfg[
            'window_size'], embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads']
    )
    net(tf.keras.Input(shape=(cfg['input_size'][0], cfg['input_size'][1], 3)))
    if pretrained is True:
        url = f'https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/{model_name}.tgz'
        pretrained_ckpt = tf.keras.utils.get_file(
            model_name, url, untar=True)
    else:
        pretrained_ckpt = pretrained

    if pretrained_ckpt:
        if tf.io.gfile.isdir(pretrained_ckpt):
            pretrained_ckpt = f'{pretrained_ckpt}/{model_name}.ckpt'

        if use_tpu:
            load_locally = tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost')
            net.load_weights(pretrained_ckpt, options=load_locally)
        else:
            net.load_weights(pretrained_ckpt)

    return net

IMAGE_SIZE = [224, 224]
EPOCHS = 36
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
SWIN_TYPE = 'large'

GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')
GCS_DS_PATH_EXT = KaggleDatasets().get_gcs_path('tf-flower-photo-tfrec')
GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]
GCS_PATH_SELECT_EXT = {
    192: '/tfrecords-jpeg-192x192',
    224: '/tfrecords-jpeg-224x224',
    331: '/tfrecords-jpeg-331x331',
    512: '/tfrecords-jpeg-512x512'
}
GCS_PATH_EXT = GCS_PATH_SELECT_EXT[IMAGE_SIZE[0]]
IMAGENET_FILES = tf.io.gfile.glob(GCS_DS_PATH_EXT + '/imagenet' + GCS_PATH_EXT + '/*.tfrec')
INATURELIST_FILES = tf.io.gfile.glob(GCS_DS_PATH_EXT + '/inaturalist' + GCS_PATH_EXT + '/*.tfrec')
OPENIMAGE_FILES = tf.io.gfile.glob(GCS_DS_PATH_EXT + '/openimage' + GCS_PATH_EXT + '/*.tfrec')
OXFORD_FILES = tf.io.gfile.glob(GCS_DS_PATH_EXT + '/oxford_102' + GCS_PATH_EXT + '/*.tfrec')
TENSORFLOW_FILES = tf.io.gfile.glob(GCS_DS_PATH_EXT + '/tf_flowers' + GCS_PATH_EXT + '/*.tfrec')
ADDITIONAL_TRAINING_FILENAMES = IMAGENET_FILES + INATURELIST_FILES + OPENIMAGE_FILES + OXFORD_FILES + TENSORFLOW_FILES
CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                          # 100 - 102

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition

TRAINING_FILENAMES = TRAINING_FILENAMES + ADDITIONAL_TRAINING_FILENAMES

# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)
with strategy.scope():
    def batch_to_numpy_images_and_labels(data):
        images, labels = data
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
            numpy_labels = [None for _ in enumerate(numpy_images)]
        # If no labels, only image IDs, return None for labels (this is the case for test data)
        return numpy_images, numpy_labels

    def title_from_label_and_target(label, correct_label):
        if correct_label is None:
            return CLASSES[label], True
        correct = (label == correct_label)
        return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                    CLASSES[correct_label] if not correct else ''), correct

    def display_one_flower(image, title, subplot, red=False, titlesize=16):
        plt.subplot(*subplot)
        plt.axis('off')
        plt.imshow(image)
        if len(title) > 0:
            plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
        return (subplot[0], subplot[1], subplot[2]+1)

    def display_batch_of_images(databatch, predictions=None):
        # data
        images, labels = batch_to_numpy_images_and_labels(databatch)
        if labels is None:
            labels = [None for _ in enumerate(images)]
        # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
        rows = int(math.sqrt(len(images)))
        cols = len(images)//rows
        # size and spacing
        FIGSIZE = 13.0
        SPACING = 0.1
        subplot=(rows,cols,1)
        if rows < cols:
            plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
        else:
            plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
        # display
        for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
            title = '' if label is None else CLASSES[label]
            correct = True
            if predictions is not None:
                title, correct = title_from_label_and_target(predictions[i], label)
            dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
            subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
        #layout
        plt.tight_layout()
        if label is None and predictions is None:
            plt.subplots_adjust(wspace=0, hspace=0)
        else:
            plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
        plt.show()

    def display_confusion_matrix(cmat, score, precision, recall):
        plt.figure(figsize=(15,15))
        ax = plt.gca()
        ax.matshow(cmat, cmap='Reds')
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        titlestring = ""
        if score is not None:
            titlestring += 'f1 = {:.3f} '.format(score)
        if precision is not None:
            titlestring += '\nprecision = {:.3f} '.format(precision)
        if recall is not None:
            titlestring += '\nrecall = {:.3f} '.format(recall)
        if len(titlestring) > 0:
            ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
        plt.show()

    def display_training_curves(training, validation, title, subplot):
        if subplot%10==1: # set up the subplots on the first call
            plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
            plt.tight_layout()
        ax = plt.subplot(subplot)
        ax.set_facecolor('#F8F8F8')
        ax.plot(training)
        ax.plot(validation)
        ax.set_title('model '+ title)
        ax.set_ylabel(title)
        #ax.set_ylim(0.28,1.05)
        ax.set_xlabel('epoch')
        ax.legend(['train', 'valid.'])

with strategy.scope():
    def random_erasing(img, sl=0.1, sh=0.2, rl=0.4):
        p=random.random()
        if p>=0.1 and p<=0.5:
            w, h, c = IMAGE_SIZE[0], IMAGE_SIZE[1], 3
            origin_area = tf.cast(h*w, tf.float32)
            e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)
            e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)
            e_height_h = tf.minimum(e_size_h, h)
            e_width_h = tf.minimum(e_size_h, w)
            erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)
            erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)
            erase_area = tf.zeros(shape=[erase_height, erase_width, c])
            erase_area = tf.cast(erase_area, tf.uint8)
            pad_h = h - erase_height
            pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
            pad_bottom = pad_h - pad_top
            pad_w = w - erase_width
            pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
            pad_right = pad_w - pad_left
            erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)
            erase_mask = tf.squeeze(erase_mask, axis=0)
            erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))
            return tf.cast(erased_img, img.dtype)
        else:
            return tf.cast(img, img.dtype)

with strategy.scope():
    def decode_image(image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
        image = tf.reshape(image, [*IMAGE_SIZE, 3])  # explicit size needed for TPU
        return image


    def transform(image, label):
        DIM = IMAGE_SIZE[0]
        XDIM = DIM % 2  # fix for size 331
        rot = 15. * tf.random.normal([1], dtype='float32')
        shr = 5. * tf.random.normal([1], dtype='float32')
        h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
        w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
        h_shift = 16. * tf.random.normal([1], dtype='float32')
        w_shift = 16. * tf.random.normal([1], dtype='float32')
        m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)
        x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
        y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
        z = tf.ones([DIM * DIM], dtype='int32')
        idx = tf.stack([x, y, z])
        idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
        idx2 = K.cast(idx2, dtype='int32')
        idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)
        idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
        d = tf.gather_nd(image, tf.transpose(idx3))
        return tf.reshape(d, [DIM, DIM, 3]), label


    def onehot(image, label):
        return image, tf.one_hot(label, len(CLASSES))


    def read_labeled_tfrecord(example):
        LABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        }
        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
        image = decode_image(example['image'])
        label = tf.cast(example['class'], tf.int32)
        return image, label  # returns a dataset of (image, label) pairs


    def read_unlabeled_tfrecord(example):
        UNLABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        }
        example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
        image = decode_image(example['image'])
        idnum = example['id']
        return image, idnum  # returns a dataset of image(s)


    def load_dataset(filenames, labeled=True, ordered=False):
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False  # disable order, increase speed

        dataset = tf.data.TFRecordDataset(filenames,
                                          num_parallel_reads=AUTO)  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order)  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
        # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
        return dataset


    def data_augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = random_erasing(image)
        return image, label


    def data_hflip(image, idnum):
        image = tf.image.random_flip_left_right(image)
        return image, idnum


    def get_training_dataset(do_onehot=False):
        dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        if do_onehot:
            dataset = dataset.map(onehot, num_parallel_calls=AUTO)
        dataset = dataset.repeat()  # the training dataset must repeat for several epochs
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
        return dataset


    def get_validation_dataset(ordered=False, do_onehot=False):
        dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
        if do_onehot:
            dataset = dataset.map(onehot, num_parallel_calls=AUTO)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.cache()
        dataset = dataset.prefetch(AUTO)
        return dataset


    def get_test_dataset(ordered=False, augmented=False):
        dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
        dataset = dataset.map(data_hflip, num_parallel_calls=AUTO)
        # dataset = dataset.map(transform, num_parallel_calls=AUTO)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(AUTO)
        return dataset


    def count_data_items(filenames):
        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
        return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALIDATION_STEPS = -(-NUM_VALIDATION_IMAGES // BATCH_SIZE)  # The "-(-//)" trick rounds up instead of down :-)
TEST_STEPS = -(-NUM_TEST_IMAGES // BATCH_SIZE)  # The "-(-//)" trick rounds up instead of down :-)
print(
    f'Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_VALIDATION_IMAGES} validation images, {NUM_TEST_IMAGES} unlabeled test images')

with strategy.scope():
    def get_lr_callback(plot_schedule=False):
        LR_START = 0.00001
        LR_MAX = 0.00005 * strategy.num_replicas_in_sync
        LR_MIN = 0.00001
        LR_RAMPUP_EPOCHS = 5
        LR_SUSTAIN_EPOCHS = 0
        LR_EXP_DECAY = .8

        def lrfn(epoch):
            if epoch < LR_RAMPUP_EPOCHS:
                lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
            elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
                lr = LR_MAX
            else:
                lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
            return lr

        return callbacks.LearningRateScheduler(lrfn, verbose=1)

with strategy.scope():
    def load_and_fit_model(model_name, print_summary=False):
        if model_name == 'efficientnet':
            base_model = efn.EfficientNetB6(weights='noisy-student',include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        elif model_name == 'densenet':
            base_model = tf.keras.applications.DenseNet201(weights='imagenet',include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        elif model_name == 'swin':
            model = tf.keras.Sequential([SwinTransformer(f'swin_{SWIN_TYPE}_{IMAGE_SIZE[0]}', include_top=False, pretrained=True)])
        else: pass
        model.trainable = True

        model = tf.keras.Sequential([
            model,
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.GlobalAveragePooling2D(),
            L.Dense(len(CLASSES), activation='softmax')
        ])

        model.compile(
            optimizer=Adam(beta_1=0.5),
            loss = 'categorical_crossentropy',
            metrics=[F1Score(len(CLASSES), average='macro')],
        )
        os.makedirs('checkpoints', exist_ok=True)
        lr_callback = get_lr_callback()
        chk_callback = callbacks.ModelCheckpoint(f'checkpoints/{model_name}_best.h5',save_weights_only=True, monitor='val_f1_score',mode='max', save_best_only=True, verbose=1)
        ton = tf.keras.callbacks.TerminateOnNaN()
        _ = model.fit(get_training_dataset(do_onehot=True),
                      steps_per_epoch=STEPS_PER_EPOCH,
                      epochs=EPOCHS,
                      validation_data=get_validation_dataset(do_onehot=True),
                      validation_steps=VALIDATION_STEPS,
                      callbacks=[lr_callback, chk_callback, ton],
                      verbose=1)
        model.load_weights(f'checkpoints/{model_name}_best.h5')
        return model

with strategy.scope():
    model = load_and_fit_model('swin')

# def find_best_alpha(valid_dataset, model_lst):
#     images_ds = valid_dataset.map(lambda image, label: image)
#     labels_ds = valid_dataset.map(lambda image, label: label).unbatch()
#     y_true = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch
#     p = []
#     for model in model_lst:
#         p.append(model.predict(images_ds))
#     scores = []
#     for alpha in np.linspace(0,1,100):
#         preds = np.argmax(alpha*p[0]+(1-alpha)*p[1], axis=-1)
#         scores.append(f1_score(y_true, preds, labels=range(len(CLASSES)), average='macro'))
#     best_alpha = np.argmax(scores)/100
#     return best_alpha

# valid_ds = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.
# alpha = find_best_alpha(valid_ds, models)

# def predict_ensemble(dataset, model_lst, alpha, steps):
#     images_ds = dataset.map(lambda image, idnum: image)
#     probs = []
#     for model in model_lst:
#         p = model.predict(images_ds,verbose=0, steps=steps)
#         probs.append(p)
#     preds = np.argmax(alpha*probs[0] + (1-alpha)*probs[1], axis=-1)
#     return preds
# cm_predictions = predict_ensemble(valid_ds, models, alpha, steps=VALIDATION_STEPS)
# labels_ds = valid_ds.map(lambda image, label: label).unbatch()
# cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch
# cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
# score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
# precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
# recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
# display_confusion_matrix(cmat, score, precision, recall)
# test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.
# predictions = predict_ensemble(test_ds, models, alpha, steps=TEST_STEPS)
# test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
# test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
# sub_df = pd.DataFrame({'id': test_ids, 'label': predictions})
# sub_df.to_csv('submission.csv', index=False)

def predict(dataset, model):
    print('Calculating predictions...')
    images_ds = dataset.map(lambda image, idnum: image)
    preds = model.predict(images_ds,verbose=0)
    preds = np.argmax(preds, axis=1)
    return preds

# valid_ds = get_validation_dataset(ordered=True)
# cm_predictions = predict(valid_ds, model)

# labels_ds = valid_ds.map(lambda image, label: label).unbatch()
# cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

# cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
# score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
# precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
# recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
# #cmat = (cmat.T / cmat.sum(axis=1)).T # normalized
# display_confusion_matrix(cmat, score, precision, recall)
# test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.
test_ds = get_test_dataset(ordered=True)
predictions = predict(test_ds, model)
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
sub_df = pd.DataFrame({'id': test_ids, 'label': predictions})
sub_df.to_csv('submission.csv', index=False)