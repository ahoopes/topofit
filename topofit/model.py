import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from . import ico


def network_config(name=None):
    config = {
        'unet_features': [
            [16, 32, 64, 64],       # encoder features
            [64, 64, 64, 64, 64]],  # decoder features
        'blocks': [
            {'order': 1, 'train_iters': 1, 'infer_iters': 1, 'unet_levels': 1, 'convs_per_unet_level': 2},
            {'order': 2, 'train_iters': 1, 'infer_iters': 1, 'unet_levels': 2, 'convs_per_unet_level': 2},
            {'order': 3, 'train_iters': 1, 'infer_iters': 1, 'unet_levels': 3, 'convs_per_unet_level': 2},
            {'order': 4, 'train_iters': 1, 'infer_iters': 1, 'unet_levels': 3, 'convs_per_unet_level': 2},
            {'order': 5, 'train_iters': 1, 'infer_iters': 1, 'unet_levels': 3, 'convs_per_unet_level': 2},
            {'order': 6, 'train_iters': 1, 'infer_iters': 1, 'unet_levels': 3, 'convs_per_unet_level': 2},
            {'order': 6, 'train_iters': 1, 'infer_iters': 1, 'unet_levels': 3, 'convs_per_unet_level': 2},
            {'order': 7, 'train_iters': 1, 'infer_iters': 1, 'unet_levels': 3, 'convs_per_unet_level': 2},
        ],
        'low_res_skip_blocks': [7],
        'include_vertex_properties': True,
        'scale_delta_prediction': 10.,
    }
    return config


class ImageConv(nn.Module):

    def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leakyrelu'):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ImageUnet(nn.Module):

    def __init__(self,
                 nb_features,
                 nb_levels=None,
                 infeats=1,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 ndims=3):

        super().__init__()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ImageConv(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ImageConv(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ImageConv(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            x = self.upsampling[level](x)
            x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class DynamicGraphConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, topology, bias=True, activation='leaky'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edges_a = topology['adj_edges_a']
        self.edges_b = topology['adj_edges_b']
        self.weights = topology['adj_weights']
        self.size = topology['size']

        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.3)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'unknown activation `{activation}`.')

        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding='valid',
            bias=bias)

    def forward(self, input_features):

        vertices = input_features[self.edges_a]
        neighbors = input_features[self.edges_b]
        concat_features = torch.cat([vertices, neighbors - vertices], -1)

        concat_features = torch.unsqueeze(torch.swapaxes(concat_features, -2, -1), 0)
        edge_features = self.conv1d(concat_features)
        edge_features = torch.squeeze(edge_features, 0)

        edge_features = torch.swapaxes(edge_features, -2, -1)
        edge_features_weighted = edge_features * self.weights
        indices = self.edges_a.unsqueeze(-1).expand(-1, self.out_channels)
        features = torch.zeros((self.size, self.out_channels), dtype=torch.float32, device=utils.get_device()).scatter_add(-2, indices, edge_features_weighted)

        # activation
        if self.activation is not None:
            features = self.activation(features)

        return features


class DynamicGraphUnet(torch.nn.Module):

    def __init__(self,
                 order,
                 mesh_collection,
                 nb_input_features,
                 input_pial_features=False,
                 start_pial=False,
                 nb_features=64,
                 train_iters=1,
                 infer_iters=1,
                 unet_levels=1,
                 convs_per_unet_level=4):
        super().__init__()

        self.order = order
        self.mesh_collection = mesh_collection
        self.train_iters = train_iters
        self.infer_iters = infer_iters
        self.input_pial_features = input_pial_features
        self.start_pial = start_pial

        # configure encoder (down-sampling path)
        curr_level = order
        prev_nf = nb_input_features
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(unet_levels):
            convs = nn.ModuleList()
            for conv in range(convs_per_unet_level):
                nf = nb_features
                convs.append(DynamicGraphConv(prev_nf, nf, self.mesh_collection[curr_level]))
                prev_nf = nf
            self.encoder.append(convs)
            if level < unet_levels - 1:
                encoder_nfs.append(prev_nf)
                curr_level -= 1

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(unet_levels - 1):
            curr_level += 1
            prev_nf += encoder_nfs[level]
            convs = nn.ModuleList()
            for conv in range(convs_per_unet_level):
                nf = nb_features
                convs.append(DynamicGraphConv(prev_nf, nf, self.mesh_collection[curr_level]))
                prev_nf = nf
            self.decoder.append(convs)

        # final conv to estimate mesh deformation
        final_nf = 6 if (self.start_pial or self.input_pial_features) else 3
        self.finalconv = DynamicGraphConv(prev_nf, final_nf, self.mesh_collection[curr_level], activation=None)

    def forward(self, x):

        # encoder forward pass
        curr_level = self.order
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            if level < len(self.encoder) - 1:
                x_history.append(x)
                x = utils.pool(x, self.mesh_collection[curr_level])
                curr_level -= 1

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            curr_level += 1
            x = utils.unpool(x, self.mesh_collection[curr_level])
            x = torch.cat([x, x_history.pop()], dim=-1)
            for conv in convs:
                x = conv(x)

        x = self.finalconv(x)
        return x


class SurfNet(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.config = network_config()

        self.image_unet = ImageUnet(self.config['unet_features'])
        self.include_vertex_properties = self.config['include_vertex_properties']
        self.scale_delta_prediction = self.config['scale_delta_prediction']

        config_blocks = self.config['blocks']
        max_order = np.max([b['order'] for b in config_blocks])

        self.mesh_collection = {o: ico.load_topology(o) for o in range(1, max_order + 1)}

        nb_input_features = self.image_unet.final_nf
        if self.include_vertex_properties:
            nb_input_features += 6

        self.blocks = nn.ModuleList()
        for n, block in enumerate(config_blocks):
            block['mesh_collection'] = self.mesh_collection
            block['nb_input_features'] = nb_input_features
            self.blocks.append(DynamicGraphUnet(**block))

        self.low_res_training = False
        self.current_neighborhood_target = None
        self.neighborhood = None

    def initialize_weights(self):

        def initialize(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv3d, torch.nn.Linear)):
                fan = torch.nn.init._calculate_correct_fan(m.weight, mode='fan_in')
                std = np.sqrt(2) / np.sqrt(fan)
                torch.nn.init.trunc_normal_(m.weight, std=std, a=(-2 * std), b=(2 * std))
                torch.nn.init.zeros_(m.bias)
        self.apply(initialize)

        for block in self.blocks:
            torch.nn.init.normal_(block.finalconv.conv1d.weight, mean=0.0, std=1e-4)

    def forward(self, image, coords):

        # 
        image_shape = list(image.shape)
        image_shape_tensor = torch.Tensor(image_shape).to(utils.get_device())

        # 
        if len(image_shape) != 3:
            raise ValueError(f'expected 3-dimensional image input, but got ndim {len(image_shape)}')

        if len(coords.shape) != 2:
            raise ValueError(f'expected 2-dimensional input coordinate array, but got ndim {len(coords.shape)}')

        # predict image-based features
        image_features = self.image_unet(image.view(1, 1, *image_shape))

        # 
        results = {'image_features': image_features}

        # squeeze batch dimension
        image_features = image_features[0, ...]

        previous_order = None

        for blockno, block in enumerate(self.blocks):

            # check whether we should skip this resolution during inital low-res training
            if self.low_res_training and block.order in self.config['low_res_skip_blocks']:
                break

            topology = self.mesh_collection[block.order]

            # upsample to the next mesh resolution (if necessary)
            if previous_order is not None and previous_order < block.order:
                indices, weights = topology['upsampler']
                coords = torch.sum(coords[indices] * weights, -2)

            # get number of block iterations (usually 1 when training)
            iters = block.train_iters if self.training else block.infer_iters

            for it in range(iters):

                # input features
                input_features = []

                # add vertex properties
                if self.include_vertex_properties:
                    scaled_coords = coords / torch.max(image_shape_tensor)
                    normals = utils.compute_normals(coords, topology['faces'])
                    input_features.extend([scaled_coords, normals])

                # sample image-based features at current white mesh position
                input_features.append(utils.point_sample(coords, image_features, image_shape_tensor))
                
                # concatenate input features
                sampled_features = torch.cat(input_features, axis=-1) if len(input_features) > 1 else input_features[0]

                # predict and apply the deformation in mesh space
                deformation = block(sampled_features)

                if self.scale_delta_prediction is not None:
                    deformation = deformation * self.scale_delta_prediction

                coords = coords + deformation
                previous_order = block.order

        results['pred_vertices'] = coords
        return (results, topology)

    def guided_chamfer_loss(self, y_true, y_pred):

        order = 6 if self.low_res_training else 7
        if self.current_neighborhood_target != order:
            if self.neighborhood is not None:
                del self.neighborhood
            self.neighborhood = torch.from_numpy(ico.neighborhood(order)).to(utils.get_device())
            self.current_neighborhood_target = order

        y_pred_gathered = y_pred[self.neighborhood]                               # [vert, neighbors, 3]
        y_true_ext = torch.unsqueeze(y_true, dim=-2)                              # [vert, 1, 3]
        a_sqr_distances = torch.sum((y_true_ext - y_pred_gathered) ** 2, dim=-1)  # [vert, neighbors]
        a_min_dist, _ = torch.min(torch.sqrt(a_sqr_distances), dim=-1)            # [neighbors]

        y_true_gathered = y_true[self.neighborhood]                               # [vert, neighbors, 3]
        y_pred_ext = torch.unsqueeze(y_pred, dim=-2)                              # [vert, 1, 3]
        b_sqr_distances = torch.sum((y_pred_ext - y_true_gathered) ** 2, dim=-1)  # [vert, neighbors]
        b_min_dist, _ = torch.min(torch.sqrt(b_sqr_distances), dim=-1)            # [neighbors]

        loss = torch.mean(torch.cat([a_min_dist, b_min_dist]))
        return loss

    def hinge_spring_loss(self, y_pred, topology):
        face_vertices = y_pred[topology['faces']]                                        # [faces, 3, 3]
        face_norms = utils.face_normals(face_vertices, clockwise=False, normalize=True)  # [faces, 3]
        edge_face_normals = face_norms[topology['edge_faces']]                           # [edges, 2, 3]

        norm_a = edge_face_normals[:, 0, :]                                              # [edges, 3]
        norm_b = edge_face_normals[:, 1, :]                                              # [edges, 3]

        dot = torch.sum(torch.multiply(norm_a, norm_b), dim=-1)                          # [edges]
        error = (1 - dot) ** 2

        loss = torch.mean(error, dim=-1)                                                 # [1]
        return loss
