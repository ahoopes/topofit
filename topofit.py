import os
import time
import numpy as np
import neurite as ne
import voxelmorph as vxm
import freesurfer as fs
import ctx
import glob

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as ki

import tensorflow_graphics as tfg
import tensorflow_graphics.geometry.representation.mesh.normals as tfg_mesh_normals
import tensorflow_graphics.geometry.representation.mesh.utils as tfg_mesh_utils
import tensorflow_graphics.nn.layer.graph_convolution as tfg_graph_conv
import tensorflow_graphics.geometry.convolution as tfg_geometry_conv
from tensorflow_graphics.util import safe_ops


# configure base paths for template files and training data
basedir = '/autofs/vast/topofit'
avgfile = os.path.join(basedir, 'template', 'lh.white.ctx.average.6')
icofile = os.path.join(basedir, 'template', 'ico.npz')
baryfile = os.path.join(basedir, 'template', 'bary.npz')
mappingfile = os.path.join(basedir, 'template', 'mapping.npz')
edgefacesfile = os.path.join(basedir, 'template', 'edgefaces.npz')
datadir = os.path.join(basedir, 'data')
scriptdir = os.path.join(basedir, 'scripts')
neighborfile = {
    6: os.path.join(basedir, 'template', 'neighborhoods-6.npz'),
    7: os.path.join(basedir, 'template', 'neighborhoods-7.npz'),
}


# lookup ico-resolution level from vertex count 
nvert_to_ico = {
    12: 0,
    42: 1,
    162: 2,
    642: 3,
    2562: 4,
    10242: 5,
    40962: 6,
    163842: 7,
}


def read_subject_list(filename):
    """
    Read a subject txt file into a list
    """
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines



class Nothing:
    """
    Empty class for building graph blocks
    """

    def __init__(self):
        pass


class EverythingCallback(tf.keras.callbacks.Callback):
    """
    Callback to handle logging, decay rate, and validation
    """

    def __init__(
            self,
            trainer,
            logfile,
            validation_freq=25,
            decay_metric='val-white-dist-masked',
            decay_factor=0.5,
            decay_patience=100,
            min_lr=1e-6,
            mode='a'):
        super().__init__()
        self.trainer = trainer
        self.logfile = logfile
        self.validation_freq = validation_freq
        self.decay_factor = decay_factor
        self.decay_min = None
        self.decay_min_epoch = None
        self.decay_metric = decay_metric
        self.decay_patience = decay_patience
        self.decay_thresh = 1e-3
        self.min_lr = min_lr
        self.keys = None

    def on_epoch_end(self, epoch, logs=None):

        # get logs
        logs = logs or {}

        # run validation
        validate_epoch = (self.keys is None) or ((epoch + 1) % self.validation_freq == 0)
        if validate_epoch:
            logs.update(self.trainer.validate(dirname='%s/%06d' % (self.trainer.valdir, epoch + 1)))

        # learning rate schedule
        if validate_epoch:
            curr_decay_value = logs[self.decay_metric]
            if (self.decay_min is None) or (curr_decay_value < (self.decay_min - self.decay_thresh)):
                self.decay_min = curr_decay_value
                self.decay_min_epoch = epoch

            from tensorflow.keras import backend
            lr = float(backend.get_value(self.model.optimizer.lr))
            if ((epoch - self.decay_min_epoch) >= self.decay_patience) and (lr > self.min_lr):
                backend.set_value(self.model.optimizer.lr, backend.get_value(lr * self.decay_factor))
            logs['lr'] = lr

        # write header
        if self.keys is None:
            self.keys = list(logs.keys())
            with open(self.logfile, 'w') as f:
                f.write(' '.join(['epoch'] + self.keys) + '\n')

        # write metrics
        if validate_epoch:
            with open(self.logfile, 'a') as f:
                f.write(str(epoch + 1) + ' ')
                for key in self.keys:
                    value = logs.get(key, 'x')
                    if hasattr(value, 'numpy'):
                        value = value.numpy()
                    f.write(str(value) + ' ')
                f.write('\n')


class BarcentricUpsample(kl.Layer):
    """
    Mesh upsampling layer
    """
        
    def __init__(self, sources, bary_coords, **kwargs):
        self.sources = sources
        self.bary_coords = bary_coords[None, ..., None]
        super().__init__(**kwargs)

    def call(self, x):
        tri_coords = tf.gather(x, self.sources, axis=-2, batch_dims=0)
        interpolated = tf.reduce_sum(self.bary_coords * tri_coords, axis=-2)
        return interpolated


class GraphPooling(kl.Layer):
    """
    Graph pooling layer
    """

    def __init__(self, mapping, algorithm='max', **kwargs):
        self.mapping = mapping
        self.algorithm = algorithm
        super().__init__(**kwargs)

    def call(self, inputs):
        # TODO what are sizes?
        return tfg_geometry_conv.graph_pooling.pool(inputs, self.mapping, sizes=None, algorithm=self.algorithm)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(self.mapping.dense_shape[1]), input_shape[-1])


class GraphUnpooling(kl.Layer):
    """
    Graph unpooling layer
    """

    def __init__(self, mapping, **kwargs):
        self.mapping = mapping
        super().__init__(**kwargs)

    def call(self, inputs):
        # TODO what are sizes?
        return tfg_geometry_conv.graph_pooling.unpool(inputs, self.mapping, sizes=None)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.mapping.dense_shape[2], input_shape[-1])


class SampleImage(kl.Layer):
    """
    Image feature sampling layer
    """

    def call(self, x):
        return vxm.tf.utils.value_at_location(x, force_post_absolute_val=False)


class ComputeNormals(kl.Layer):
    """
    Compute normals on a mesh
    """

    def __init__(self, faces, **kwargs):
        self.faces = faces[None]
        super().__init__(**kwargs)

    def call(self, x):
        x = x[0][None]  # this is a bad hack
        return tfg_mesh_normals.vertex_normals(x, self.faces, clockwise=False)  # [B, V, 3]


class SurfNet:

    def __init__(self, outdir=None, streamdir=None, loss='dist-1000', validation_dir='validate-real', block_type='unet',
                 target=7, start_ico=1, start_read_epoch=0, regularize=None, regularize_max=None):

        # template mesh information        
        self.target = target
        self.start_ico = start_ico
        self.ico = np.load(ctx.icofile, allow_pickle=True)
        self.bary = np.load(ctx.baryfile, allow_pickle=True)
        self.neighborhoods = np.load(ctx.neighborfile[self.target], allow_pickle=True)
        self.nvertices = len(self.ico[f'ico-{self.target}-vertices'])
        self.faces = self.ico[f'ico-{self.target}-faces']
        mapping_npz = np.load(ctx.mappingfile, allow_pickle=True)
        self.mapping = mapping_npz[f'mapping-7-to-{self.target}'] if self.target != 7 else ...
        self.source_mapping = mapping_npz[f'mapping-6-to-{self.start_ico}']
        self.nvertices_source = len(self.ico[f'ico-{self.start_ico}-vertices'])

        # topology regularization
        self.regularize = None if regularize == 0 else regularize
        if self.regularize is not None:
            self.edge_faces = np.load(ctx.edgefacesfile)[f'ico-{self.target}']

        # data information
        self.shape = (96, 144, 208)
        self.scale_factor = np.max(self.shape)
        self.initial_epoch = 0

        # settings
        self.start_read_epoch = start_read_epoch
        self.verbose = False
        self.max_stream_wait = 600
        self.keep_stream_files = False
        self.save_freq = 25
        self.validation_freq = 25

        self.include_vertex_properties = False
        self.scale_delta_prediction = 10.0
        self.activate_delta = False
        self.use_feature_steered = False

        # file management
        if outdir is not None:
            self.validation_dir = validation_dir
            self.outdir = os.path.join(ctx.basedir, 'results', outdir)
            self.valdir = os.path.join(self.outdir, 'validation')
            self.modeldir = os.path.join(self.outdir, 'models')
            if streamdir is None:
                streamdir = os.path.join(self.outdir, 'streaming')
            self.streamdir = streamdir
            os.makedirs(self.outdir, exist_ok=True)
            os.makedirs(self.valdir, exist_ok=True)
            os.makedirs(self.modeldir, exist_ok=True)
            os.makedirs(self.streamdir, exist_ok=True)
            datadir = os.path.join(ctx.datadir, 'real-ico7-left')
            self.validation_files = [os.path.join(datadir, s) + '.npz' for s in ctx.read_subject_list(os.path.join(datadir, 'validation-nouk.txt'))]
        self.validation_model = None

        if loss.startswith('dist-'):
            self.neighborhood = self.neighborhoods[loss.split('-')[-1]]
            loss = self.distance_loss
        elif loss == 'mse':
            pass
        else:
            raise ValueError(f'Unknown loss type: {loss}')

        self.loss = [loss]
        self.loss_weights = [1.0]

        self.regularize_max = regularize_max
        if self.regularize is not None:
            self.loss.append(self.hinge_spring_loss)
            self.loss_weights.append(self.regularize)

        self.num_weight_matrices = 8
        self.graph_channels = 64
        self.unet_features = [
            [16, 32, 32, 64],
            [64, 64, 64, 64, 64]]

        if block_type == 'unet':
            blocks = [
                [0, 1, 3],  # 1
                [1, 2, 2],  # 2
                [1, 3, 2],  # 3
                [1, 3, 2],  # 4
                [1, 3, 2],  # 5
                [1, 3, 2],  # 6
                [0, 3, 2],  # 6
                [1, 3, 2],  # 7
            ]
        elif block_type == 'snet':
            blocks = [
                [0, 1, 10],  # 1
                [1, 1, 10],  # 2
                [1, 1, 10],  # 3
                [1, 1, 10],  # 4
                [1, 1, 10],  # 5
                [1, 1, 10],  # 6
                [0, 1, 10],  # 6
                [1, 1, 10],  # 7
            ]
        elif block_type == 'snet-r':
            blocks = [
                [0, 1, 10],  # 1
                [1, 1, 10],  # 2
                [1, 1, 10],  # 3
                [1, 1, 10],  # 4
                [1, 1, 10],  # 5
                [1, 1, 10],  # 6
                [0, 3, 2],  # 6
                [1, 3, 2],  # 7
            ]
        else:
            raise ValueError(f'Unknown block type {block_type}')

        # adjust upsampling for starting resolution
        blocks = np.array(blocks)
        starting_res = self.start_ico
        if starting_res > 1:
            blocks[:starting_res, 0] = 0

        self.resolution_blocks = []
        for block in blocks:
            p = Nothing()
            p.upsample = bool(block[0])
            p.levels = block[1]
            p.convs_per_level = block[2]
            if p.upsample:
                starting_res += 1
            if starting_res > self.target:
                break
            p.ico = starting_res
            p.faces = self.ico[f'ico-{starting_res}-faces']
            p.sources = self.bary[f'ico-{starting_res}-sources']
            p.bary = self.bary[f'ico-{starting_res}-bary']
            self.resolution_blocks.append(p)

        self.adjacency_maps = {n: self.build_adjacency(n) for n in range(1, self.target + 1)}
        self.pool_maps = {n: self.build_pool_map(n) for n in range(1, self.target + 1)}
        self.model = None

    def build_pool_map(self, ico_high):
        """
        Build mesh upsample mapping
        """
        ico_low = ico_high - 1
        indices = self.ico[f'mapping-{ico_low}-to-{ico_high}-indices']
        values = self.ico[f'mapping-{ico_low}-to-{ico_high}-values']
        shape = self.ico[f'mapping-{ico_low}-to-{ico_high}-shape']
        max_k = self.ico[f'mapping-{ico_low}-to-{ico_high}-max-k']
        mapping = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        mapping = tf.sparse.reshape(mapping, (1, *shape))
        return mapping

    def build_adjacency(self, ico):
        """
        Build mesh adjacency matrices
        """
        indices = self.ico[f'ico-{ico}-adjacency-indices']
        values = self.ico[f'ico-{ico}-adjacency-values']
        shape = self.ico[f'ico-{ico}-adjacency-shape']
        adjacency = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        adjacency = tf.sparse.reshape(adjacency, (1, *shape))
        return adjacency

    def build_model(self, summary=True):

        # model name
        name = 'ctx'

        # vertex inputs 
        source_vertices = tf.keras.Input(shape=(self.nvertices_source, 3), name='%s_source_points' % name)

        # build unet
        unet = vxm.networks.Unet(inshape=(*self.shape, 1), nb_features=self.unet_features, name=name)
        input_image = unet.input
        image_features = unet.output

        # vertex output at each graph-block
        self.vertex_checkpoints = [source_vertices]

        # resolution loop
        vertices = source_vertices
        for n, p in enumerate(self.resolution_blocks):

            prefix = f'{name}_res_{n}'

            # --- 1. UPSAMPLE MESH ---

            if p.upsample:
                vertices = BarcentricUpsample(p.sources, p.bary, name=f'{prefix}_upsample')(vertices)

            # --- 2. SAMPLE FEATURES FROM THE UNET ---

            # do the image sampling
            sampled_features = SampleImage(name=f'{prefix}_image_sampling')([image_features, vertices])
            
            # concatenate vertex information with the features
            if self.include_vertex_properties:
                scaled_vertices = kl.Lambda(lambda x: x / self.scale_factor, name=f'{prefix}_vertex_normed')(vertices)
                normals = ComputeNormals(p.faces, name=f'{prefix}_vertex_normals')(vertices)
                sampled_features = kl.Concatenate(axis=-1, name=f'{prefix}_merged_channels')([scaled_vertices, normals, sampled_features])

            # --- 3. CONFIGURE THE GRAPH-NET BLOCK ---

            y = self.build_graph_block(inputs=sampled_features, p=p, prefix=prefix)
            delta = self.graph_conv(inputs=y, channels=3, adjacency=self.adjacency_maps[p.ico],
                        initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-4), name=f'{prefix}_delta')
            if self.activate_delta:
                delta = kl.Activation('tanh', name=f'{prefix}_delta_activation')(delta)

            if self.scale_delta_prediction is not None:
                delta = kl.Lambda(lambda x: x * self.scale_delta_prediction, name=f'{prefix}_delta_scaling')(delta)

            vertices = kl.Add(name=f'{prefix}_vertices')([vertices, delta])
            self.vertex_checkpoints.append(vertices)

        outputs = [vertices]

        if self.regularize is not None:
            outputs.append(vertices)

        self.model = tf.keras.Model(inputs=[input_image, source_vertices], outputs=outputs, name=name)
        if summary:
            self.model.summary(line_length=150)

    def build_graph_block(self, inputs, p, prefix):
        """
        Convolutional block operated on the mesh graph.
        """
        y = inputs
        curr_ico = p.ico
        skip_connections = []
        for level_id in range(p.levels):
            for conv_id in range(p.convs_per_level):
                layer_name = f'{prefix}_down_arm_{level_id}_conv_{conv_id}'
                y = self.graph_conv(inputs=y, channels=self.graph_channels, adjacency=self.adjacency_maps[curr_ico], name=layer_name)
                y = kl.LeakyReLU(name=f'{layer_name}_activation')(y)

            if level_id < p.levels - 1:
                skip_connections.append(y)
                y = GraphPooling(self.pool_maps[curr_ico], name=f'{prefix}_down_arm_{level_id}_pooling')(y)
                curr_ico -= 1

        for level_id in range(p.levels - 1):
            curr_ico += 1
            y = GraphUnpooling(self.pool_maps[curr_ico], name=f'{prefix}_up_arm_{level_id}_unpooling')(y)
            y = kl.Concatenate(axis=-1, name=f'{prefix}_up_arm_{level_id}_skip_connection')([y, skip_connections.pop()])

            for conv_id in range(p.convs_per_level):
                layer_name = f'{prefix}_up_arm_{level_id}_conv_{conv_id}'
                y = self.graph_conv(inputs=y, channels=self.graph_channels, adjacency=self.adjacency_maps[curr_ico], name=layer_name)
                y = kl.LeakyReLU(name=f'{layer_name}_activation')(y)

        return y

    def graph_conv(self, inputs, channels, adjacency, name=None, initializer=None):
        """
        Convolutional on the mesh graph.
        """
        if self.use_feature_steered:
            y = tfg_graph_conv.FeatureSteeredConvolutionKerasLayer(
                    num_weight_matrices=self.num_weight_matrices,
                    num_output_channels=channels,
                    initializer=initializer,
                    name=name)([inputs, adjacency])
        else:
            if initializer is None:
                initializer = 'he_normal'
            y = tfg_graph_conv.DynamicGraphConvolutionKerasLayer(
                    num_output_channels=channels,
                    reduction='weighted',
                    kernel_initializer=initializer,
                    name=name)([inputs, adjacency])
        return y

    def distance_loss(self, y_true, y_pred):
        """
        Compute distance predicted and target meshes
        """
        y_pred_gathered = tf.gather(y_pred, self.neighborhood, axis=1)                 # [batch, vert, neighbors, 3]
        y_true_ext = tf.expand_dims(y_true, axis=-2)                                   # [batch, vert, 1, 3]
        a_sqr_distances = tf.reduce_sum((y_true_ext - y_pred_gathered) ** 2, axis=-1)  # [batch, vert, neighbors]
        a_min_dist = tf.reduce_min(tf.sqrt(a_sqr_distances), axis=-1)                  # [batch, neighbors]

        y_true_gathered = tf.gather(y_true, self.neighborhood, axis=1)                 # [batch, vert, neighbors, 3]
        y_pred_ext = tf.expand_dims(y_pred, axis=-2)                                   # [batch, vert, 1, 3]
        b_sqr_distances = tf.reduce_sum((y_pred_ext - y_true_gathered) ** 2, axis=-1)  # [batch, vert, neighbors]
        b_min_dist = tf.reduce_min(tf.sqrt(b_sqr_distances), axis=-1)                  # [batch, neighbors]

        loss = tf.reduce_mean(tf.concat([a_min_dist, b_min_dist], axis=-1), axis=-1)   # [batch]
        return loss

    def hinge_spring_loss(self, _, y_pred):
        """
        Compute hinge spring force on predicted mesh
        """
        face_vertices = tfg_mesh_normals.gather_faces(y_pred, self.faces[np.newaxis])                 # [batch, faces, 3, 3]
        face_normals = tfg_mesh_normals.face_normals(face_vertices, clockwise=False, normalize=True)  # [batch, faces, 3]
        edge_face_normals = tf.gather(face_normals, self.edge_faces, axis=1)                          # [batch, edges, 2, 3]
        norm_a = edge_face_normals[:, :, 0, :]                                                        # [batch, edges, 3]
        norm_b = edge_face_normals[:, :, 1, :]                                                        # [batch, edges, 3]
        dot = tf.reduce_sum(tf.multiply(norm_a, norm_b), axis=-1)                                     # [batch, edges]
        error = (1 - dot) ** 2
        if self.regularize_max is not None:
            error, _ = tf.math.top_k(error, self.regularize_max)
        loss = tf.reduce_mean(error, axis=-1)                                                         # [batch]
        return loss

    def load(self, epoch):
        """
        Load model training epoch
        """
        model_file = os.path.join(self.outdir, 'models', '%04d.h5' % epoch)
        print('loading model file:', model_file)
        self.model.load_weights(model_file, by_name=True)
        self.initial_epoch = epoch

    def load_from_file(self, model_file):
        """
        Load from h5
        """
        self.model.load_weights(model_file)

    def train(self, epochs=1000, lr=1e-4, decay=True):
        """
        Run full training loop
        """
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=self.loss, loss_weights=self.loss_weights)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(os.path.join(self.outdir, 'models', '{epoch:04d}.h5'), period=self.save_freq, save_weights_only=True),
            EverythingCallback(trainer=self, logfile=os.path.join(self.outdir, 'history.txt')),
        ]
        self.model.fit(
            self.build_generator(),
            epochs=epochs,
            initial_epoch=self.initial_epoch,
            steps_per_epoch=100,
            callbacks=callbacks,
            verbose=1 if self.verbose else 2)

    def parse_data(self, data, mask=False):
        """
        Parse input from npz file
        """
        image = data['image'][np.newaxis, ..., np.newaxis]
        source = data['white-source'][self.source_mapping]
        white = data['white'][self.mapping]
        source = source[np.newaxis]
        white = white[np.newaxis]
        outputs = [image, source, white]
        if mask:
            return (outputs, data['mask'][self.mapping])
        else:
            return outputs

    def build_generator(self, batch_size=1, start_epoch=0):
        """
        Configure training data file stream
        """
        sr = ctx.stream.StreamReader(self.streamdir, max_wait=self.max_stream_wait, keep=self.keep_stream_files, start=(self.start_read_epoch * 100 + 1))
        zeros = np.zeros((batch_size, 10, 3), dtype='float32')
        if batch_size != 1:
            raise NotImplementedError('cant use batch > 1 yet')
        while True:
            data_group = self.parse_data(sr.next())
            inputs = data_group[:2]
            outputs = data_group[2:]
            if self.regularize is not None:
                outputs.append(zeros)
            yield (inputs, outputs)

    def validate(self, dirname='validation'):
        """
        Run evaluation on validation subjects
        """
        os.makedirs(dirname, exist_ok=True)

        metrics = {
            'val-white-dist': [],
            'val-white-dist-masked': [],
            'val-white-intersections': [],
        }

        refvol = fs.Volume(np.zeros(self.shape), affine=fs.transform.LIA())
        geom = refvol.geometry()
        affine = refvol.vox2surf()

        for file in self.validation_files:
        
            # get data
            data_group, mask = self.parse_data(np.load(file), mask=True)
            image, source = data_group[:2]
            white_true = data_group[2].squeeze()
            arrays = {}

            # predict
            start = time.time()
            pred = self.model.predict([image, source])
            if self.regularize is not None:
                pred = pred[0]
            white_pred = pred.squeeze()

            print('validation time: %.4f' % (time.time() - start))

            # white surface metrics
            white_pred_surf = fs.Surface(affine.transform(white_pred), self.ico[f'ico-{self.target}-faces'], geom=geom)
            white_true_surf = fs.Surface(affine.transform(white_true), self.ico[f'ico-{self.target}-faces'], geom=geom)
            ab = fs.surface.distance(white_pred_surf, white_true_surf).data
            ba = fs.surface.distance(white_true_surf, white_pred_surf).data
            metrics['val-white-dist'].append(np.concatenate([ab, ba]).mean())
            metrics['val-white-dist-masked'].append(np.concatenate([ab[mask], ba[mask]]).mean())
            metrics['val-white-intersections'].append(white_pred_surf.count_intersections())
            arrays['white_pred'] = white_pred
            
            np.savez_compressed(os.path.join(dirname, os.path.basename(file)), **arrays)
        print('saved validation results to', dirname)

        for key, value in metrics.items():
            metrics[key] = np.mean(metrics[key])
        return metrics
