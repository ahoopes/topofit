import os
import numpy as np
import torch
import surfa as sf

from . import utils


loaded_mesh_data = {}


def get_ico_data(key):
    """
    Retrieve icosphere topology information from npz file
    """
    npd = loaded_mesh_data.get('ico')
    if npd is None:
        npd = np.load(os.path.join(os.path.dirname(__file__), 'ico.npz'))
        loaded_mesh_data['ico'] = npd
    return npd[key]


def get_mapping(source, target):
    """
    Retrieve icosphere topology mapping from npz file. These map vertices
    across different icosphere orders
    """
    npd = loaded_mesh_data.get('mapping')
    if npd is None:
        npd = np.load(os.path.join(os.path.dirname(__file__), 'mapping.npz'))
        loaded_mesh_data['mapping'] = npd
    return npd[f'mapping-{source}-to-{target}']


def get_initial_template(hemi):
    """
    Retrieve the initial template mesh from file
    """
    surf = loaded_mesh_data.get(f'template-{hemi}')
    if surf is None:
        surf = sf.load_mesh(os.path.join(os.path.dirname(__file__), f'template.{hemi}.surf'))
        surf = surf.convert(space='vox')
        loaded_mesh_data[f'template-{hemi}'] = surf
    return surf.copy()


def neighborhood(order):
    """
    Retrieve the precomputed neighborhood mapping for icospheres
    """
    filename = os.path.join(os.path.dirname(__file__), f'neighborhoods.npz')
    if not os.path.isfile(filename):
        raise RuntimeError(f'{filename} cannot be located - make sure it downloaded per instructions in the readme')
    npd = np.load(filename)
    return npd[f'ico-{order}-1000'].astype(np.int64)


def load_topology(order):
    """
    Load mesh topology information for a specific icosphere order
    """
    device = utils.get_device()
    topology = {
        'order': order,
        'size': nvertices(order),
        'faces': torch.from_numpy(faces(order).astype(np.int64, copy=False)).to(device),
        'adj_edges_a': torch.from_numpy(adjancency_indices(order)[:, 0].astype(np.int64, copy=False)).to(device),
        'adj_edges_b': torch.from_numpy(adjancency_indices(order)[:, 1].astype(np.int64, copy=False)).to(device),
        'adj_weights': torch.from_numpy(adjancency_weights(order).astype(np.float32, copy=False)).to(device),
        'upsampler': [
            torch.from_numpy(upsampling_sources(order).astype(np.int64, copy=False)).to(device),
            torch.from_numpy(upsampling_weights(order).astype(np.float32, copy=False)).to(device),
        ],
        'pooling_a': torch.from_numpy(pooling_sources(order)[:, 0].astype(np.int64, copy=False)).to(device),
        'pooling_b': torch.from_numpy(pooling_sources(order)[:, 1].astype(np.int64, copy=False)).to(device),
        'pooling_weights': torch.from_numpy(pooling_weights(order).astype(np.float32, copy=False)).to(device),
        'pooling_shape_a': pooling_shapes(order)[0].astype(np.int64, copy=False),
        'pooling_shape_b': pooling_shapes(order)[1].astype(np.int64, copy=False),
    }
    if order in (6, 7):
        topology['edge_faces'] = torch.from_numpy(edge_faces(order).astype(np.int64, copy=False)).to(device)
    return topology


def faces(order):
    return get_ico_data(f'ico-{order}-faces')


def vertices(order):
    return get_ico_data(f'ico-{order}-vertices')


def nvertices(order):
    return vertices(order).shape[0]


def edges(order):
    return get_ico_data(f'ico-{order}-edges')


def adjancency_indices(order):
    return get_ico_data(f'ico-{order}-adjacency-indices')


def adjancency_weights(order):
    return np.expand_dims(get_ico_data(f'ico-{order}-adjacency-values'), -1)


def upsampling_sources(order):
    return get_ico_data(f'ico-{order}-sources')


def upsampling_weights(order):
    return np.expand_dims(get_ico_data(f'ico-{order}-bary'), -1)


def pooling_sources(order):
    return get_ico_data(f'mapping-{order - 1}-to-{order}-indices')


def pooling_weights(order):
    return get_ico_data(f'mapping-{order - 1}-to-{order}-values')


def pooling_shapes(order):
    return get_ico_data(f'mapping-{order - 1}-to-{order}-shape')


def edge_faces(order):
    return get_ico_data(f'ico-{order}-edge-faces')
