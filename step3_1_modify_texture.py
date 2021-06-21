import argparse
import os
import numpy as np
import cv2
import tensorflow as tf
from utils.recon_depth import split_data, get_pixel_value, get_face_texture, uvd2xyz
from utils.create_renderer import create_renderer_graph
from scipy.io import savemat, loadmat
from utils.loader import load_data, load_lm3d, load_center3d, read_facemodel
import math as m

tf.enable_eager_execution()

_FACE_V_NUM = 35709
_FACE_T_NUM = 70789
_HAIREAR_V_NUM = 28000
_HAIREAR_T_NUM = 54000

def load_depthrecon_graph(graph_filename, image_size=256):
    with tf.gfile.GFile(graph_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        inputs = tf.placeholder(name='inputs', shape=[
            None, image_size, image_size, 5], dtype=tf.float32)
        tf.import_graph_def(graph_def, name='resnet', input_map={
                            'inputs:0': inputs})
        output = graph.get_tensor_by_name('resnet/depth_map:0')
        return graph, inputs, output

def create_shaperecon_graph(image_size=256):
    with tf.Graph().as_default() as graph:
        imgs = tf.placeholder(dtype=tf.float32, shape=[1, image_size, image_size, 3])
        hairear_uv = tf.placeholder(dtype=tf.float32, shape=[1, _HAIREAR_V_NUM, 2])
        hairear_dmap = tf.placeholder(dtype=tf.float32, shape=[1, image_size, image_size, 1])
        face3d_data = tf.placeholder(dtype=tf.float32, shape=[1, 396])
        face_shape2d = tf.placeholder(dtype=tf.float32, shape=[1, _FACE_V_NUM, 2])
        
        focal, center, _ = split_data(face3d_data)
        face_texture = get_face_texture(imgs, face_shape2d)
        hairear_d = get_pixel_value(hairear_dmap, hairear_uv)
        hairear_uv_trans = hairear_uv + 0.5
        hairear_uv_trans = tf.concat([hairear_uv_trans[:, :, 0:1], 
            256 - hairear_uv_trans[:, :, 1:]], axis=-1)
        hairear_uvd = tf.concat([hairear_uv_trans, hairear_d], axis=-1)
        hairear_xyz = uvd2xyz(hairear_uvd, focal, center)
        hairear_texture = get_pixel_value(imgs, hairear_uv)
    return graph, imgs, hairear_uv, hairear_dmap, face3d_data, face_shape2d, face_texture, hairear_xyz, hairear_texture

def Texture_formation_block(tex_coeff, facemodel):
    face_texture = np.einsum('ij,aj->ai',facemodel.texBase,tex_coeff) + facemodel.meantex

    # reshape face texture to [batchsize,N,3], note that texture is in RGB order
    face_texture = np.reshape(face_texture,[-1,3])

    return face_texture

def Compute_norm(face_shape,facemodel):
    shape = face_shape
    face_id = facemodel.tri
    point_id = facemodel.point_buf
    
    ## expand dim to [batch, N, 3] #TODO:
    shape = tf.expand_dims(shape, axis = 0)

    # face_id and point_id index starts from 1
    face_id = tf.cast(face_id - 1,tf.int32)
    point_id = tf.cast(point_id - 1,tf.int32)

    #compute normal for each face
    v1 = tf.gather(shape,face_id[:,0], axis = 1)
    v2 = tf.gather(shape,face_id[:,1], axis = 1)
    v3 = tf.gather(shape,face_id[:,2], axis = 1)
    e1 = v1 - v2
    e2 = v2 - v3
    face_norm = tf.cross(e1,e2)

    face_norm = tf.nn.l2_normalize(face_norm, dim = 2) # normalized face_norm first
    ## TODO:
    face_norm = tf.cast(face_norm, tf.float32)
    face_norm = tf.concat([face_norm,tf.zeros([tf.shape(shape)[0],1,3])], axis = 1) #TODO:

    #compute normal for each vertex using one-ring neighborhood
    v_norm = tf.reduce_sum(tf.gather(face_norm, point_id, axis = 1), axis = 2)
    v_norm = tf.nn.l2_normalize(v_norm, dim = 2)
    
    return v_norm

def Compute_rotation_matrix(angles):
    n_data = tf.shape(angles)[0]

    # compute rotation matrix for X-axis, Y-axis, Z-axis respectively
    rotation_X = tf.concat([tf.ones([n_data,1]),
        tf.zeros([n_data,3]),
        tf.reshape(tf.cos(angles[:,0]),[n_data,1]),
        -tf.reshape(tf.sin(angles[:,0]),[n_data,1]),
        tf.zeros([n_data,1]),
        tf.reshape(tf.sin(angles[:,0]),[n_data,1]),
        tf.reshape(tf.cos(angles[:,0]),[n_data,1])],
        axis = 1
        )

    rotation_Y = tf.concat([tf.reshape(tf.cos(angles[:,1]),[n_data,1]),
        tf.zeros([n_data,1]),
        tf.reshape(tf.sin(angles[:,1]),[n_data,1]),
        tf.zeros([n_data,1]),
        tf.ones([n_data,1]),
        tf.zeros([n_data,1]),
        -tf.reshape(tf.sin(angles[:,1]),[n_data,1]),
        tf.zeros([n_data,1]),
        tf.reshape(tf.cos(angles[:,1]),[n_data,1])],
        axis = 1
        )

    rotation_Z = tf.concat([tf.reshape(tf.cos(angles[:,2]),[n_data,1]),
        -tf.reshape(tf.sin(angles[:,2]),[n_data,1]),
        tf.zeros([n_data,1]),
        tf.reshape(tf.sin(angles[:,2]),[n_data,1]),
        tf.reshape(tf.cos(angles[:,2]),[n_data,1]),
        tf.zeros([n_data,3]),
        tf.ones([n_data,1])],
        axis = 1
        )

    rotation_X = tf.reshape(rotation_X,[n_data,3,3])
    rotation_Y = tf.reshape(rotation_Y,[n_data,3,3])
    rotation_Z = tf.reshape(rotation_Z,[n_data,3,3])

    # R = RzRyRx
    rotation = tf.matmul(tf.matmul(rotation_Z,rotation_Y),rotation_X)

    rotation = tf.transpose(rotation, perm = [0,2,1])

    return rotation

def Illumination_block(face_texture,norm_r,gamma):
    n_data = tf.shape(gamma)[0]
    n_point = tf.shape(norm_r)[1]
    gamma = tf.reshape(gamma,[n_data,3,9])
    # set initial lighting with an ambient lighting
    init_lit = tf.constant([0.8,0,0,0,0,0,0,0,0])
    gamma = gamma + tf.reshape(init_lit,[1,1,9])

    # compute vertex color using SH function approximation
    a0 = m.pi 
    a1 = 2*m.pi/tf.sqrt(3.0)
    a2 = 2*m.pi/tf.sqrt(8.0)
    c0 = 1/tf.sqrt(4*m.pi)
    c1 = tf.sqrt(3.0)/tf.sqrt(4*m.pi)
    c2 = 3*tf.sqrt(5.0)/tf.sqrt(12*m.pi)

    Y = tf.concat([tf.tile(tf.reshape(a0*c0,[1,1,1]),[n_data,n_point,1]),
        tf.expand_dims(-a1*c1*norm_r[:,:,1],2),
        tf.expand_dims(a1*c1*norm_r[:,:,2],2),
        tf.expand_dims(-a1*c1*norm_r[:,:,0],2),
        tf.expand_dims(a2*c2*norm_r[:,:,0]*norm_r[:,:,1],2),
        tf.expand_dims(-a2*c2*norm_r[:,:,1]*norm_r[:,:,2],2),
        tf.expand_dims(a2*c2*0.5/tf.sqrt(3.0)*(3*tf.square(norm_r[:,:,2])-1),2),
        tf.expand_dims(-a2*c2*norm_r[:,:,0]*norm_r[:,:,2],2),
        tf.expand_dims(a2*c2*0.5*(tf.square(norm_r[:,:,0])-tf.square(norm_r[:,:,1])),2)],axis = 2)

    color_r = tf.squeeze(tf.matmul(Y,tf.expand_dims(gamma[:,0,:],2)),axis = 2)
    color_g = tf.squeeze(tf.matmul(Y,tf.expand_dims(gamma[:,1,:],2)),axis = 2)
    color_b = tf.squeeze(tf.matmul(Y,tf.expand_dims(gamma[:,2,:],2)),axis = 2)
    
    face_texture = tf.expand_dims(face_texture, axis = 0)
    #[batchsize,N,3] vertex color in RGB order
    face_color = tf.stack([color_r*face_texture[:,:,0],color_g*face_texture[:,:,1],color_b*face_texture[:,:,2]],axis = 2)

    return face_color

def depth_recon(data_path, save_path):
    print(f'[INFO] [Step3] Read data from {data_path}')
    # create face recon graph
    depthrecon_graph, inputs, depth_map = load_depthrecon_graph('model/depth_net.pb')
    depth_recon_sess = tf.Session(graph=depthrecon_graph)

    # create shape recon graph
    shaperecon_graph, input_imgs, input_uv, input_dmap, input_facedata, input_face2d, \
        output_face_texture, output_hairear_xyz, output_hairear_texture = create_shaperecon_graph()
    shape_recon_sess = tf.Session(graph=shaperecon_graph)

    # create renderer graph
    depth_render_graph, input_focal, input_center, input_depth, \
        input_vertex, input_tri, output_depthmap = create_renderer_graph(v_num=_FACE_V_NUM + _HAIREAR_V_NUM, t_num=_FACE_T_NUM + _HAIREAR_T_NUM)
    render_sess = tf.Session(graph=depth_render_graph)

    names = [i for i in os.listdir(data_path) if i.endswith('mat')]
    for i, name in enumerate(names):
        # print(i, name.split('.')[0])
        # read and prepare data
        data_input = loadmat(os.path.join(data_path, name))
        imgs_input = data_input['img'].astype(np.float32).reshape([1, 256, 256, 3])
        
        face_d_input = data_input['face_depthmap'].astype(np.float32).reshape([1, 256, 256, 1])
        face_xyz_input = np.expand_dims(data_input['face_projection'].astype(np.float32), 0)
        facewoh_m_input = data_input['facemask_withouthair'].astype(np.float32).reshape([1, 256, 256, 1])
        face3d_data_input = data_input['face3d'].astype(np.float32).reshape([1, 396]) #TODO:

        xy = data_input['points_index']
        uv = np.concatenate([xy[:, 1:], xy[:, :1]], axis=1)
        hairear_uv_input = uv.astype(np.float32).reshape([1, _HAIREAR_V_NUM, 2])
        hairear_m_input = data_input['input_mask'].astype(np.float32).reshape([1, 256, 256, 1])
        
        depth_input = np.concatenate([imgs_input/255, (10 - face_d_input) * facewoh_m_input,
         hairear_m_input], -1)

        # recon hairear depth
        depth_output = depth_recon_sess.run(depth_map, feed_dict={
            inputs: depth_input
        })
        
        # recover head shape from hairear depth
        h_xyz, h_texture, f_texture = shape_recon_sess.run([
            output_hairear_xyz, output_hairear_texture, output_face_texture], 
            feed_dict={
                input_imgs: imgs_input,
                input_dmap: depth_output,
                input_face2d: face_xyz_input,
                input_uv: hairear_uv_input,
                input_facedata: face3d_data_input
        })
        

        # render head depth
        head_xyz = np.concatenate([
            np.expand_dims(data_input['face_shape'], 0), h_xyz], axis=-2)
        head_d = np.tile(10 - head_xyz[..., -1:], [1, 1, 3])
        head_tri = np.concatenate([
            data_input['face_tri'], data_input['points_tri'] + _FACE_V_NUM], axis=0)
        head_dmap = render_sess.run(output_depthmap, feed_dict={
            input_focal: data_input['face3d'][:, 0].reshape([1]),
            input_center: data_input['face3d'][:, 1:3].reshape([1, 1, 2]),
            input_depth: head_d,
            input_vertex: head_xyz,
            input_tri: np.expand_dims(head_tri, 0) - 1,
        })

        # Get Texture from coefficient
        facemodel = read_facemodel()
        coeff = face3d_data_input[:,3:260]
        tex_coeff = coeff[:,144:224]
        face_texture = Texture_formation_block(tex_coeff, facemodel)

        angles = coeff[:,224:227]
        rotation = Compute_rotation_matrix(angles)

        face_norm = Compute_norm(data_input['face_shape'], facemodel)
        norm_r = tf.matmul(face_norm,rotation)

        gamma = coeff[:,227:254]
        face_color = Illumination_block(face_texture, norm_r, gamma)
        face_color = face_color.numpy()
        face_color = np.squeeze(face_color)
        # face texture clipping to 0~255
        face_color = np.clip(face_color, 0, 255)
        face_xyz = data_input['face_shape'].astype(np.float32)

        result = {
            'hairear_shape': h_xyz.squeeze(0),
            'hairear_texture': h_texture.squeeze(0),
            'hairear_tri': data_input['points_tri'],
            'face_shape': data_input['face_shape'],
            'face_texture': face_color,
            'face_tri': data_input['face_tri'],
            'hairear_index': data_input['points_index'],
            'facemask_withouthair': data_input['facemask_withouthair'],
            'depth': head_dmap[..., 0].squeeze(0), 
            'mask': head_dmap[..., -1].squeeze(0)
        }
        savemat(os.path.join(save_path, name), result, do_compression=True)

    depth_recon_sess.close()
    shape_recon_sess.close()
    render_sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.')
    args = parser.parse_args()

    data_path = os.path.join(args.root_dir, 'output/step2') 
    save_path = os.path.join(args.root_dir, 'output/step3') 
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # recon depth and recover the head geometry
    depth_recon(data_path, save_path)
    
