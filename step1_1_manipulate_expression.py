import argparse
import tensorflow as tf 
import numpy as np
import cv2
from PIL import Image
import os
from scipy.io import loadmat,savemat
from utils.preprocess import POS, headrecon_preprocess_withmask, facerecon_preprocess_yu_5p, facerecon_preprocess
from utils.loader import load_data, load_lm3d, load_center3d, read_facemodel
from utils.recon_face import  compute_center2d, compute_faceshape
from utils.create_renderer import create_renderer_graph
from PIL import Image
import math as m

tf.enable_eager_execution()


def load_facerecon_graph(graph_filename):
    with tf.gfile.GFile(graph_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        input = tf.placeholder(name='input_imgs', shape=[
            None, 224, 224, 3], dtype=tf.float32)
        tf.import_graph_def(graph_def, name='resnet', input_map={
                            'input_imgs:0': input})
        output = graph.get_tensor_by_name('resnet/coeff:0')
    return graph, input, output

def Texture_formation_block(tex_coeff, facemodel):
    # print(f"tex_coeff: {tex_coeff.shape}") #(1, 80)
    # print(f"facemodel.texBase: {facemodel.texBase.shape}") # (107127, 80)
    # print(f"facemodel.meantex: {facemodel.meantex.shape}") # (1, 107127)
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


def face_recon(src_path, tgt_path, input_path, output_path, output, vis_path=None, s_factor=1.5, focal=1015, center=112, align_nums=10, degree=100):
    # load BFM
    facemodel = read_facemodel()
    # read standard landmarks for face recon preprocessing
    lm3D = load_lm3d(align_nums)
    # read head center for depth recon preprocessing
    head_center3d = load_center3d()
    
    # create face recon graph
    face_recon_graph, images, coef = load_facerecon_graph('model/model_mask3_white_light.pb')
    face_recon_sess = tf.Session(graph=face_recon_graph)

    # create renderer graph
    depth_render_graph, input_focal, input_center, input_depth, \
        input_vertex, input_tri, output_depthmap = create_renderer_graph()
    render_sess = tf.Session(graph=depth_render_graph)
    
    print("source", src_path.split(os.path.sep)[-1].split('.')[0])
    # print("target", tgt_path.split(os.path.sep)[-1].split('.')[0])
    print("target", tgt_path.split('.')[0])

    #TODO: # get source image's coefficient
    mask = loadmat(os.path.join(input_path, src_path.split(os.path.sep)[-1].split('.')[0] + '.mat'))['mask']
    ## load images and corresponding 5 facial landmarks
    if align_nums == 5:
        img, lm = load_data(src_path, 
            os.path.join(input_path, src_path.split(os.path.sep)[-1].split('.')[0] + '_detection.txt'))

        lm = lm[-10:].reshape([5, 2])
        input_img, inv_params = facerecon_preprocess_yu_5p(img, lm, lm3D)
    elif align_nums == 10:
        img, lm = load_data(src_path, 
            os.path.join(input_path, src_path.split(os.path.sep)[-1].split('.')[0] + '_landmark.txt'))

        lm = lm.reshape([68, 2])
        input_img, inv_params = facerecon_preprocess(img, lm, lm3D)
    
    # recon face
    coeff = face_recon_sess.run(coef, feed_dict={images: np.expand_dims(input_img, 0)})[..., :-1]

    #TODO: # get target image's coefficient
    tgt_mask = loadmat(tgt_path.split('.')[0] + '.mat')['mask']
    ## load images and corresponding 5 facial landmarks
    if align_nums == 5:
        tgt_img, tgt_lm = load_data(tgt_path, 
            tgt_path.split('.')[0] + '_detection.txt')

        tgt_lm = tgt_lm[-10:].reshape([5, 2])
        tgt_input_img, tgt_inv_params = facerecon_preprocess_yu_5p(tgt_img, tgt_lm, lm3D)
    elif align_nums == 10:
        tgt_img, tgt_lm = load_data(tgt_path, 
            tgt_path.split('.')[0] + '_landmark.txt')

        tgt_lm = tgt_lm.reshape([68, 2])
        tgt_input_img, tgt_inv_params = facerecon_preprocess(tgt_img, tgt_lm, lm3D)
    
    # recon face
    tgt_coeff = face_recon_sess.run(coef, feed_dict={images: np.expand_dims(tgt_input_img, 0)})[..., :-1]

    # transfer target expression to source expression 
    degree = degree/100
    coeff[:, 80:144] = tgt_coeff[:, 80:144] * degree + coeff[:, 80:144] * (1-degree)
    
    # preprocess input image for depth recon net
    # reproject the reconstructed face to raw image with adjusted focal and center
    f = focal * inv_params[0]
    p_center = inv_params[0] * center + inv_params[1]
    face_shape, face_projection, landmarks_2d = compute_faceshape(coeff, facemodel, inv_params)
    
    # crop the raw image with head center as the image center
    center2d, displacement = compute_center2d(head_center3d, coeff, facemodel, f, p_center)
    _, s =  POS(face_projection.transpose(), facemodel.meanshape.reshape([-1, 3]).transpose())
    crop_img, crop_mask, inv_params_, crop_lm, crop_param = headrecon_preprocess_withmask(img, mask, landmarks_2d, center2d.reshape([2]), s*s_factor/100)
    #TODO:
    # save processed data
    data = np.zeros([3 + 257 + 136])
    data[0] = f / inv_params_[0]                                            
    data[1: 3] = (p_center - inv_params_[1].reshape([2]))/inv_params_[0]     
    data[3: 260] = coeff.reshape([257])
    data[257: 260] = data[257: 260] - displacement.reshape([3])
    data[260:] = crop_lm.reshape([136])
    face_projection_cropped, _ = compute_center2d(np.expand_dims(face_shape, 0),
        np.expand_dims(data[3:260], 0), facemodel, data[0], data[1:3], displace_flag=False, apply_pose=False)

    # render face depth
    d = 10 - face_shape[:, 2:]
    d = np.tile(np.expand_dims(d, 0), [1, 1, 3])   
    d_map = render_sess.run(output_depthmap, feed_dict={
        input_focal: data[0].reshape([1]),
        input_center: data[1: 3].reshape([1, 1, 2]),
        input_depth: d,
        input_vertex: np.expand_dims(face_shape, 0),
        input_tri: np.expand_dims(facemodel.tri, 0) - 1 # start from 0
    })

    if vis_path:
        cv2.imwrite(os.path.join(vis_path, output + '.png'),
        crop_img.astype(np.uint8))
        cv2.imwrite(os.path.join(vis_path, output + '_dmap.png'), d_map[0] * 255)
    savemat(os.path.join(output_path, output + '.mat'), 
        {'img': crop_img.astype(np.uint8),
            'mask': crop_mask.astype(np.uint8),
            'crop_param': crop_param.astype(np.float32),
            'face3d': data.astype(np.float32), # data --> face3d #TODO:
            # 0: focal; [1, 3) center; [3,260): face coeff; [260~396): landmark  
            'face_shape':face_shape.astype(np.float32),
            'face_projection': face_projection_cropped.squeeze(0).astype(np.float32),
            'face_depthmap': d_map[..., 0].squeeze(0), 
            'face_mask': d_map[..., -1].squeeze(0),
            'face_tri': facemodel.tri}, do_compression=True)

    # # Get Texture from coefficient
    # face_xyz = face_shape.astype(np.float32)
    # tex_coeff = coeff[:,144:224]
    # face_texture = Texture_formation_block(tex_coeff, facemodel)
    
    # angles = coeff[:,224:227]
    # rotation = Compute_rotation_matrix(angles)

    # face_norm = Compute_norm(face_shape, facemodel)
    # norm_r = tf.matmul(face_norm,rotation)

    # gamma = coeff[:,227:254]
    # face_color = Illumination_block(face_texture, norm_r, gamma)
    # face_color = face_color.numpy()
    # face_color = np.squeeze(face_color)
    # face_tri = facemodel.tri

    # with open(os.path.join(save_path, 'step1' + '.obj'), 'w') as f:
    #     face_color_ = np.clip(face_color,0,255)/255
    #     for i in range(face_xyz.shape[0]):
    #         f.write('v %f %f %f %f %f %f\n' %(*face_xyz[i, :], *face_color_[i, :]))
    #     for i in range(face_tri.shape[0]):
    #         f.write('f {} {} {}\n'.format(*face_tri[i, :]))
    
    face_recon_sess.close()
    render_sess.close()


if __name__ == '__main__':
    input_path = 'inputs'
    save_path = 'outputs/step1'
    vis_path = 'outputs/step1/vis'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # define parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', default='inputs/junghoo.jpg')
    parser.add_argument('--tgt_path', default='expressions/wide_smile.png')
    parser.add_argument('--output', default='edit_expression')
    args = parser.parse_args()

    degree = 100
    # recon 3d face and prepare the input to depth recon
    face_recon(args.src_path, args.tgt_path, input_path, save_path, args.output, vis_path, degree=degree)

