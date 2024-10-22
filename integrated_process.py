import os, argparse
from step1_1_manipulate_expression import face_recon
from step2_face_segmentation import prepare_mask
from step3_1_modify_texture import depth_recon
from step4_1_modify_texture import save_obj
import sys
sys.path.append('face-parsing.PyTorch')
from step0_get_segmentation import get_face_alignment
sys.path.remove('face-parsing.PyTorch')

def filemaking(input_img):
    # run function
    get_face_alignment(
        './inputs',
        './inputs',
        debug='store_true')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    expr_path = 'expressions'
    input_path = 'inputs'
    expr_img = [i for i in os.listdir(expr_path) if (i.endswith('png') or i.endswith('jpg') or i.endswith('jpeg')) and (i[:2] != '._') ]
    get_face_alignment(expr_path, expr_path, debug='store_true')
    src_img = os.path.join(input_path, input_img)
    for target in expr_img :
        for degree in range(0, 110, 10):
            output_name = src_img.split(os.path.sep)[1].split('.')[0] + '_' + target.split('.')[0] +'_' + str(degree) + '%'
            face_recon(src_img, os.path.join(expr_path, target), input_path, 'outputs/step1', output_name,'outputs/step1/vis', degree=degree)
    prepare_mask('outputs/step1', 'outputs/step2', 'outputs/step1', 'outputs/step2/vis')
    depth_recon('outputs/step2', 'outputs/step3')
    save_obj('outputs/step3','outputs/step4', True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', default='AI_gen.jpg')
    args = parser.parse_args()
    filemaking(args.input_img)

