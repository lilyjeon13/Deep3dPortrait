import os
#from step0_68points import get_facial_landmark
from step1_recon_3d_face import face_recon
from step2_face_segmentation import prepare_mask
from step3_get_head_geometry import depth_recon
from step4_save_obj import save_obj
import sys
sys.path.append('face-parsing.PyTorch')
from step0_get_segmentation import get_face_alignment
sys.path.remove('face-parsing.PyTorch')
from flask import Flask, flash, request, redirect, url_for
from flask import send_file

def filemaking():
    # run function
    get_face_alignment(
        '/workspace/FaceReenactment/Deep3DPortrait/inputs',
        '/workspace/FaceReenactment/Deep3DPortrait/inputs',
        debug='store_true')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    face_recon('inputs', 'outputs/step1','outputs/step1/vis')
    prepare_mask('outputs/step1', 'outputs/step2', 'outputs/step1', 'outputs/step2/vis')
    depth_recon('outputs/step2', 'outputs/step3')
    save_obj('outputs/step3','outputs/step4', True)

app = Flask (__name__)
 
@app.route('/',methods = ['POST'])
def upload_file():
        # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    # print(file)
    print(os.path.join(os.getcwd(), "inputs",file.filename))
    file.save(os.path.join(os.getcwd(), "inputs",file.filename))
    filemaking()
    return send_file(os.path.join(os.getcwd(),"outputs/step4","input.obj"),as_attachment=True)
 
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8888)