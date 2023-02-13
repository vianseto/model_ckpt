import json
import re
import string
import requests
import os
import cv2
import base64
import uuid
import shutil
import time
import torch
import random
import wave
import struct
import time
from subprocess import getoutput
from PIL import Image
from io import BytesIO
from datetime import datetime
from pathvalidate import replace_symbol
from flask import Flask, request
from threading import Thread

avatar_dir = "/data/server/avatar"

def wav2lip_gan(face_path, audio_path, output_path):
	torch.cuda.empty_cache()
	os.system("python /data/server/Wav2Lip/inference.py --checkpoint_path /data/server/Wav2Lip/checkpoints/wav2lip_gan.pth --face="+face_path+" --audio="+audio_path+" --outfile="+output_path+" --nosmooth --fps=30")

def real_esrgan_video(video_path, output_path, scale):
	torch.cuda.empty_cache()
	os.system("python /data/server/Real-ESRGAN/inference_realesrgan_video.py -n RealESRGAN_x4plus -i="+video_path+" -o="+output_path+" --outscale="+scale+" --face_enhance --fps 30")

def gfpgan(id_path, filename, scale):
	torch.cuda.empty_cache()
	# Upscale temp Folder khusus filename
	upscale_path = os.path.join(id_path,filename+"_upscale")
	frame_path = os.path.join(upscale_path,"frame")
	video_path = os.path.join(id_path,filename+".mp4")
	audio_path = os.path.join(id_path,filename+".wav")
	output_path = os.path.join(id_path,filename+"_upscale.mp4")
	os.system(f"sudo mkdir {upscale_path}")
	os.system(f"sudo mkdir {frame_path}")

	# Membuat frame dari video
	os.system(f"sudo ffmpeg {video_path} -vf fps=30 {upscale_path}/frame/frame_%08d.jpg")
	os.system("python /data/server/GFPGAN/inference_gfpgan.py -i "+upscale_path+"/frame -o "+upscale_path+" -v 1.4 -s 2 --bg_upsampler realesrgan")

	# Menggambungkan frame hasil
	os.system("sudo ffmpeg -framerate 30 -i "+upscale_path+"/restored_imgs/frame_%08d.jpg -i "+audio_path+" -c:v libx264 "+output_path)

	# Hapus folder upscale
	#os.system("sudo rm -r "+upscale_path)

	return output_path

def sanitize(json_param, param_key):
	try:
		if param_key=="id": 
			json_param = replace_symbol(json_param.strip())
		elif (param_key=="img") or (param_key=="video"):
			if json_param.endswith(".mp4"): 
				json_param = json_param.replace(".mp4", "")
			elif json_param.endswith(".jpg"):
				json_param = json_param.replace(".jpg", "")
			elif json_param.endswith(".png"):
				json_param = json_param.replace(".png", "")
				json_param = replace_symbol(json_param.strip())
		elif (param_key=="text") or (param_key=="type"):
			remove = string.punctuation
			json_param = json_param.strip()
			if (json_param!=None) and (param_key=='text'):
				exclude = ",.!?-"
				for char in exclude:
					remove = remove.replace(char,'')
				for char in remove:
					json_param = json_param.translate( { ord(char): None } )    
		elif (json_param!=None) and (param_key=='type'):
			remove = remove.replace('_','')
			for char in remove:
				json_param = json_param.translate( { ord(char): None } ) 
		elif param_key=="template":
			remove = string.punctuation
			remove = remove.replace('_','')
			json_param = json_param.strip()
			for char in remove:
				json_param = json_param.translate( { ord(char): None } )
			json_param = json_param+".mp4"
		elif (param_key=="prompt") or (param_key=="negative_prompt"):
			remove = string.punctuation
			exclude = "()[]<>,.|:-"
			for char in exclude:
				remove = remove.replace(char,'')
			for char in remove:
				json_param = json_param.strip()
				json_param = json_param.translate( { ord(char): None } )
	except:
		respond = {'msg': 'input parameter not valid'}
		return json.dumps(respond, indent=2)
	else:
		if (param_key=="data") or (param_key=="audio"):
		    return json_param
		else:    
		    return ''.join(char for char in json_param if ord(char) < 128)

def botika_tts(text,type,id_dir,filename):
	url = "https://tts.botika.online/tts/index.php"
	
	payload = json.dumps({
		"accessToken": "785c1152-10d0-e883-5eb1-27489aa59db4",
		"api": "text_to_speech",
		"text": text,
		"voice": type,
		"language": "id",
		"returnAs": "url"
	})
	headers = {
	'Content-Type': 'application/json'
	}

	response = requests.request("POST", url, headers=headers, data=payload).text
	link = json.loads(response)['url']
	save_path = os.path.join(id_dir,filename+".wav")
	os.system("sudo wget "+link+" -O "+save_path)
	return save_path

app = Flask(__name__)

@app.route('/')
def index():
	return ''

# Route Wav2Lip

@app.route('/avatar/video', methods=['POST'])
def avatar_video():
	#log_input_request(request.get_json(), "avatar_video.txt", request.url, request.method)
	if request.method == 'POST':
		# Sanitasi ID
		if not isinstance(request.json['id'],str):
			respond = {'msg': 'id value must be string'}
			return json.dumps(respond, indent=2)
		else:
			id = sanitize(request.json['id'],"id") 

        	# Sanitasi template
		if not isinstance(request.json['template'],str):
			respond = {'msg': 'template value must be string of filename'}
			return json.dumps(respond, indent=2)
		else:
			template = sanitize(request.json['template'],"template")  
        
		# Sanitasi TTS
		if not isinstance(request.json['tts'], dict):
			respond = {'msg': 'tts value must be dictionary'}
			return json.dumps(respond, indent=2)
		else:
			# memastikan parameter memiliki semua kunci yang dibutuhkan
			required_keys = ['text', 'type']
			for key in required_keys:
				if key not in request.json['tts']:
					respond = {'msg': f'missing key [{key}] in tts'}
					return json.dumps(respond, indent=2)
			for key, value in request.json.get('tts').items():
				if  not isinstance(value,str):
					respond = {'msg': f'{key} value on tts must be string'}
					return json.dumps(respond, indent=2)
			tts = request.json['tts']
			text = sanitize(tts["text"],"text")
			type = sanitize(tts["type"],"type") 
        
		# Sanitasi Audio
		audio = request.json['audio']
		try:
			if audio!=None:
				audio = base64.b64decode(request.json['audio'])
		except:
			respond = {'msg': 'data value not a valid base64 string'}
			return json.dumps(respond, indent=2)
            
		# Sanitasi enhance
		if not isinstance(request.json['enhance'],bool):
			respond = {'msg': 'enhance value must be true / false'}
			return json.dumps(respond, indent=2)
		else:
			enhance = request.json['enhance']

		id_dir = os.path.join(avatar_dir,"video",id)

		# Jika folder id tidak ada, result : not found
		if os.path.exists(id_dir) != True:
			os.system(f"sudo mkdir {id_dir}")
                
		template_path = os.path.join(avatar_dir,"template",template)
		if os.path.isfile(template_path)!=True:
			respond={'msg':'template not found'} 
			return json.dumps(respond, indent=2)
		else:
			filename = uuid.uuid4().hex
			lock = os.path.join(id_dir,filename+"_lock")
			if ((tts['text'] is None) and (tts['type'] is None)) and (audio is None):
				respond={'msg':'no audio specifed'} 
				return json.dumps(respond, indent=2)
			elif (tts['text'] is None) and (tts['type'] is None):
				wav_path = os.path.join(id_dir,filename+".wav")
				wav_file = open(wav_path, "wb")
				wav_file.write(audio)
				try:
					wavefile = wave.open(wav_path, 'r')
					wavefile.close()
				except:
					os.remove(wav_path)
					respond = {'msg': "uploaded file not an audio"}
					return json.dumps(respond, indent=2)
			elif audio is None:
				wav_path = botika_tts(tts['text'], tts['type'], id_dir, filename) 
            
			output_dir = os.path.join(id_dir,filename+".mp4")        
			torch.cuda.empty_cache()
			os.system(f"sudo touch {lock}")

			# Thumbnail
			video = cv2.VideoCapture(template_path)
			ret, frame = video.read()
			resized_frame = cv2.resize(frame, (640, int(640 / frame.shape[1] * frame.shape[0])))
			thumbnail_path = os.path.join(id_dir,filename+"_thumbnail.jpg")
			cv2.imwrite(thumbnail_path, resized_frame)
			video.release()
            
			# Generate Video
			try:
				wav2lip_gan(template_path, wav_path, output_dir)
			except:
				os.system(f"sudo rm {wav_path}") 
				os.system(f"sudo rm {thumbnail_path}")
				os.system(f"sudo rm {lock}")
				respond = {'msg': "fail to generate video, maybe because memory usage"}
				return json.dumps(respond, indent=2)
			else: 
				if enhance:
					#try:
					enhance_path = gfpgan(id_dir, filename, 2)
					os.system("sudo mv -f "+enhance_path+" "+output_dir)
					#except:
						#respond = {'msg': "out of memory"}
						#return json.dumps(respond, indent=2)

				with open(output_dir, 'rb') as open_file:
					byte_video = open_file.read()

				base64_bytes = base64.b64encode(byte_video)
				base64_string = base64_bytes.decode('utf-8')
				os.system(f"sudo rm {lock}")
				os.system(f"sudo rm {wav_path}")

				respond = {'result': "success", 'id':id, 'video':filename, 'data':base64_string}
				return json.dumps(respond, indent=2)              
	else:
		respond = {'msg': 'please use POST method'}
		return json.dumps(respond, indent=2)    
    
@app.route('/avatar/video/list', methods=['POST'])
def avatar_video_list():
	#log_input_request(request.get_json(), "avatar_video_list.txt", request.url, request.method)
	if request.method == 'POST':
		# Sanitasi ID
		if not isinstance(request.json['id'],str):
			respond = {'msg': 'id value must be string'}
			return json.dumps(respond, indent=2)
		else:
			id = sanitize(request.json['id'],"id")  

		# Cek folder dengan id tersebut apakah ada / tidak
		id_dir = os.path.join(avatar_dir,"video", id)

		# Jika folder id tidak ada, result : not found
		if os.path.exists(id_dir) != True:
			respond = {'msg': f"id {id} not found"}
			return json.dumps(respond, indent=2)
		else:
			extensions = (".mp4", "_lock")
			files = [f for f in os.listdir(id_dir) if f.endswith(extensions)]

			if len(files)==0:
				respond = {'msg': "doesn't generated any video yet"}
				return json.dumps(respond, indent=2)
			else:
				files.sort(key=lambda x: os.path.getctime(os.path.join(id_dir, x)))
				respond_list = []
				for filename in files:
					if filename.endswith("_lock"):
						filename = filename.replace("_lock","")
						status = "process"
					else:
						filename = filename.replace(".mp4","")
						status = "done"
					respond_dict = {'name':filename, 'status':status}
					respond_list.append(respond_dict)
				respond = {'result': 'success', 'id': id, 'video' : respond_list}
				return json.dumps(respond, indent=2)               
	else:
		respond = {'msg': 'please use POST method'}
		return json.dumps(respond, indent=2)

@app.route('/avatar/video/download', methods=['POST'])
def avatar_video_download():
	#log_input_request(request.get_json(), "avatar_video_download.txt", request.url, request.method)
	if request.method == 'POST':
		# Sanitasi ID
		if not isinstance(request.json['id'],str):
			respond = {'msg': 'id value must be string'}
			return json.dumps(respond, indent=2)
		else:
			id = sanitize(request.json['id'],"id")

		# Sanitasi video filename
		if not isinstance(request.json['video'],str):
			respond = {'msg': 'video value must be string of filename'}
			return json.dumps(respond, indent=2)
		else:
			video = sanitize(request.json['video'],"video")  

		# Sanitasi img thumbnail
		if not isinstance(request.json['thumbnail'],bool):
			respond = {'msg': 'thumbnail value must be true / false'}
			return json.dumps(respond, indent=2)
		else:
			thumbnail = request.json['thumbnail'] 

		id_dir = os.path.join(avatar_dir,"video", id)
		# Jika folder id tidak ada, result : not found
		if os.path.exists(id_dir) != True:
			respond = {'msg': f"id {id} not found"}
			return json.dumps(respond, indent=2)
		else:        
			if thumbnail:
				# Thumbnail path
				filename = video+"_thumbnail"
				thumbnail_path = os.path.join(id_dir,filename+".jpg")

				with open(thumbnail_path, 'rb') as open_file:
					byte_img = open_file.read()

				base64_bytes = base64.b64encode(byte_img)
				base64_string = base64_bytes.decode('utf-8')
			else:
				# Video path
				filename = video
				video_path = os.path.join(id_dir,video+".mp4")

				with open(video_path, 'rb') as open_file:
					byte_video = open_file.read()

				base64_bytes = base64.b64encode(byte_video)
				base64_string = base64_bytes.decode('utf-8')

			respond = {'result': "success", 'id': id, 'video':filename, 'data': base64_string}
			return json.dumps(respond, indent=2)              
	else:
		respond = {'msg': 'please use POST method'}
		return json.dumps(respond, indent=2)

@app.route('/avatar/video/delete', methods=['POST'])
def avatar_video_delete():
	#log_input_request(request.get_json(), "avatar_video_delete.txt", request.url, request.method)
	if request.method == 'POST':
		# Sanitasi ID
		if not isinstance(request.json['id'],str):
			respond = {'msg': 'id value must be string'}
			return json.dumps(respond, indent=2)
		else:
			id = sanitize(request.json['id'],"id") 

		# Sanitasi video filename
		if not isinstance(request.json['video'],str):
			respond = {'msg': 'video value must be string of filename'}
			return json.dumps(respond, indent=2)
		else:
			video = sanitize(request.json['video'],"video")   

		# Cek folder dengan id tersebut apakah ada / tidak
		id_dir = os.path.join(avatar_dir,"video", id)

		# Jika folder id tidak ada, result : not found
		if os.path.exists(id_dir) != True:
			respond = {'msg': f"id {id} not found"}
			return json.dumps(respond, indent=2)
		else:
			# Video path
			video_path = os.path.join(id_dir,video+".mp4")

			# Thumbnail path
			thumbnail_path = os.path.join(id_dir,video+"_thumbnail.jpg")

			# Jika file tidak ada, result : not found
			if (os.path.isfile(video_path) != True) and (os.path.isfile(thumbnail_path) != True):
				respond = {'msg': f'file {video} not found'}
				return json.dumps(respond, indent=2)
			elif (os.path.isfile(video_path) == True) and (os.path.isfile(thumbnail_path) != True):
				os.system(f"sudo rm {video_path}")
			elif (os.path.isfile(video_path) != True) and (os.path.isfile(thumbnail_path) == True):
				os.system(f"sudo rm {thumbnail_path}")
				respond = {'msg': f'file {video} not found'}
				return json.dumps(respond, indent=2)
			else:
				os.system(f"sudo rm {video_path}")
				os.system(f"sudo rm {thumbnail_path}")

			respond = {'result': 'success'}
			return json.dumps(respond, indent=2)                
	else:
		respond = {'msg': 'please use POST method'}
		return json.dumps(respond, indent=2)


if __name__ == '__main__':
	app.run(threaded=False, processes=3, host='0.0.0.0', port='8000')
