# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import App
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.video_objectdetection import VideoObjectDetection
from datetime import datetime, UTC

from arduino.app_peripherals.camera import Camera

import os
import subprocess
import cv2

vlm_processing = False
vlm_prompting_label = "beer"
vlm_prompting_confidence = 0.7
vlm_prompt = "Describe the kind of beverages on the table.."
vlm_tokens = "60"

# NOTE: On the Uno Q 2GB, the SmolVLM-256M VLM processing takes around 40 seconds with 18 tokens and 96x96 image sizes
# NOTE: On the Uno Q 4GB, the SmolVLM-256M VLM processing takes around 34 seconds with 18 tokens and 96x96 image sizes
# NOTE: On the Uno Q 4GB, the SmolVLM-500M VLM processing takes around 50 seconds with 60 tokens and 500x500 image sizes

RESIZED_IMAGE_WIDTH = 500
RESIZED_IMAGE_HEIGHT = 500

def get_vlm_description(image_path, prompt=vlm_prompt):
    global vlm_processing

    # point to the models folder where the .so files are
    vlm_dir = "./models"
    vlm_binary = os.path.join(vlm_dir, "llama-mtmd-cli")

    # NOTE: To load the SmolVLM-256M model, you need to first download these files and put them in the models folder: 
    # mmproj-SmolVLM-256M-Instruct-Q8_0.gguf and SmolVLM-256M-Instruct-Q8_0.gguf

    #model_path = os.path.join(vlm_dir, "SmolVLM-256M-Instruct-Q8_0.gguf")
    #mmproj_path = os.path.join(vlm_dir, "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")  
    
    # Default loading the SmolVLM-500M model
    model_path = os.path.join(vlm_dir, "SmolVLM-500M-Instruct-Q8_0.gguf")
    mmproj_path = os.path.join(vlm_dir, "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf")  

    if not os.path.exists(vlm_binary):
        return f"binary not found at {vlm_binary}"

    os.chmod(vlm_binary, 0o755) # ensure the binary is executable

    # set library path to the local directory inside the container
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{vlm_dir}:" + env.get("LD_LIBRARY_PATH", "")

    command = [
        vlm_binary,
        "-m", model_path,
        "--mmproj", mmproj_path,
        "--image", image_path,
        "-p", prompt,
        "-n", vlm_tokens,
        "--temp", "0.2"
    ]    

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, env=env)     
        if result.returncode != 0:
            vlm_processing = False
            return f"VLM Error: {result.stderr}\nSTDOUT: {result.stdout}"

        vlm_processing = False        
        return result.stdout.strip()

    except Exception as e:
        vlm_processing = False
        return f"process failed: {str(e)}"

# For debugging, uncomment for manual VLM test at startup
#print("VLM TEST: checking binary availability...")
#print(f"VLM TEST RESULT: {get_vlm_description('./test_images/image5.jpg')}") # optional. Create a test_images folder with image(s) for testing

camera = Camera()

ui = WebUI()

detection_stream = VideoObjectDetection(camera, confidence=0.5, debounce_sec=0.0) 

ui.on_message("override_th", lambda sid, threshold: detection_stream.override_threshold(threshold))

# The detection callback
def on_detection(detections):
    global vlm_processing   
    
    for key, values in detections.items():
        ui.send_message("detection", {"content": key, "confidence": values[0]['confidence'], "timestamp": datetime.now(UTC).isoformat()})
        
        # ensure we are not actively running the VLM model
        if key == vlm_prompting_label and values[0]['confidence'] > vlm_prompting_confidence and not vlm_processing:
            vlm_processing = True
            try:
                image = camera.capture()
                
                if image is not None:
                    image = cv2.resize(image, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                    # Hack: the Web UI brick is hard-coded to look at a specific folder for files. Such as the assets/img folder
                    img_dir = "./assets/img" # save image to show on Web UI
                    temp_path = os.path.join(img_dir, "vlm_temp.jpg")

                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir, exist_ok=True)
                        print(f"created directory: {img_dir}")
        
                    cv2.imwrite(temp_path, image)
                    print("Temporaty image saved... Starting VLM processing...")
                    
                    vlm_description = get_vlm_description(temp_path)
                    print(f"VLM response: {vlm_description}")
                    
                    ui.send_message("vlm_response", {
                        "text": vlm_description,
                        "timestamp": datetime.now(UTC).isoformat()
                    })
                    

                else:
                    print("error: camera.get_frame() returned None")
            
            except Exception as e:
                print(f"callback crashed: {e}")
            
            finally:
                vlm_processing = False

detection_stream.on_detect_all(on_detection)
App.run()