import requests
import os
import re
import shutil
import json
import ast
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import zipfile
import uvicorn

app = FastAPI()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_CODE_ROOT = os.path.join(SCRIPT_DIR, "Client_code_generator")

class GenerateRequest(BaseModel):
    email: str
    project_name: str


def extract_code(text):
    matches = re.findall(r'```(?:[a-zA-Z0-9]*)\n(.*?)```', text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    return text.strip()


def check_arduino_library_availability(library_queries):
  """Return Arduino CLI library search results for required libraries."""
  cli_path = shutil.which("arduino-cli")
  if not cli_path:
    return {
      "cli_available": False,
      "available": [],
      "missing": list(dict.fromkeys(library_queries))
    }

  available = []
  missing = []
  for lib_query in dict.fromkeys(library_queries):
    try:
      result = subprocess.run(
        [cli_path, "lib", "search", lib_query, "--format", "json"],
        capture_output=True,
        text=True,
        timeout=10
      )
      if result.returncode != 0:
        missing.append(lib_query)
        continue

      payload = (result.stdout or "").strip()
      found = False
      if payload:
        try:
          parsed = json.loads(payload)
          if isinstance(parsed, list):
            found = len(parsed) > 0
          elif isinstance(parsed, dict):
            for key in ("libraries", "results", "items"):
              if isinstance(parsed.get(key), list) and parsed.get(key):
                found = True
                break
            if not found:
              found = any(isinstance(v, list) and len(v) > 0 for v in parsed.values())
        except Exception:
          found = '"name"' in payload.lower()

      if found:
        available.append(lib_query)
      else:
        missing.append(lib_query)
    except Exception as e:
      print(f"Arduino CLI library check failed for '{lib_query}': {e}")
      missing.append(lib_query)

  return {
    "cli_available": True,
    "available": available,
    "missing": missing
  }


def build_dynamic_library_queries(mcu_name, context_fragments, component_class_map=None):
  """Build likely Arduino library queries from board type + connected component names/classes."""
  queries = ["Servo", "Arduino_JSON"]
  if "STM32" in (mcu_name or "").upper():
    queries.append("STM32FreeRTOS")

  text_parts = [str(mcu_name or "")]
  text_parts.extend(str(x) for x in (context_fragments or []))
  if component_class_map:
    for comp_name, comp_class in component_class_map.items():
      text_parts.append(f"{comp_name} {comp_class}")
  context_text = " ".join(text_parts).lower()

  keyword_library_map = {
    "tca9548": ["ClosedCube TCA9548A"],
    "i2c_multiplexer": ["ClosedCube TCA9548A"],
    "multiplexer": ["ClosedCube TCA9548A"],
    "mpu6050": ["I2Cdev", "MPU6050"],
    "icm20948": ["Adafruit ICM20X", "Adafruit ICM20948", "Adafruit Unified Sensor"],
    "icm20": ["Adafruit ICM20X", "Adafruit ICM20948", "Adafruit Unified Sensor"],
    "imu": ["I2Cdev", "MPU6050", "Adafruit ICM20X", "Adafruit ICM20948", "Adafruit Unified Sensor"],
    "bmp280": ["Adafruit BMP280 Library", "Adafruit Unified Sensor"],
    "bme280": ["Adafruit BME280 Library", "Adafruit Unified Sensor"],
    "gps": ["TinyGPSPlus"],
    "neo-6m": ["TinyGPSPlus"],
    "nmea": ["TinyGPSPlus"],
    "neopixel": ["Adafruit NeoPixel"],
    "ws2812": ["Adafruit NeoPixel"],
    "oled": ["Adafruit SSD1306", "Adafruit GFX Library", "Adafruit Unified Sensor"],
    "ssd1306": ["Adafruit SSD1306", "Adafruit GFX Library", "Adafruit Unified Sensor"],
    "vl53": ["Adafruit VL53L0X"],
    "tof": ["Adafruit VL53L0X"],
    "ina219": ["Adafruit INA219"],
    "pca9685": ["Adafruit PWM Servo Driver Library"]
  }

  for keyword, libs in keyword_library_map.items():
    if keyword in context_text:
      queries.extend(libs)

  return list(dict.fromkeys(queries))

def update_status(email, project_name, status_value):
    # Post to both development (192.168.50.161) and production (192.168.50.247) main servers
    for ip in ["192.168.50.161", "192.168.50.247"]:
        try:
            requests.post(f"http://{ip}:8080/process_endtrigger", json={
                "email": email,
                "project_name": project_name,
                "status_type": "synthesis",
                "status": status_value
            }, timeout=2.0)
        except Exception as e:
            print(f"Failed to post status to {ip}:", e)


def ask_llm(email, project_name, command, SERVICE_URLS):
    """Try primary Gemini, then fallback to local LLM helpers."""
    # Try primary Gemini service
    try:
        url = SERVICE_URLS['gemini'] + "/ask_gemini"
        print(f"[ask_llm] Trying primary: {url}")
        res = requests.post(url, json={"email": email, "project_name": project_name, "command": command}, timeout=60)
        if res.status_code == 200:
            res_json = res.json()
            if email in res_json and res_json[email]:
                return res_json[email]
            print(f"[ask_llm] Primary returned empty or missing key, trying fallbacks...")
    except Exception as e:
        print(f"[ask_llm] Primary Gemini failed: {e}")

    # Fallback 1: Local LLM Helper llama3.2 3B
    try:
        url = "http://192.168.50.4:8332/llama323db"
        print(f"[ask_llm] Trying fallback 1: {url}")
        res = requests.post(url, json={"email": email, "command": command}, timeout=60)
        if res.status_code == 200:
            res_json = res.json()
            if email in res_json and res_json[email]:
                return res_json[email]
    except Exception as e:
        print(f"[ask_llm] Fallback 1 (llama323db) failed: {e}")

    # Fallback 2: Local LLM Helper Nemotron Chat
    try:
        url = "http://192.168.50.4:8332/nemotron_chat"
        print(f"[ask_llm] Trying fallback 2: {url}")
        res = requests.post(url, json={"email": email, "command": command}, timeout=60)
        if res.status_code == 200:
            res_json = res.json()
            if email in res_json and res_json[email]:
                return res_json[email]
    except Exception as e:
        print(f"[ask_llm] Fallback 2 (nemotron_chat) failed: {e}")

    # Fallback 3: Direct local Ollama
    try:
        url = "http://localhost:11434/api/generate"
        print(f"[ask_llm] Trying fallback 3 (Ollama): {url}")
        res = requests.post(url, json={"model": "llama3.2:3B", "prompt": command, "stream": False}, timeout=60)
        if res.status_code == 200:
            return res.json().get("response", "")
    except Exception as e:
        print(f"[ask_llm] Fallback 3 (Ollama) failed: {e}")

    raise RuntimeError("[ask_llm] All LLM endpoints failed.")


def get_project_dir(email, project_name):
    return os.path.join(CLIENT_CODE_ROOT, email, project_name)

@app.post("/download_nodecode")
def download_nodecode(request: GenerateRequest):
    email = request.email
    project_name = request.project_name
    project_dir = get_project_dir(email, project_name)
    
    if not os.path.exists(project_dir):
        raise HTTPException(status_code=404, detail="Project code not found")
        
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, project_dir)
                zipf.write(file_path, arcname)
    temp_zip.close()
    return FileResponse(temp_zip.name, media_type="application/zip", filename=f"{project_name}_code.zip")

@app.post("/post_nodecode")
def generate_code(request: GenerateRequest):


    email = request.email
    project_name = request.project_name

    # Always fetch reqdat at the start so it is defined for all usages
    reqdat = requests.post("http://192.168.50.247:8774/get_node_code",json={"email":email,"project_name":project_name}).json()['drawflow']['Home']['data']
    device_class_map = {}
    try:
      cat_data = requests.get("http://192.168.50.247:9074/getdevice_cat").json()
      device_class_map = cat_data.get(email, {}).get(project_name, {})
    except Exception as e:
      print(f"Could not fetch component class map for library inference: {e}")

    # Check if the connection list of the output is blank or not blank
    for node_id, node_data in reqdat.items():
        outputs = node_data.get('outputs', {})
        for output_name, output_info in outputs.items():
            connections = output_info.get('connections', [])
            if not connections:
                print(f"Node {node_id} output '{output_name}' has a blank connection list.")
            else:
                print(f"Node {node_id} output '{output_name}' has connections: {connections}")

    update_status(email, project_name, "Started generating node components...")

    # Clean up previously generated directories for this project
    project_base_dir = get_project_dir(email, project_name)
    for dir_name in ["middleware", "firmware", "Installer"]:
        dir_path = os.path.join(project_base_dir, dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up old directory: {dir_path}")
            except Exception as e:
                print(f"Failed to clean up {dir_path}: {e}")

    optimize_select = requests.get("http://192.168.50.247:9060/get_optimize_select").json()[email][project_name] #Getting the optimize select for the components and category of the components inside the list 
    try:
        robot_desc_req = requests.get("http://192.168.50.247:9060/get_robot_description").json()
        robot_description = robot_desc_req.get(email, {}).get(project_name, "")
    except Exception as e:
        print(f"Error fetching robot description: {e}")
        robot_description = ""

    SERVICE_URLS = {
        'storage': 'http://192.168.50.247:9060',
        'gemini': 'http://192.168.50.247:9886',
        'semantic_match': 'http://192.168.50.4:8466',
        'component_extract': 'http://192.168.50.161:4972',
        'mcu_db': 'http://192.168.50.247:5978',
        'node_store': 'http://192.168.50.247:8774',
        'component_request': 'http://192.168.50.4:7795',
        'llm_helper': 'http://192.168.50.4:8332',
        'search_service': 'http://192.168.50.247:9767',
        'restart_service': 'http://192.168.50.247:5987',
        'classcomp': 'http://192.168.50.247:9060/get_classcomp'
    }
    ROBOREACTOR_COMPUTER_VISION_HARDWARE_MAP = {
    "vision_profiles": {
    "RPI_ZERO_2W": {
      "hardware": {
        "cpu": "Quad Cortex-A53",
        "ram_mb": 512,
        "gpu": "VideoCore IV",
        "ai_acceleration": False,
        "recommended_resolution": [
          "320x240",
          "416x416",
          "640x480"
        ],
        "camera_backend": [
          "Picamera2",
          "V4L2",
          "OpenCV"
        ]
      },

     
      "opencv_support": {
        "image_processing": {
          "status": "excellent",
          "fps": "20-30",
          "functions": [
            "cv2.Canny",
            "cv2.GaussianBlur",
            "cv2.threshold",
            "cv2.adaptiveThreshold",
            "cv2.erode",
            "cv2.dilate",
            "cv2.morphologyEx",
            "cv2.equalizeHist",
            "cv2.resize",
            "cv2.flip",
            "cv2.rotate",
            "cv2.filter2D",
            "cv2.Sobel",
            "cv2.Laplacian"
          ]
        },
        "aruco_and_marker": {
          "status": "excellent",
          "fps": "15-30",
          "functions": [
            "cv2.aruco.detectMarkers",
            "cv2.aruco.drawDetectedMarkers",
            "cv2.aruco.estimatePoseSingleMarkers",
            "cv2.solvePnP",
            "cv2.QRCodeDetector",
            "AprilTag"
          ]
        },
        "feature_detection": {
          "status": "good",
          "fps": "10-20",
          "functions": [
            "cv2.ORB_create",
            "cv2.BRISK_create",
            "cv2.FastFeatureDetector_create",
            "cv2.goodFeaturesToTrack",
            "cv2.cornerHarris"
          ]
        },
        "tracking": {
          "status": "good",
          "fps": "10-20",
          "functions": [
            "cv2.calcOpticalFlowPyrLK",
            "cv2.calcOpticalFlowFarneback",
            "cv2.meanShift",
            "cv2.CamShift"
          ]
        },
        "robot_navigation": {
          "status": "excellent",
          "applications": [
            "line_following",
            "lane_detection",
            "wall_following",
            "obstacle_detection",
            "visual_odometry",
            "basic_slam",
            "aruco_navigation",
            "qr_navigation"
          ]
        },
        "dnn_ai": {
          "status": "moderate",
          "fps": "1-5",
          "supported_models": [
            "YOLOv5n",
            "YOLOv8n",
            "NanoDet",
            "MobileNetSSD",
            "EfficientDetLite"
          ]
        }
      }
    },
    "RPI_4": {
      "hardware": {
        "cpu": "Quad Cortex-A72",
        "ram_options_gb": [2, 4, 8],
        "gpu": "VideoCore VI",
        "ai_acceleration": False,
        "recommended_resolution": [
          "640x480",
          "720p"
        ]
      },
      "opencv_support": {
        "image_processing": "excellent",
        "aruco": "excellent",
        "optical_flow": "excellent",
        "feature_matching": "excellent",
        "monocular_slam": "good",
        "stereo_vision": "moderate",
        "tiny_yolo": "good",
        "medium_yolo": "limited",
        "pose_estimation": "moderate",
        "face_detection": "good",
        "object_tracking": "good",
        "depth_estimation": "moderate"
      },
      "recommended_robotics_tasks": [
        "autonomous_navigation",
        "warehouse_robotics",
        "swarm_control",
        "digital_twin_streaming",
        "sensor_fusion",
        "vision_processing"
      ]
    },
    "RPI_5": {
      "hardware": {
        "cpu": "Quad Cortex-A76",
        "ram_options_gb": [4, 8],
        "gpu": "VideoCore VII",
        "ai_acceleration": False,
        "recommended_resolution": [
          "720p",
          "1080p"
        ]
      },
      "opencv_support": {
        "image_processing": "excellent",
        "aruco": "excellent",
        "dense_optical_flow": "good",
        "stereo_vision": "good",
        "monocular_slam": "excellent",
        "rtabmap": "moderate",
        "yolov5": "good",
        "yolov8n": "good",
        "pose_estimation": "good",
        "segmentation": "moderate",
        "multi_camera": "moderate"
      },
      "recommended_robotics_tasks": [
        "advanced_navigation",
        "robot_digital_twin",
        "vision_ai_processing",
        "multi_sensor_fusion",
        "mapping",
        "slam"
      ]
    },
    "ORANGE_PI": {
      "hardware": {
        "cpu": "RK3588 / H618 dependent",
        "ram_options_gb": [2, 4, 8, 16],
        "gpu": "Mali GPU",
        "npu": "up to 6 TOPS on RK3588"
      },
      "opencv_support": {
        "image_processing": "excellent",
        "aruco": "excellent",
        "slam": "good",
        "tiny_yolo": "excellent",
        "yolov8": "good",
        "segmentation": "moderate",
        "pose_estimation": "good",
        "stereo_depth": "good"
      },
      "recommended_robotics_tasks": [
        "edge_ai",
        "robotics_ai",
        "local_inference",
        "vision_navigation"
      ]
    },
    "BANANA_PI": {
      "hardware": {
        "cpu": "ARM Cortex series",
        "ram_options_gb": [2, 4, 8],
        "gpu": "Mali GPU"
      },
      "opencv_support": {
        "image_processing": "excellent",
        "aruco": "excellent",
        "tiny_yolo": "moderate",
        "slam": "moderate",
        "tracking": "good"
      }
    },
    "JETSON_NANO": {
      "hardware": {
        "cpu": "Quad Cortex-A57",
        "ram_gb": 4,
        "gpu": "128-core Maxwell CUDA",
        "cuda": True,
        "tensor_rt": True
      },
      "opencv_support": {
        "image_processing": "excellent",
        "aruco": "excellent",
        "slam": "good",
        "yolov5": "good",
        "yolov8": "good",
        "segmentation": "moderate",
        "pose_estimation": "good",
        "depth_ai": "good",
        "stereo_depth": "good"
      },
      "recommended_robotics_tasks": [
        "autonomous_robotics",
        "ai_navigation",
        "vision_ai",
        "slam",
        "edge_inference"
      ]
    },
    "JETSON_ORIN_NANO": {
      "hardware": {
        "cpu": "6-core ARM A78AE",
        "gpu": "1024-core Ampere",
        "tensor_cores": 32,
        "ram_gb": [4, 8],
        "ai_performance_tops": 40
      },
      "opencv_support": {
        "image_processing": "excellent",
        "aruco": "excellent",
        "slam": "excellent",
        "rtabmap": "excellent",
        "yolov8": "excellent",
        "segmentation": "excellent",
        "pose_estimation": "excellent",
        "depth_estimation": "excellent",
        "multi_camera": "excellent"
      },
      "recommended_robotics_tasks": [
        "industrial_robotics",
        "advanced_ai",
        "3d_mapping",
        "multi_robot_coordination",
        "digital_twin"
      ]
    },
    "NUC_PC": {
      "hardware": {
        "cpu": "Intel i5/i7/i9",
        "ram_options_gb": [8, 16, 32, 64],
        "gpu": [
          "Intel Xe",
          "NVIDIA GPU optional"
        ]
      },
      "opencv_support": {
        "all_opencv_functions": "excellent",
        "slam": "excellent",
        "dense_mapping": "excellent",
        "yolov8_large": "excellent",
        "segmentation": "excellent",
        "transformer_ai": "good",
        "3d_reconstruction": "excellent",
        "gaussian_splatting": "moderate"
      },
      "recommended_robotics_tasks": [
        "robotics_server",
        "fleet_management",
        "simulation",
        "training_ai_models",
        "digital_twin_server",
        "centralized_ai"
      ]
    },
    "GENERIC_COMPUTER": {
      "hardware": {
        "cpu": "x86_64 or ARM",
        "ram_gb": "variable",
        "gpu": "optional"
      },
      "opencv_support": {
        "basic_vision": "excellent",
        "aruco": "excellent",
        "tracking": "excellent",
        "slam": "depends_on_hardware",
        "ai_inference": "depends_on_gpu"
      },
      "robotics_role": [
        "development_machine",
        "simulation_machine",
        "vision_processing",
        "robotics_control_server"
      ]
    }
  },
  "global_supported_computer_vision_applications": [
    "aruco_navigation",
    "apriltag_localization",
    "qr_navigation",
    "line_following",
    "lane_detection",
    "optical_flow",
    "visual_odometry",
    "monocular_slam",
    "stereo_vision",
    "feature_matching",
    "object_tracking",
    "motion_detection",
    "face_detection",
    "gesture_detection",
    "pose_estimation",
    "object_detection",
    "semantic_segmentation",
    "depth_estimation",
    "3d_mapping",
    "point_cloud_processing",
    "digital_twin_visualization",
    "robot_navigation",
    "warehouse_robotics",
    "swarm_robotics",
    "industrial_robotics",
    "drone_navigation"
  ]
}
    commu_mapnode = {
          "SBC_node": 
          { 
            "input":{ 
            "Serial":"input_1", 
            "Camera": "input_2", 
            "UART": "input_3", 
            "GPIO": "input_4",
             "I2C": "input_5", 
             "SPI": "input_6",
             "CSI":"input_7",
             "I2S":"input_8",
            
             }, 
             "output":{ 
              "Logic_control":"output_1" 
              } 
              }, 
            "mcus_node": 
            { 
             "input":{ 
               #"Serial":"input_1",
               "UART": "input_1", 
               "Servo": "input_2", 
               "PWM": "input_3", 
               "Digital": "input_4", 
               "DAC": "input_5", 
               "Analog": "input_6", 
                "I2C": "input_7", 
                "CAN": "input_8", 
                "SPI": "input_9"
              
               },
              "output":{ 
                "Serial":"output_1",
                "PWM":"output_2"
                } 
                }, 
            "Servo_multiplexer": { 
              "output": { 
                "I2C": "output_1",
                "VCC": "output_3", 
                "GND": "output_3", 
                "PWM": "output_4" 
                }, 
                "input": { 
                  "OE": "input_1",
                  "PWM": "input_2" 
                  } }, 
            "I2C_multiplexer": {
              "input": {
                "I2C": "input_1"
              },
              "output": {
                "I2C": "output_1"
              }
            },
            "DC_motor_driver":{ 
              "input": { 
                "VCC": "input_1",
                "GND": "input_2" 
              }, 
              "output": { 
                "PWM": "output_1", 
                "Digital": "output_2",
                "Motor_A": "output_3", 
                "Motor_B": "output_4" 
                } 
              }, 
            "Stepper_motor_driver": { 
              "input": { 
                "VCC": "input_1", 
                "GND": "input_2"
              }, 
              "output": { 
                "Digital": "output_1", 
                "Digital_2": "output_2",
                "Digital_3": "output_3",
                "Coil_A": "output_4", 
                "Coil_B": "output_5" 
                } }, 
            "BLDC_motor_driver": { 
              "input": { 
                "VCC": "input_1", 
                "GND": "input_2"
               }, 
               "output": { 
                "PWM": "output_1",
                "Phase_U": "output_2", 
                "Phase_V": "output_3",
                "Phase_W": "output_4", 
                "Telemetry": "output_5"
                 } }, 
             "Flight_controller": { 
              "input": { 
                "IMU": "input_1", 
                "GPS": "input_2", 
                "PWM": "input_3", 
                "RC_Input": "input_4" 
                }, 
             "output": { 
              "UART": "output_1", 
              "Telemetry": "output_2",
              "PWM": "output_3"
              } 
              }, 
           "ESC_electronics_speed_controller": { 
            "output": { 
              "PWM": "output_1",
              "Phase_V": "output_2",
              "Phase_W": "output_3",
              "RPM_Feedback": "output_4" 
             }, 
             "input": { 
              "PWM": "input_1", 
              "VBAT": "input_2", 
              "GND": "input_3" }
            }, 
          "servo_motor": { 
           "input": { 
            "PWM": "input_1", 
            "VCC": "input_2", 
            "GND": "input_3" 
            }, 
            "output": { 
              "Angle": "output_1"
               }
            },
           "USB_camera": {
            "input": {
              "camera_detection": "input_1",
              "VCC": "input_2",
              "GND": "input_3"
            },
            "output": {
              "Serial": "output_1"
            }
           },
           "CSI_camera": {
            "input": {
              "camera_detection": "input_1",
              "VCC": "input_2",
              "GND": "input_3"
            },
            "output": {
              "CSI": "output_1"
            }
           },
           "LIDAR": {
            "input": {
              "scan_cmd": "input_1",
              "VCC": "input_2",
              "GND": "input_3"
            },
            "output": {
              "cloud_point": "output_1",
              "Serial": "output_2"
            }
           },
           "GPS": {
            "input": {
              "VCC": "input_1",
              "GND": "input_2"
            },
            "output": {
              "nmea_data": "output_1",
              "UART": "output_2"
            }
           },
           "Ultrasonic_sensor": {
            "input": {
              "trigger": "input_1",
              "VCC": "input_2",
              "GND": "input_3"
            },
            "output": {
              "Digital": "output_1"
            }
           },
           "navigation_input": {
            "input": {
              "waypoints": "input_1",
              "map_data": "input_2"
            },
            "output": {
              "navigation_status": "output_1",
              "target_pose": "output_2"
            }
           },
           "stepper_motor": {
            "input": { 
              "Coil_A": "input_1", 
              "Coil_B": "input_2" 
            }, 
            "output": { 
              "Position": "output_1" 
              }
           },
           "IMU": {
            "input": {
              "VCC": "input_1",
              "GND": "input_2"
            },
            "output": {
              "I2C": "output_1",
              "gyro": "output_2",
              "mag": "output_3"
            }
           },
           "Distance_sensor": {
            "input": {
              "VCC": "input_1",
              "GND": "input_2"
            },
            "output": {
              "Analog": "output_1"
            }
           },
           "Force_sensor": {
            "input": {
              "VCC": "input_1",
              "GND": "input_2"
            },
            "output": {
              "Analog": "output_1"
            }
           },
           "Proximity_sensor": {
            "input": {
              "VCC": "input_1",
              "GND": "input_2"
            },
            "output": {
              "Digital": "output_1"
            }
           },
           "Encoder": {
            "input": {
              "VCC": "input_1",
              "GND": "input_2"
            },
            "output": {
              "Digital": "output_1",
              "Digital_2": "output_2"
            }
           },
          "dc_motor": { 
            "output": { 
              "PWM": "input_1", 
              "PWM": "input_2" 
              }, 
            "input": { 
              "Rotation": "output_1" 
              }
         }, 
            "bldc_motor": { 
              "output": { 
                "PWM": "input_1", 
                "PWM": "input_2", 
                "PWM": "input_3" 
              }, 
             "input": { 
              "RPM": "output_1" 
              } 
            }, 
            "Duct_fan_thruster": 
            { "output": { 
              "PWM": "output_1", 
              "VCC": "output_2", 
              "GND": "output_3" 
              }, 
            "input": { 
              "PWM": "input_1" 
              } 
            }, 
          "Cellular_LTE":{
               "input":{
                    "VCC":"input_1",
                    "GND":"input_2",
                    "Antenna":"input_3"
               },
               "output":{
                    "Serial":"output_1"
               }   
          },
          "Micro_jet_engine": { 
            "input": { 
              "Fuel": "input_1", 
              "Ignition": "input_2", 
              "PWM": "input_3" }, 
            "output": { 
              "Thrust": "output_1" 
              } 
            },
            "Audio_amplifier": {
              "input": { "VCC": "input_1", "GND": "input_2", "Analog": "input_3" },
              "output": { "Speaker+": "output_1", "Speaker-": "output_2" }
            },
            "USB_sound_card": {
              "input": { "Analog": "input_1" },
              "output": { "Serial": "output_1", "I2S": "output_3", "Analog": "output_2" }
            },
            "Microphone": {
              "input": { "VCC": "input_1", "GND": "input_2" },
              "output": { "Analog": "output_1" }
            },
            "Speaker": {
              "input": { "In+": "input_1", "In-": "input_2" },
              "output": { "Audio_Status": "output_1" }
            }
          }
    id_device_mapping = {} #Getting the id device mapping for th node id data processing 
    totalcomp_class = requests.get("http://192.168.50.247:9060/get_classcomp").json() #Getting the total components class selection 
    BRIDGE_COMPONENT_CLASSES = {"ADC mux", "Servo_multiplexer", "I2C_mux", "I2C_multiplexer", "Flight_controller"}
    I2C_BRIDGE_CLASSES = {"I2C_mux", "I2C_multiplexer"}

    def normalize_component_key(name):
      return re.sub(r"\s+", " ", str(name)).strip()

    def clean_component_name(name):
      # Remove trailing node id suffixes like "_92" with optional whitespace around separator.
      return re.sub(r"\s*_\d+$", "", str(name)).strip()

    def resolve_device_class_from_cache(component_name, class_cache):
      """Resolve component class using tolerant key matching against cached server category map."""
      if not class_cache:
        return None
      raw_name = str(component_name or "")
      clean_name = clean_component_name(raw_name)
      key_candidates = [
        raw_name,
        normalize_component_key(raw_name),
        clean_name,
        normalize_component_key(clean_name),
      ]
      for key in key_candidates:
        if key in class_cache:
          return class_cache.get(key)
      return None

    def extract_gemini_text(response_json, email, allow_any_value=False):
      """Extract the best text field from /ask_gemini responses.

      When allow_any_value is False, only use response fields tied to the
      requesting context to avoid cross-request payload leakage.
      """
      if isinstance(response_json, dict):
        if isinstance(response_json.get(email), str) and response_json[email].strip():
          return response_json[email].strip()
        for fallback_key in ("response", "text", "message", "output"):
          value = response_json.get(fallback_key)
          if isinstance(value, str) and value.strip():
            return value.strip()
        if allow_any_value:
          for value in response_json.values():
            if isinstance(value, str) and value.strip():
              return value.strip()
      elif isinstance(response_json, str) and response_json.strip():
        return response_json.strip()
      return ""

    def looks_like_non_python_firmware(text):
      lowered = (text or "").lower()
      firmware_markers = [
        "#include <",
        "void setup()",
        "void loop()",
        "semaphorehandle_t",
        "xtaskcreate",
        "arduino.h",
        "stm32freertos",
      ]
      return any(marker in lowered for marker in firmware_markers)

    def is_valid_python_vision_code(text):
      lowered = (text or "").lower()
      has_python_signals = any(
        marker in lowered
        for marker in [
          "import cv2",
          "from ultralytics import yolo",
          "cv2.",
          "def post_telemetry",
          "cameraonlyvo",
          "calcopticalflowpyr",
        ]
      )
      return has_python_signals and not looks_like_non_python_firmware(text)

    def is_valid_python_middleware_code(text):
      candidate = (text or "").strip()
      if not candidate:
        return False
      if looks_like_non_python_firmware(candidate):
        return False

      lowered = candidate.lower()
      required_signals = [
        "from fastapi import",
        "fastapi(",
        "/motor_neuron",
      ]
      if not all(signal in lowered for signal in required_signals):
        return False

      try:
        ast.parse(candidate)
      except SyntaxError:
        return False

      return True

    def build_fallback_middleware_code(mcu_baudrates):
      serial_ports_config = {}
      if isinstance(mcu_baudrates, dict) and mcu_baudrates:
        for idx, (mcu_name, baudrate) in enumerate(mcu_baudrates.items()):
          serial_ports_config[mcu_name] = {
            "port": f"/dev/ttyUSB{idx}",
            "baudrate": int(baudrate)
          }
      else:
        serial_ports_config = {
          "mcu_1": {"port": "/dev/ttyUSB0", "baudrate": 115200}
        }

      serial_ports_literal = json.dumps(serial_ports_config, indent=4)
      return f'''import json
import threading
import time

from fastapi import FastAPI, Request
import serial
import uvicorn

app = FastAPI()
store_sensory = {{}}
store_lock = threading.Lock()


class RobotHardwareBridge:
    def __init__(self, serial_ports_config):
        self.serial_ports = {{}}
        self.serial_locks = {{}}
        for mcu_name, config in serial_ports_config.items():
            self.serial_locks[mcu_name] = threading.Lock()
            try:
                self.serial_ports[mcu_name] = serial.Serial(
                    config["port"],
                    int(config["baudrate"]),
                    timeout=0.05
                )
                print(f"Connected to {{mcu_name}} on {{config['port']}} @ {{config['baudrate']}}")
                reader = threading.Thread(
                    target=self._read_telemetry_loop,
                    args=(mcu_name,),
                    daemon=True,
                )
                reader.start()
            except Exception as e:
                self.serial_ports[mcu_name] = None
                print(f"Failed to open serial for {{mcu_name}}: {{e}}")

    def _send_command(self, mcu_name, command):
        serial_port = self.serial_ports.get(mcu_name)
        if serial_port is None:
            print(f"MCU '{{mcu_name}}' unavailable")
            return False
        try:
            with self.serial_locks[mcu_name]:
                payload = json.dumps(command) + "\\n"
                serial_port.write(payload.encode("utf-8"))
                serial_port.flush()
            return True
        except Exception as e:
            print(f"Send error for {{mcu_name}}: {{e}}")
            return False

    def _read_telemetry_loop(self, mcu_name):
        serial_port = self.serial_ports.get(mcu_name)
        if serial_port is None:
            return
        while True:
            try:
                with self.serial_locks[mcu_name]:
                    raw = serial_port.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    time.sleep(0.01)
                    continue
                payload = json.loads(raw)
                frame_type = str(payload.get("type", "telemetry"))
                key = f"{{mcu_name}}_{{frame_type}}"
                with store_lock:
                    store_sensory[key] = payload
            except Exception:
                time.sleep(0.01)

    def set_native_servo(self, mcu_name, pin, angle):
        command = {{"cmd": "set_servo", "pin": int(pin), "value": int(angle)}}
        return self._send_command(mcu_name, command)

    def set_pca_pwm(self, mcu_name, channel, value):
        command = {{"cmd": "set_pwm", "pin": int(channel), "value": int(value)}}
        return self._send_command(mcu_name, command)


serial_ports_config = {serial_ports_literal}
robot_bridge = RobotHardwareBridge(serial_ports_config)


@app.get("/get_totalsense")
async def get_totalsense():
    with store_lock:
        return dict(store_sensory)


@app.post("/sensory_message")
async def sensory_message(request: Request):
    sense_message = await request.json()
    with store_lock:
        for key, value in sense_message.items():
            store_sensory[key] = value
        return dict(store_sensory)


@app.post("/motor_neuron")
async def motor_neuron(request: Request):
    motion_message = await request.json()
    mcu_name = motion_message.get("mcu_name")
    if not mcu_name:
        return {{"status": "Error", "message": "MCU name missing in request"}}

    if "servo" in motion_message:
        servo_cfg = motion_message.get("servo", {{}})
        pin = servo_cfg.get("pin")
        angle = servo_cfg.get("angle")
        if pin is None or angle is None:
            return {{"status": "Error", "message": "servo.pin and servo.angle are required"}}
        ok = robot_bridge.set_native_servo(mcu_name, pin, angle)
    elif "pwm" in motion_message:
        pwm_cfg = motion_message.get("pwm", {{}})
        channel = pwm_cfg.get("channel")
        value = pwm_cfg.get("value")
        if channel is None or value is None:
            return {{"status": "Error", "message": "pwm.channel and pwm.value are required"}}
        ok = robot_bridge.set_pca_pwm(mcu_name, channel, value)
    else:
        passthrough = motion_message.copy()
        passthrough.pop("mcu_name", None)
        ok = robot_bridge._send_command(mcu_name, passthrough)

    if ok:
        return {{"status": f"Command sent to {{mcu_name}}"}}
    return {{"status": "Error", "message": f"Failed to send command to {{mcu_name}}"}}


if __name__ == "__main__":
    uvicorn.run("middleware:app", host="0.0.0.0", port=8095, reload=True)
'''

    def build_fallback_vision_code(vision_component_name, node_id, config_filename, is_nav_cam):
      """Return a deterministic Python vision script when LLM output is invalid."""
      if is_nav_cam:
        return f'''import os
import cv2
import json
import time
import numpy as np
import requests

CONFIG_FILENAME = "{config_filename}"
VISION_COMPONENT_NAME = "{vision_component_name}"
SENSORY_URL = "http://127.0.0.1:8095/sensory_message"


def get_camera_index_from_config(config_file):
    try:
        with open(config_file, "r") as f:
            config = f.read().strip()
            try:
                config_json = json.loads(config)
                if isinstance(config_json, dict) and "camera_index" in config_json:
                    return int(config_json["camera_index"])
                if isinstance(config_json, int):
                    return config_json
                return int(config_json)
            except json.JSONDecodeError:
                return int(config)
    except (FileNotFoundError, ValueError):
        return 0


def post_telemetry(vision_component_name, data, feature_name, node_id=None):
    if feature_name == "VIO_camera":
        payload = {{vision_component_name + "_VIO_camera": data}}
    else:
        payload = {{vision_component_name + "_" + str(node_id) + "_" + feature_name: data}}
    try:
        requests.post(SENSORY_URL, json=payload, timeout=0.5)
    except Exception:
        pass


def run_vio(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[VIO] Cannot open camera")
        return

    ok, prev = cap.read()
    if not ok:
        print("[VIO] Failed initial frame")
        return

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=7)
    x = y = z = 0.0
    last_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_pts is None or len(prev_pts) < 16:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=7)
            prev_gray = gray
            continue

        curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        if curr_pts is None or st is None:
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=7)
            continue

        st = st.reshape(-1) == 1
        p0 = prev_pts[st].reshape(-1, 2)
        p1 = curr_pts[st].reshape(-1, 2)
        if len(p0) >= 8:
            flow = p1 - p0
            x += float(np.mean(flow[:, 0])) * 0.001
            y += float(np.mean(flow[:, 1])) * 0.001
            z += float(np.mean(np.linalg.norm(flow, axis=1))) * 0.0005
            now = time.time()
            dt = max(1e-6, now - last_t)
            last_t = now
            payload = {{
                "timestamp": now,
                "x": x,
                "y": y,
                "z": z,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "tracked_points": int(len(p1)),
                "cloud": [{{"x": float(px), "y": float(py), "z": z}} for px, py in p1[:80]],
            }}
            post_telemetry(VISION_COMPONENT_NAME, payload, "VIO_camera")
            print("fps: %.2f" % (1.0 / dt))

        prev_gray = gray
        prev_pts = p1.reshape(-1, 1, 2)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    camera_index = get_camera_index_from_config(config_path)
    run_vio(camera_index)
'''

      return f'''import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import json
import time
import requests
from ultralytics import YOLO

CONFIG_FILENAME = "{config_filename}"
VISION_COMPONENT_NAME = "{vision_component_name}"
NODE_ID = "{node_id}"
SENSORY_URL = "http://127.0.0.1:8095/sensory_message"


def get_camera_index_from_config(config_file):
    try:
        with open(config_file, "r") as f:
            config = f.read().strip()
            try:
                config_json = json.loads(config)
                if isinstance(config_json, dict) and "camera_index" in config_json:
                    return int(config_json["camera_index"])
                if isinstance(config_json, int):
                    return config_json
                return int(config_json)
            except json.JSONDecodeError:
                return int(config)
    except (FileNotFoundError, ValueError):
        return 0


def post_telemetry(vision_component_name, data, feature_name, node_id=None):
    if feature_name == "VIO_camera":
        payload = {{vision_component_name + "_VIO_camera": data}}
    else:
        payload = {{vision_component_name + "_" + str(node_id) + "_" + feature_name: data}}
    try:
        requests.post(SENSORY_URL, json=payload, timeout=0.5)
    except Exception:
        pass


def run_realtime_detection(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[OBJ] Cannot open camera")
        return

    model = YOLO("yolov8n.pt")
    last_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        detected_objects = {{}}
        results = model(frame, verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_idx = int(box.cls[0].item())
                label = model.names.get(cls_idx, str(cls_idx))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detected_objects[label] = {{"position": {{"x": cx, "y": cy}}}}

        post_telemetry(VISION_COMPONENT_NAME, detected_objects, "object", node_id=NODE_ID)
        now = time.time()
        print("fps: %.2f" % (1.0 / max(1e-6, now - last_t)))
        last_t = now


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    camera_index = get_camera_index_from_config(config_path)
    run_realtime_detection(camera_index)
'''
    

    def classify_component_with_gemini(component_name, email, project_name, SERVICE_URLS):
        """
        Classify the component using Gemini and the commu_mapnode keys as class options.
        Returns the class name as a string.
        """
        class_list = list(commu_mapnode.keys())
        prompt = (
          f"Classify the component '{component_name}' into one of the following classes: {class_list}. "
          "Return only the class name from the list."
        )
        try:
          class_name = ask_llm(email, project_name, prompt, SERVICE_URLS)
          if not class_name:
            raise ValueError("Empty classification response")
          return class_name.strip()
        except Exception as e:
          print(f"Error classifying component '{component_name}':", e)
          return None

    component_class_cache = {}

    def get_component_class(component_name, device_cat_cache):
        name_raw = str(component_name)
        cache_key = normalize_component_key(name_raw)
        if cache_key in component_class_cache:
            return component_class_cache[cache_key]

        clean_name = clean_component_name(name_raw)
        # Prefer cached server categories first for deterministic behavior.
        for k in (clean_name, normalize_component_key(clean_name), normalize_component_key(name_raw)):
            if k in device_cat_cache:
                component_class_cache[cache_key] = device_cat_cache[k]
                return device_cat_cache[k]

        ret_class = classify_component_with_gemini(name_raw, email, project_name, SERVICE_URLS)
        component_class_cache[cache_key] = ret_class
        return ret_class
    def semantic_processing(aiselectclass,comp):
         sem_resp = requests.post(
            f"{SERVICE_URLS['semantic_match']}/processing_part_match",
            json={email: {"project_name": project_name, 'command': aiselectclass, 'ref_data': comp}}
         )
         sem_json = sem_resp.json()
         compclassdat = sem_json.get('max_command')
         return compclassdat #Getting the return semantic selection data 
    #Mapping the list of the navigation 
    if email not in id_device_mapping:
        id_device_mapping[email] = {}
    if project_name not in id_device_mapping[email]:
        id_device_mapping[email][project_name] = {}

    for node_id in reqdat:
        # Ensure node_id is treated as a string for mapping consistency
        id_device_mapping[email][project_name][str(node_id)] = reqdat[node_id]['name']
        print(f"Mapped node {node_id} to {reqdat[node_id]['name']}")

    device_catcheck_global = {}
    try:
      cat_data = requests.get("http://192.168.50.247:9074/getdevice_cat").json()
      if email in cat_data and project_name in cat_data[email]:
        device_catcheck_global = cat_data[email][project_name]
    except Exception as e:
      print(f"Error requesting the device category cache: {e}")

    req_processor = requests.get("http://192.168.50.247:9060/total_selected").json()[email][project_name]
    #Checking if the system found the single board computer inside the list of the total_selected or not 
    if "Single_Board_computer" in list(req_processor) and "Microcontroller" in list(req_processor):
        #Generate the code in the case where the system detected single board computer
        sbcname = req_processor["Single_Board_computer"]  #Getting the single board computer from the selected list of the components by the email input of the project 
        print("Get current SBC name: ",sbcname) 
        # Find the SBC node ID dynamically instead of hardcoding '1'
        sbc_node_id_found = None
        for n_id, n_data in reqdat.items():
            if n_data['name'].startswith(sbcname):
                sbc_node_id_found = n_id
                break
        
        if sbc_node_id_found:
            print(f"Found SBC node ID: {sbc_node_id_found}")
            sbc_nodeid = reqdat[sbc_node_id_found]["inputs"]
        else:
            print(f"Warning: Could not find node ID for SBC {sbcname}, falling back to node '1'")
            sbc_nodeid = reqdat.get('1', {}).get('inputs', {})
        #print("Serial device and processing: ",sbc_nodeid)
        #Checking the device serial first 
        #for dev in sbc_nodeid:
        port_io  = commu_mapnode['SBC_node']['input']["Serial"] #Getting the list of the serial protocol device 
        print("List connection for serial: ",sbc_nodeid[port_io]) #Getting the list of the serial connection       
        #Getting the list of the port id connection 
        list_serialcom = sbc_nodeid[port_io]['connections'] #getting the list port io data 
        print("Getting the list port",list_serialcom)   
        store_mcusserial = {} #Store the mcus node inside the    
        store_sensorserial = {} #Store other serial sensors
        store_mainbridge_proc= {} #Store other sub components for sub processing chip part in the connection like I2C_mux ADC_mux Servo mux CAN-bus mux 
        for c_id in list_serialcom: 
             #print("Get node id connection: ",c_id) #Getting the serial connection   
             node_serial =c_id['node']  #Getting the node name  
             #print("Serial node name: ",node_serial) #Getting theserial data of the node
             #Getting the device name from the mapping data 
             node_id_str = str(node_serial)
             if node_id_str in id_device_mapping[email][project_name]:
                 dev_name = id_device_mapping[email][project_name][node_id_str]
                 print("Getting device name: ", dev_name)
                 # Clean dev_name by removing the last node ID number (suffix like _92)
                 dev_name_parts = str(dev_name).split("_")
                 if len(dev_name_parts) > 1:
                     dev_name = "_".join(dev_name_parts[:-1])
                 print("Cleaned device name for AI: ", dev_name)

             else:
                 print(f"Warning: Serial node ID {node_id_str} not found in id_device_mapping")
                 continue # Skip this connection if node is unknown
             #Checking the device name and category existing map data 
             device_catcheck = device_catcheck_global
             if dev_name not in device_catcheck:
                 #Use the semantic processing to get the components name and the device class components 
                 categ_list = list(req_processor) #Getting the list category of the components from the list of the components serial device connection 
                 update_status(email, project_name, f"Selecting semantic category for {dev_name}...")
                 #Use AI select from the list first to get the cateogory from the components selection 
                 aiagent_search = ask_llm(email, project_name, f"Select the category of the component {dev_name} from the list {categ_list} answer in single word selected from the list ", SERVICE_URLS)
                 print("AI selected category: ",aiagent_search)
                 semanticselect = semantic_processing(aiagent_search,categ_list) #Getting the semantic serial device connection from the list by semantically selection 
                 print("Semantic select: ",semanticselect) 
                 try:
                     reqcatstore = requests.post("http://192.168.50.247:9074/device_mappost",json={"email":email,"project_name":project_name,"category_payload":{dev_name:semanticselect}}) #Getting the category mapping with the device name data 
                     device_catcheck_global[dev_name] = semanticselect
                 except:
                     print("Category server post requets error") #Getting the category mapping error 
             else:
                 semanticselect = device_catcheck[dev_name]
                 print("Using cached category for: ", dev_name, " -> ", semanticselect)
             
             #Mapping the data to the server to store the category of the components existing categorization to reduce AI usage 
             #Mcus from total select detection for the connection list extraction 
             #Checking only the microcntroller node detection 
             if semanticselect == "Microcontroller":
                     #Getting the id from the microcontroller detected 
                     mcus_nodeconnect_in = reqdat[node_id_str].get('inputs', {})
                     mcus_nodeconnect_out = reqdat[node_id_str].get('outputs', {})
                     mcus_nodeconnect = {**mcus_nodeconnect_in, **mcus_nodeconnect_out}
                     print("mcus_nodeconnectlist: ",mcus_nodeconnect)
                     store_mcusserial[dev_name]  =  mcus_nodeconnect
             else:
                     print(f"Non-MCU Serial device detected: {dev_name} ({semanticselect})")
                     store_sensorserial[dev_name] = semanticselect
        
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Store Serial device (MCUs): ",store_mcusserial) #Getting the store mcus device connection list 
        print("Store Serial device (Sensors): ",store_sensorserial)
        #For loop each mcus to scanning for the list of the device connection from the node id 
        store_direct_terminal_proc = {} # To store terminal components directly connected to MCU
        for mcsr in list(store_mcusserial):  
                print(mcsr,store_mcusserial[mcsr])
                for mcyr in store_mcusserial[mcsr]:
                   if store_mcusserial[mcsr][mcyr]['connections'] != []:
                     print(mcsr,mcyr,store_mcusserial[mcsr][mcyr]['connections']) #Getting the list of the components connection inside data 
                     for list_connect in store_mcusserial[mcsr][mcyr]['connections']:
                                 target_node_id = str(list_connect['node'])
                                 if target_node_id in id_device_mapping[email][project_name]:
                                     print("Get the device name from node number", target_node_id, id_device_mapping[email][project_name][target_node_id])
                                     target_dev_name_raw = id_device_mapping[email][project_name][target_node_id]
                                     target_dev_name_parts = str(target_dev_name_raw).split("_")
                                     if len(target_dev_name_parts) > 1:
                                         target_dev_name = "_".join(target_dev_name_parts[:-1])
                                     else:
                                         target_dev_name = target_dev_name_raw
                                     
                                     device_catcheck = device_catcheck_global
                                     
                                     if target_dev_name not in device_catcheck or device_catcheck[target_dev_name] not in commu_mapnode.keys():
                                         categ_list = list(commu_mapnode.keys())
                                         aiagent_search = ask_llm(email, project_name, f"Select the category of the component {target_dev_name} from the list {categ_list} answer in single word selected from the list ", SERVICE_URLS)
                                         semanticselect_target = semantic_processing(aiagent_search,categ_list)
                                         # Cache the result regardless of what it is, so we never process it in AI search again
                                         try:
                                             reqcatstore = requests.post("http://192.168.50.247:9074/device_mappost",json={"email":email,"project_name":project_name,"category_payload":{target_dev_name:semanticselect_target}})
                                             device_catcheck_global[target_dev_name] = semanticselect_target
                                         except:
                                             print("Category server post requets error")
                                             
                                         if semanticselect_target in ["ADC mux","Servo_multiplexer", "I2C_mux", "I2C_multiplexer", "Flight_controller"]:
                                             print(f"Detected Component: {target_dev_name} -> {semanticselect_target}")
                                     else:
                                         semanticselect_target = device_catcheck[target_dev_name]
                                         print(f"Using cached Component: {target_dev_name} -> {semanticselect_target}")
                                     
                                     # Store component regardless of whether it was cached or freshly fetched
                                     if semanticselect_target in BRIDGE_COMPONENT_CLASSES:
                                         if semanticselect_target not in store_mainbridge_proc:
                                             print("Store the device mapping by the category of the sub node component detected")
                                             store_mainbridge_proc[semanticselect_target] = [] 
                                         if target_dev_name_raw not in store_mainbridge_proc[semanticselect_target]:
                                             store_mainbridge_proc[semanticselect_target].append(target_dev_name_raw)
                                     else:
                                         if semanticselect_target not in store_direct_terminal_proc:
                                             print("Store the direct terminal component to the MCU")
                                             store_direct_terminal_proc[semanticselect_target] = []
                                         if target_dev_name_raw not in store_direct_terminal_proc[semanticselect_target]:
                                             store_direct_terminal_proc[semanticselect_target].append(target_dev_name_raw)
                                 else:
                                     print(f"Warning: Node ID {target_node_id} not found in id_device_mapping")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
        print("Get mainbridge component protocol: ",store_mainbridge_proc)
        print("Get direct terminal components: ",store_direct_terminal_proc)
        store_terminal_connections = {}
        for mainbrige_ri in list(store_mainbridge_proc):
                      print(mainbrige_ri,store_mainbridge_proc[mainbrige_ri])
                      for m_cse in store_mainbridge_proc[mainbrige_ri]:
                              if m_cse not in store_terminal_connections:
                                  store_terminal_connections[m_cse] = []
                              print("Checking conneciton bridge inside ",m_cse) #getting the each main bridge connection list 
                              mc_len = len(str(m_cse).split("_")) #Getting the m_cse len for the components name 
                              idcomp = str(m_cse).split("_")[mc_len-1] 
                              print(m_cse," ID components :",idcomp)
                              print("Main bridge component: ",m_cse,reqdat[idcomp]) #Getting the main bridge components from the id input data 
                              
                              if 'inputs' in reqdat[idcomp]:
                                  for input_port in reqdat[idcomp]['inputs']:
                                      for conn in reqdat[idcomp]['inputs'][input_port]['connections']:
                                          conn_node_id = str(conn['node'])
                                          if conn_node_id in id_device_mapping[email][project_name]:
                                              conn_dev_name = id_device_mapping[email][project_name][conn_node_id]
                                              print(f"Main bridge {m_cse} Input connected to: Node {conn_node_id} -> {conn_dev_name}")
                                              raw_node_name = reqdat[conn_node_id]['name']
                                              print("Node components connected: ",raw_node_name)
                                              store_terminal_connections[m_cse].append(raw_node_name)
                                              
                              if 'outputs' in reqdat[idcomp]:
                                  for output_port in reqdat[idcomp]['outputs']:
                                      for conn in reqdat[idcomp]['outputs'][output_port]['connections']:
                                          conn_node_id = str(conn['node'])
                                          if conn_node_id in id_device_mapping[email][project_name]:
                                              conn_dev_name = id_device_mapping[email][project_name][conn_node_id]
                                              print(f"Main bridge {m_cse} Output connected to: Node {conn_node_id} -> {conn_dev_name}")
                                              raw_node_name = reqdat[conn_node_id]['name']
                                              print("Node components connected: ",raw_node_name)
                                              store_terminal_connections[m_cse].append(raw_node_name)
                        
                      print("---------------------------------------------------------------------------")
                      
        print("Terminal Connections Map (via Bridge): ", store_terminal_connections)
        
        mcus_all_connections = {}
        mcu_protocols = {**commu_mapnode['mcus_node']['input'], **commu_mapnode['mcus_node']['output']}
        rev_mcu_protocols = {v: k for k, v in mcu_protocols.items()}

        print("---------------------------------------------------------------------------------------------------------------------------")     
        print("Total Serial: ", store_mcusserial) 
        for mcu_dev in store_mcusserial.keys():
            print("----------------------------------------------------------------------------------------------------------------------")
            print("Mcus name: ", mcu_dev)
            print("Get list of the components connect on mcus:", store_mcusserial[mcu_dev])
            mcus_all_connections[mcu_dev] = {}
            for pins_comp in store_mcusserial[mcu_dev]:
                if store_mcusserial[mcu_dev][pins_comp]['connections'] != []:
                    protocol_name = rev_mcu_protocols.get(pins_comp, pins_comp)
                    if protocol_name not in mcus_all_connections[mcu_dev]:
                        mcus_all_connections[mcu_dev][protocol_name] = []
                    print("Get the list of the components connection inside data ", store_mcusserial[mcu_dev][pins_comp]['connections'])
                    for nodelist in store_mcusserial[mcu_dev][pins_comp]['connections']:
                        print("Get node list connection: ", nodelist)
                        target_node_id = str(nodelist['node'])
                        component_name = reqdat[target_node_id]['name'] if target_node_id in reqdat else 'UNKNOWN'
                        print("Component name for node id", target_node_id, ":", component_name)
                        retcomp_class = get_component_class(component_name, device_catcheck_global)
                        print("components class: ", retcomp_class)
                        
                        comp_info = {
                            "component_name": component_name,
                            "component_class": retcomp_class
                        }
                        
                        if retcomp_class in commu_mapnode:
                            pinsmapcomp = commu_mapnode[retcomp_class]
                            if "output" in pinsmapcomp:
                                print("Gettings pins map comp: ", list(pinsmapcomp["output"]))
                        
                        is_bridge_component = retcomp_class in BRIDGE_COMPONENT_CLASSES
                        is_i2c_bridge = retcomp_class in I2C_BRIDGE_CLASSES or "TCA9548A" in component_name.upper()
                        if is_bridge_component:
                          bridge_children = []
                          # Robust key matching for bridge node names that may vary by whitespace/newline formatting.
                          bridge_key_candidates = [
                            component_name,
                            normalize_component_key(component_name),
                          ]
                          for bridge_key in store_terminal_connections.keys():
                            if normalize_component_key(bridge_key) == normalize_component_key(component_name):
                              bridge_key_candidates.append(bridge_key)

                          for key_candidate in bridge_key_candidates:
                            children = store_terminal_connections.get(key_candidate, [])
                            for child in children:
                              if child not in bridge_children:
                                bridge_children.append(child)

                          if bridge_children:
                            comp_info["sub_devices"] = bridge_children
                          elif is_i2c_bridge:
                            print(f"[WARN] I2C bridge '{component_name}' has no extracted child nodes; check drawflow wiring and category mapping.")

                          if retcomp_class in commu_mapnode:
                            pinsmapcomp_tca = commu_mapnode[retcomp_class]
                            print("Gettings pins map comp: ", pinsmapcomp_tca)
                                
                        mcus_all_connections[mcu_dev][protocol_name].append(comp_info)
        print("\n--- Generating SBC Middleware Core Router ---")
        update_status(email, project_name, f"Generating SBC Core Bridge for {sbcname}...")
        
        sbc_prompt = f"Act as an expert Robotics Middleware Architect. Generate a production-ready, object-oriented Python backbone server for a single board computer ({sbcname}) using FastAPI.\n\n"
        sbc_prompt += "CRITICAL FRAMEWORK RULES:\n"
        sbc_prompt += "- This module must ONLY handle serial bus routing, actuator control, telemetry aggregation, and thread-safe control methods.\n"
        sbc_prompt += "- DO NOT write any computer vision, OpenCV, or YOLO code. Vision tasks will run in separate, independent system files and POST their results to this server.\n"
        sbc_prompt += "- The server must expose a POST endpoint `/sensory_message` that receives JSON payloads from vision and sensor modules, storing them in a global dictionary (e.g., `store_sensory`). Each key should represent a sensor type and id, and the value is the latest data.\n"
        sbc_prompt += "- The server must expose a GET endpoint `/get_totalsense` to return the current sensory data dictionary.\n"
        sbc_prompt += "- The server must expose a POST endpoint `/motor_neuron` that receives actuator/motor control commands as JSON. It must extract the target MCU name (e.g., `mcu_name`) and send the command to the correct MCU using a method like `RobotHardwareBridge._send_command`.\n"
        sbc_prompt += "- FIRMWARE SERIAL WRITE CONTRACT (MANDATORY): For servo control, send newline-delimited JSON exactly in this schema: {\"cmd\": \"set_servo\", \"pin\": <int>, \"value\": <int>}. Do NOT send alternate keys like `action` or `angle` to MCU firmware.\n"
        sbc_prompt += "- FIRMWARE SERIAL READ CONTRACT (MANDATORY): Implement a continuous background read loop per active MCU serial port (`while True`) that reads full lines, parses JSON telemetry frames emitted by firmware, and updates `store_sensory` automatically without requiring external POST calls.\n"
        sbc_prompt += "- Telemetry parsing rule: when firmware sends JSON like {\"type\":\"imu\",\"mpu_ax\":...,\"icm_gz\":...}, middleware must store it under a deterministic key such as `<mcu_name>_imu` (or `<mcu_name>_<type>`), and preserve latest frame values.\n"
        sbc_prompt += "- `/motor_neuron` must accept high-level payloads like {\"mcu_name\":\"...\",\"servo\":{\"pin\":X,\"angle\":Y}} and translate them into the firmware contract JSON (`cmd/set_servo/pin/value`) before serial write.\n"
        sbc_prompt += "- All serial reads and writes must be lock-protected and thread-safe; avoid blocking request handlers on long serial waits.\n"
        sbc_prompt += "- The actuator control logic must support serial communication with MCUs (using pyserial), and provide clean methods for servo and PWM control. The baudrate for each MCU must match the value used in its firmware (e.g., detected from Serial.begin in the corresponding .ino file).\n"
        sbc_prompt += "- All serial communication must be robust, non-blocking, and thread-safe.\n"
        sbc_prompt += "- Include baseline dependencies (`fastapi`, `uvicorn`, `serial`, `json`, `threading`, `time`).\n"
        sbc_prompt += "- Encapsulate all hardware logic inside a primary class named `RobotHardwareBridge`.\n"
        sbc_prompt += "- Expose clean setter methods (`set_native_servo(mcu_name, pin, angle)`, `set_pca_pwm(mcu_name, channel, value)`) that format commands cleanly into lines and flush them down the active serial ports immediately.\n"
        sbc_prompt += "- Output ONLY the full compileable Python code block within a single markdown code block with no prefaces or conversational notes.\n"
        sbc_prompt += "- The overall structure and endpoints should closely follow this example (but with actuator/serial logic in /motor_neuron):"\
        """\n\nimport os\nimport time\nimport json\nimport asyncio\nfrom fastapi import FastAPI, Request\nimport uvicorn\nimport serial\n\napp = FastAPI()\nstore_sensory = {}\n\n@app.get('/get_totalsense')\nasync def get_totalsense():\n    return store_sensory\n\n@app.post('/sensory_message')\nasync def sensory_message(request: Request):\n    sense_message = await request.json()\n    for key, value in sense_message.items():\n        store_sensory[key] = value\n    return store_sensory\n\n@app.post('/motor_neuron')\nasync def motor_neuron(request: Request):\n    motion_message = await request.json()\n    # Here, send the command to the correct MCU using RobotHardwareBridge\n    # Example: robot_bridge._send_command(motion_message['mcu_name'], motion_message)\n    return {'status': 'Command sent'}\n\nif __name__ == '__main__':\n    uvicorn.run('middleware:app', host='0.0.0.0', port=8095)\n"""
        
        # Dynamically extract baudrate from firmware files for each MCU
        mcu_baudrates = {}
        firmware_dir = os.path.join(get_project_dir(email, project_name), "firmware")
        if os.path.exists(firmware_dir):
          for fname in os.listdir(firmware_dir):
            if fname.endswith(".ino"):
              mcu_name = fname.split(".")[0]
              with open(os.path.join(firmware_dir, fname), "r") as f:
                content = f.read()
                match = re.search(r"Serial.begin\((\d+)\)", content)
                if match:
                  mcu_baudrates[mcu_name] = int(match.group(1))
        print("[MCU BAUDRATES]", mcu_baudrates)

        # Pass baudrate info to the prompt if found
        if mcu_baudrates:
          baudrate_info = "\n".join([f"- {k}: {v}" for k, v in mcu_baudrates.items()])
          sbc_prompt += f"- The following MCUs and their required baudrates were detected from firmware:\n{baudrate_info}\n"
        try:
          middleware_dir = os.path.join(get_project_dir(email, project_name), "middleware")
          os.makedirs(middleware_dir, exist_ok=True)

          sbc_code_extracted = ""
          middleware_error_reason = ""
          for attempt in range(3):
            try:
              sbc_text = ask_llm(email, project_name, sbc_prompt, SERVICE_URLS)
              candidate_code = extract_code(sbc_text)
              if is_valid_python_middleware_code(candidate_code):
                sbc_code_extracted = candidate_code
                break
              middleware_error_reason = "Gemini returned invalid middleware content"
            except Exception as attempt_error:
              middleware_error_reason = str(attempt_error)

            print(f"[WARN] Middleware generation attempt {attempt + 1}/3 failed: {middleware_error_reason}")

          if not sbc_code_extracted:
            print(f"[WARN] Falling back to deterministic local middleware template: {middleware_error_reason}")
            sbc_code_extracted = build_fallback_middleware_code(mcu_baudrates)

          with open(os.path.join(middleware_dir, "middleware.py"), "w") as f:
            f.write(sbc_code_extracted)
          print(f"SBC Core Middleware saved to {middleware_dir}/middleware.py")
        except Exception as e:
          print(f"Error generating SBC Core Middleware: {e}")

        # ==================================================================================
        # PLACE THE MULTI-CAMERA VISION GENERATION BLOCK HERE (OUTSIDE THE TRY-EXCEPT ABOVE)
        # ==================================================================================
        print("\n--- Scanning for Vision System Modules ---")
        
        # Extract individual vision devices by checking class first, then deterministic name fallback.
        vision_devices = []
        vision_class_debug = {}
        vision_name_keywords = [
          "camera", "vision", "webcam", "realsense", "stereo", "depth", "csi", "usb_cam", "flir", "fisheye", "eye"
        ]
        # Use cached class map first for deterministic behavior, then fallback by name, then Gemini.
        for node_id, node_data in reqdat.items():
          node_name = node_data.get('name', '')
          if not node_name:
            continue
          try:
            comp_class = resolve_device_class_from_cache(node_name, device_class_map)
            if not comp_class and any(keyword in node_name.lower() for keyword in vision_name_keywords):
              comp_class = "camera_fallback"
            if not comp_class:
              comp_class = classify_component_with_gemini(node_name, email, project_name, SERVICE_URLS)
            vision_class_debug[node_name] = comp_class
            if comp_class and ("camera" in comp_class.lower() or "vision" in comp_class.lower()):
              vision_devices.append((node_id, node_data))
          except Exception as e:
            print(f"Error classifying node {node_name}: {e}")

        # Remove accidental duplicates while preserving order.
        dedup_vision_devices = []
        seen_vision_ids = set()
        for vision_device in vision_devices:
          if vision_device[0] not in seen_vision_ids:
            dedup_vision_devices.append(vision_device)
            seen_vision_ids.add(vision_device[0])
        vision_devices = dedup_vision_devices

        print(f"Identified Visual Hardware Assets for Multi-Agent Task Routing (by class): {[nd[1].get('name','') for nd in vision_devices]}")
        if not vision_devices:
          print(f"No vision devices found from classification map: {vision_class_debug}")

        # In multi-camera systems, designate exactly one camera as the VIO source.
        vision_count = len(vision_devices)
        robot_description_str_global = robot_description if isinstance(robot_description, str) else str(robot_description)
        selected_vio_node_id = None
        if vision_count > 1:
          nav_keywords = ["nav", "navigation", "positioning", "slam", "vio", "odometry"]
          for candidate_node_id, candidate_node_data in vision_devices:
            candidate_name = candidate_node_data.get('name', '')
            candidate_context = f"{candidate_name} {robot_description_str_global}".lower()
            if any(k in candidate_context for k in nav_keywords):
              selected_vio_node_id = candidate_node_id
              break
          if selected_vio_node_id is None:
            selected_vio_node_id = vision_devices[0][0]
          print(f"Selected VIO camera node in multi-camera setup: {selected_vio_node_id}")

        # Iterate through every detected vision hardware module separately (by class)
        for cam_idx, (node_id, node_data) in enumerate(vision_devices):
          node_name = node_data.get('name', '')
          vision_component_name = node_name
          safe_component = "".join([c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in vision_component_name])
          print(f"Generating specialized, independent vision script for asset: {vision_component_name} (Node ID: {node_id})")
          update_status(email, project_name, f"Generating Vision pipeline for {vision_component_name}...")

          # Determine functional baseline target categories based on name and description requirements
          robot_description_str = robot_description if isinstance(robot_description, str) else str(robot_description)
          nav_keywords = ["nav", "navigation", "positioning", "slam", "vio", "odometry"]
          has_nav_name_hint = any(k in vision_component_name.lower() for k in nav_keywords)
          nav_required_by_system = any(k in robot_description_str.lower() for k in nav_keywords)
          requires_yolo = any(k in robot_description_str.lower() for k in ["object", "detect", "track", "yolo", "person", "sorting", "autonomous navigation"])
          is_selected_vio_camera = vision_count > 1 and node_id == selected_vio_node_id

          # Restore deterministic VIO behavior:
          # - Multi-camera: exactly one selected VIO camera.
          # - Single-camera: allow VIO when nav is required or camera name indicates nav usage.
          if vision_count > 1:
            is_nav_cam = is_selected_vio_camera
          else:
            is_nav_cam = has_nav_name_hint or nav_required_by_system

          # VIO cameras must stay pure VIO (no YOLO/object-detection section mixed in).
          should_add_yolo_section = requires_yolo and not is_selected_vio_camera and not is_nav_cam

          # --- Camera index config addition ---
          config_filename = f"{safe_component}_{node_id}_camera.conf"
          vision_prompt = f"Act as an expert Embedded Vision and Robotics AI Engineer. Generate a standalone, production-grade Python script for the visual system hardware asset named '{vision_component_name}'.\n"
          vision_prompt += "CRITICAL SYSTEM OPERATION STATE: This script operates as a decoupled, isolated hardware process. It must run completely independent of any serial bridge bottlenecks.\n\n"

          vision_prompt += "### Camera Index Configuration Requirement:\n"
          vision_prompt += f"- The script MUST read the camera index from a config file named '{config_filename}' (plain text integer or JSON with a 'camera_index' key). Use this robust Python pattern to load the index:\n"
          vision_prompt += f"- REQUIRED MODULE CONSTANT: define `CONFIG_FILENAME = \"{config_filename}\"` near the top of the script and use this constant for all config-path construction.\n"
          vision_prompt += "- REQUIRED STARTUP FLOW (MANDATORY FOR EVERY CAMERA SCRIPT): In `if __name__ == \"__main__\":`, build `config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)`, load index via `get_camera_index_from_config(config_path)`, fall back to camera scan only if config is invalid/missing, then pass the resolved index into the runtime pipeline.\n"
          vision_prompt += "- The runtime pipeline must accept `camera_index` as an explicit argument (for example `run_realtime_detection(camera_index)`) and must not silently replace it with another random index when a valid config index is provided.\n"
          vision_prompt += "```python\ndef get_camera_index_from_config(config_file):\n    try:\n        with open(config_file, 'r') as f:\n            config = f.read().strip()\n            try:\n                config_json = json.loads(config)\n                if isinstance(config_json, dict) and 'camera_index' in config_json:\n                    return int(config_json['camera_index'])\n                elif isinstance(config_json, int):\n                    return config_json\n                else:\n                    return int(config_json)\n            except json.JSONDecodeError:\n                return int(config)\n    except (FileNotFoundError, ValueError):\n        return 0\n```\n- The script MUST use this function to get the camera index, and never call .get on an int. If missing/invalid, default to 0.\n"
          vision_prompt += "- If the config file is missing or invalid, the script must fall back to auto-detecting the camera index by scanning available devices.\n"
          vision_prompt += "- The config file must be located in the same directory as the script.\n\n"

          vision_prompt += "### Network Architecture & Telemetry Reporting Constraints:\n"
          vision_prompt += "- The processing loop must run an internal initialization routine that uses the configured or detected camera index for cv2.VideoCapture.\n"
          vision_prompt += "- HEADLESS EXECUTION REQUIREMENT (MANDATORY): this script must run without any GUI display. Do NOT call `cv2.imshow`, `cv2.namedWindow`, `cv2.waitKey`, `cv2.resizeWindow`, or any other UI/window APIs.\n"
          vision_prompt += "- STRICT FORBIDDEN CALLS: generated source MUST NOT contain `imshow(`, `namedWindow(`, `waitKey(`, `destroyAllWindows(`, `startWindowThread(`, or any keyboard-interactive display loop.\n"
          vision_prompt += "- Logging-only runtime rule: print FPS/diagnostics to console and send telemetry over HTTP; do not render preview windows.\n"
          vision_prompt += "- TELEMETRY ENDPOINT EXPORT: After extracting tracking updates, spatial coordinates, or class confidence metrics, you MUST seamlessly format observations into JSON envelopes and post them over HTTP network requests to the backbone data collection endpoint: `http://127.0.0.1:8095/sensory_message`.\n"
          vision_prompt += "- TELEMETRY FUNCTION DESIGN (MANDATORY): Implement ONE unified function named `post_telemetry(vision_component_name, data, feature_name, node_id=None)` and use it for all telemetry categories, including VIO and object detection. Do NOT create a separate `post_vio_telemetry` function.\n"
          vision_prompt += "- Unified key rule inside that single function: if `feature_name == \"VIO_camera\"`, key must be `f'{vision_component_name}_VIO_camera'`; otherwise key must be `f'{vision_component_name}_{node_id}_{feature_name}'` when node_id is provided.\n"
          vision_prompt += "- For non-VIO telemetry, the POST payload key MUST use '<vision_component_name>_<node_id>_<function>', for example: {\"USB_camera_5MPX.glb_92_object\": {\"keyboard\": {\"x\": 45, \"y\": 67}}}.\n"
          vision_prompt += "- The key must be dynamically constructed in the code using the sanitized vision component name, the node id, and the function name (such as 'object', 'slam', etc.), always separated by underscores.\n\n"

          if is_selected_vio_camera:
            vision_prompt += "### Multi-Camera VIO Selection Requirement:\n"
            vision_prompt += f"- This project has {vision_count} camera inputs and THIS camera is the designated VIO camera.\n"
            vision_prompt += f"- For VIO telemetry posted to /sensory_message, preserve the same payload envelope style but the top-level key MUST be exactly: '{{\"{vision_component_name}_VIO_camera\": {{<VIO payload here>}}}}'.\n"
            vision_prompt += "- The value of that key must contain the VIO payload from this script (pose/odometry/feature-tracking data).\n"
            vision_prompt += "- The VIO telemetry call site MUST invoke the same unified `post_telemetry(...)` function using `feature_name='VIO_camera'`.\n"
            vision_prompt += "- Keep posting non-VIO data with existing key rules; only VIO payload must use '<Camera_name>_VIO_camera'.\n\n"

          if is_nav_cam:
            vision_prompt += "### Algorithmic Directive: VISUAL INERTIAL ODOMETRY & OPTIC FLOW TRACKING\n"
            vision_prompt += "- Because this visual sensor is mapped to handle high-accuracy machine localization and navigation parameters, you MUST construct a custom feature tracker.\n"
            vision_prompt += "- Implement spatial translation modeling utilizing custom OpenCV corner detection pipelines (such as `cv2.goodFeaturesToTrack` or `cv2.ORB_create`) combined with sparse optical flow state arrays (`cv2.calcOpticalFlowPyrLK`).\n"
            vision_prompt += "- Transform tracking anomalies and vector displacement offsets into localized coordinate updates maps (tracking absolute position markers: x, y, z data fields along with frame-to-frame delta steps).\n"
            vision_prompt += "- VIO-ONLY REQUIREMENT (MANDATORY): Do NOT import or use YOLO/ultralytics in this script. Do NOT generate object detection payloads for this camera.\n"
            vision_prompt += "- HEADLESS RULE (MANDATORY): Do not include any OpenCV UI calls (`cv2.imshow`, `cv2.namedWindow`, `cv2.waitKey`) in this VIO script.\n"
            vision_prompt += "- Build one `CameraOnlyVO`-style pipeline and post ONLY VIO telemetry for this camera to `/sensory_message`.\n"
            vision_prompt += "- The VIO payload value must include at minimum: `timestamp`, `x`, `y`, `z`, `rotation`, `tracked_points`, and `cloud` (list of `{x,y,z}`).\n"
            vision_prompt += "- Use robust scalar extraction for translation vectors (e.g., if using a `(3,1)` vector, convert via `[0,0]`, `[1,0]`, `[2,0]`) so runtime scalar-conversion errors cannot occur.\n"
            vision_prompt += "- BOOTSTRAP REQUIREMENT (MANDATORY): implement a `__main__` block that builds the config path with `os.path.join(os.path.dirname(os.path.abspath(__file__)), \"" + config_filename + "\")`, loads camera index through `get_camera_index_from_config`, then runs `CameraOnlyVO(camera_index=camera_index)` exactly in that order.\n"
            vision_prompt += "- The script must remain compatible with the camera_vo.py execution style: a `CameraOnlyVO` class with a `run()` loop and explicit camera open/check handling.\n"
            try:
              vo_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_vo.py")
              with open(vo_template_path, "r") as vo_file:
                vo_snippet = vo_file.read()
              vision_prompt += f"Adhere strictly to this structured platform orientation, array translation matrix, and coordinate wrapping logic pattern for your odometry tracking engine structures:\n```python\n{vo_snippet}\n```\n\n"
            except Exception as e:
              print(f"Vision generator script helper couldn't reach camera_vo.py style library: {e}")

          if should_add_yolo_section:
            vision_prompt += "### Algorithmic Directive: DEEP LEARNING INFERENCE REAL-TIME OBJECT DETECTION\n"
            vision_prompt += "- LEGACY COMPATIBILITY (MANDATORY): Preserve the exact legacy object-detection telemetry schema/keys and detection semantics only; DO NOT preserve any legacy UI/window behavior.\n"
            vision_prompt += "- The script MUST use the following structure and imports at the top (do not omit):\n"
            vision_prompt += "```python\nimport os\nos.environ[\"CUDA_VISIBLE_DEVICES\"] = \"\"\nimport cv2\nimport requests\nfrom ultralytics import YOLO\nimport time\n```\n"
            vision_prompt += "- The script MUST read the camera index from the config file (plain integer or JSON with 'camera_index'). Use this robust Python pattern to load the index:\n"
            vision_prompt += "```python\ndef get_camera_index_from_config(config_file):\n    try:\n        with open(config_file, 'r') as f:\n            config = f.read().strip()\n            try:\n                config_json = json.loads(config)\n                if isinstance(config_json, dict) and 'camera_index' in config_json:\n                    return int(config_json['camera_index'])\n                elif isinstance(config_json, int):\n                    return config_json\n                else:\n                    return int(config_json)\n            except json.JSONDecodeError:\n                return int(config)\n    except (FileNotFoundError, ValueError):\n        return 0\n```\n- The script MUST use this function to get the camera index, and never call .get on an int. If missing/invalid, default to 0.\n"
            vision_prompt += f"- In this object-detection script, define `CONFIG_FILENAME = \"{config_filename}\"` and in `__main__` load camera index from `os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)` before starting detection.\n"
            vision_prompt += "- The script MUST load the YOLOv8 Nano model ('yolov8n.pt') and force CPU usage.\n"
            vision_prompt += "- The script MUST be headless: do NOT use cv2.imshow, cv2.namedWindow, cv2.waitKey, cv2.resizeWindow, or any GUI display functions.\n"
            vision_prompt += "- Forbidden UI symbols in final code: `imshow(`, `namedWindow(`, `waitKey(`, `destroyAllWindows(`. If any appears, rewrite the code before returning output.\n"
            vision_prompt += "- The main loop MUST: read frame, run YOLO inference, compute FPS, and always print FPS as 'fps: <value>' to the console.\n"
            vision_prompt += "- For each detection, detected_objects[label] = {\"position\": {\"x\": x, \"y\": y}} where x, y are the center of the bounding box.\n"
            vision_prompt += "- When drawing bounding boxes for detected objects, use a different color for each object/class (multi-color rectangles), as in the original yolov8_realtime.py. You may use a color palette or generate colors based on the class index.\n"
            vision_prompt += "- The object telemetry payload key MUST remain exactly `f'{vision_component_name}_{node_id}_object'` via the unified `post_telemetry(...)` function (`feature_name='object'`, `node_id=node_id`).\n"
            vision_prompt += "- The object telemetry value MUST remain the same legacy schema (`detected_objects` dictionary) and must not be renamed, nested, or wrapped in an additional envelope.\n"
            vision_prompt += "- For object detection telemetry, the script MUST call the same unified `post_telemetry(...)` function using `feature_name='object'` and `node_id=node_id`.\n"
            vision_prompt += "- The script MUST be robust, ready-to-run, and require no manual edits after generation.\n"
            vision_prompt += "- The output MUST be a single, complete, compileable Python script in a markdown code block, with no extra explanations.\n"

          vision_prompt += f"### Custom Application Profile Feature Tailoring:\n"
          vision_prompt += f"- Review this high-level functional system objective statement: '{robot_description}'.\n"
          vision_prompt += f"- Inspect and customize edge algorithms specifically to match this workflow layout. If thermal cameras or multi-spectral names are found (like FLIR), ensure appropriate visual filters or adaptive lookup structures are initialized. If specific threshold operations or masking algorithms are highlighted by the statement text, append their dedicated helper matrix operations using native OpenCV processing options cleanly.\n\n"

          vision_prompt += "### Output Structural Code Requirements:\n"
          vision_prompt += "- Include all required native system libraries cleanly (`cv2`, `requests`, `json`, `time`, `numpy`).\n"
          vision_prompt += "- HEADLESS VALIDATION RULE: before final output, ensure the script source does not contain `imshow(`, `namedWindow(`, `waitKey(`, or other UI-window function calls.\n"
          vision_prompt += "- Package execution frames inside a clean, robust primary script layout with defensive error isolation wrappers handling temporary data frame collection failures.\n"
          vision_prompt += "- Output ONLY the clean, fully realized, compileable Python tracking pipeline within a single markdown script block without adding conversational introductions or summary explanations before or after the code frame block."

          try:
            vision_code_extracted = ""
            last_generation_error = None
            for attempt in range(1, 4):
              retry_suffix = ""
              if attempt > 1:
                retry_suffix = (
                  "\n\nCRITICAL RETRY REQUIREMENT: Previous output was invalid. "
                  "Return ONLY Python vision code. Do NOT output Arduino/C++ firmware. "
                  "Do NOT include any '#include', 'void setup()', or 'void loop()'."
                )

              vision_code_response = ask_llm(email, project_name, vision_prompt + retry_suffix, SERVICE_URLS)
              if not vision_code_response:
                last_generation_error = ValueError("Empty/unsupported Gemini response for vision generation")
                continue

              candidate_code = extract_code(vision_code_response)
              if not candidate_code.strip():
                last_generation_error = ValueError("No extractable code block returned for vision generation")
                continue

              if looks_like_non_python_firmware(candidate_code):
                last_generation_error = ValueError("Gemini returned Arduino/C++ firmware instead of Python vision code")
                continue

              if not is_valid_python_vision_code(candidate_code):
                last_generation_error = ValueError("Gemini returned non-vision or malformed Python for vision generation")
                continue

              vision_code_extracted = candidate_code
              break

            if not vision_code_extracted:
              fallback_reason = str(last_generation_error or "unknown")
              print(f"[WARN] Falling back to deterministic local vision template for {vision_component_name}: {fallback_reason}")
              vision_code_extracted = build_fallback_vision_code(
                vision_component_name=vision_component_name,
                node_id=node_id,
                config_filename=config_filename,
                is_nav_cam=is_nav_cam,
              )
            # Sanitize the component name and node id for filename
            if is_nav_cam:
              camera_function_suffix = "vio"
            elif requires_yolo:
              camera_function_suffix = "object_detect"
            else:
              camera_function_suffix = "generic"
            safe_filename = f"{safe_component}_{node_id}_vision_{camera_function_suffix}.py"
            middleware_dir = os.path.join(get_project_dir(email, project_name), "middleware")
            os.makedirs(middleware_dir, exist_ok=True)
            target_vision_path = os.path.join(middleware_dir, safe_filename)
            with open(target_vision_path, "w") as f:
              f.write(vision_code_extracted)
            # --- Write camera index based on vision system count ---
            config_path = os.path.join(middleware_dir, config_filename)
            with open(config_path, "w") as conf_f:
              conf_f.write(f"{cam_idx}\n")
            print(f"Independent Vision System saved successfully to: {target_vision_path}")
            print(f"Camera config (index {cam_idx}) saved to: {config_path}")
          except Exception as e:
            middleware_dir = os.path.join(get_project_dir(email, project_name), "middleware")
            os.makedirs(middleware_dir, exist_ok=True)
            failure_name = f"{safe_component}_{node_id}_vision_generation_error.log"
            failure_path = os.path.join(middleware_dir, failure_name)
            try:
              with open(failure_path, "w") as ferr:
                ferr.write(f"Vision code generation failed for node_id={node_id}, component={vision_component_name}\n")
                ferr.write(f"Error: {e}\n\n")
                ferr.write("---- Prompt ----\n")
                ferr.write(vision_prompt)
              print(f"Saved vision generation failure log to: {failure_path}")
            except Exception as write_err:
              print(f"Additionally failed writing generation failure log: {write_err}")
            print(f"Fatal Exception encountered while exporting independent vision system asset for {vision_component_name} (Node ID: {node_id}): {e}")
        # ==================================================================================

        # 2. GENERATE INSTALLER SCRIPTS WITH HARDWARE DETECTION FOR FRESH OS
        installer_dir = os.path.join(get_project_dir(email, project_name), "Installer")
        os.makedirs(installer_dir, exist_ok=True)
        installer_script_path = os.path.join(installer_dir, "install_dependencies.sh")
        bash_script = """#!/bin/bash
    echo "Starting Installation..."
    echo "Updating system and installing base dependencies..."
    sudo apt-get update
    sudo apt-get install -y curl git build-essential python3 python3-pip python3-venv python3-opencv libasound2-dev portaudio19-dev sox libsox-fmt-all

    echo "Configuring Audio Peripheral Permissions..."
    # Ensure current system user belongs to the native audio group for PyAudio/Vosk access
    if ! groups "$USER" | grep -q "\baudio\b"; then
        echo "⚙️ Adding user $USER to the 'audio' hardware access group..."
        sudo usermod -aG audio "$USER"
        echo "⚠️ Note: You will need to log out and log back in (or restart your terminal session) for group permission updates to apply!"
    else
        echo "ℹ️ User $USER is already a member of the 'audio' group."
    fi

    echo "Detecting System Specifications..."
    # Detect Memory
    TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
    echo "Total Memory: ${TOTAL_MEM} MB"

    # Detect Python Version and verify compatibility (Ultralytics requires Python >= 3.8)
    echo "Checking Python Version Compatibility..."
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is not installed on this system."
        exit 1
    fi
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    echo "Detected Python: ${PYTHON_MAJOR}.${PYTHON_MINOR}"
    if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 8 ]; then
        echo "Error: Python 3.8 or higher is required to run the generated middleware libraries (ultralytics/YOLOv8)."
        echo "Please upgrade your Python installation and try again."
        exit 1
    fi

    # Detect SBC or Laptop
    IS_SBC=0
    if [ -f /sys/firmware/devicetree/base/model ]; then
        MODEL=$(tr -d '\\0' < /sys/firmware/devicetree/base/model)
        echo "Detected SBC Model: $MODEL"
        IS_SBC=1
    else
        echo "Generic PC/Laptop detected."
    fi

    # CPU Architecture
    ARCH=$(uname -m)
    echo "Architecture: $ARCH"

    # Installing/Updating Rust Language Compiler
    echo "Checking Rust Compiler..."
    if command -v rustc &> /dev/null; then
        echo "Rust is already installed. Checking for updates..."
        if command -v rustup &> /dev/null; then
            rustup update stable
        else
            echo "rustup not found. Attempting to update rustc and cargo via apt..."
            sudo apt-get install -y --only-upgrade rustc cargo
        fi
    else
        echo "Rust is not installed. Installing Rust via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi
    fi

    # Verify Rust compiler setup and fallback if needed
    if command -v rustc &> /dev/null; then
        echo "Rust compiler verified successfully:"
        rustc --version
        cargo --version
    else
        echo "rustup environment not loaded. Attempting fallback installation via apt..."
        sudo apt-get install -y rustc cargo
        if command -v rustc &> /dev/null; then
            echo "Rust compiler installed via apt:"
            rustc --version
        else
            echo "Warning: Rust compiler could not be set up. Some libraries requiring Rust compilation might fail to install."
        fi
    fi

    # Arduino CLI Ecosystem Installation Layer
    echo "Checking Arduino CLI Binary Suite installation..."
    if ! command -v arduino-cli &> /dev/null; then
        echo "⚙️ arduino-cli is not present. Executing official installation layer..."
        curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
        
        # Check and handle direct local binary directory output fallback
        if [ -d "./bin" ]; then
            sudo cp ./bin/arduino-cli /usr/local/bin/
            rm -rf ./bin
        fi
        
        if ! command -v arduino-cli &> /dev/null; then
            echo "❌ Error: Failed to install arduino-cli or bind it into path environment variables." >&2
            exit 1
        fi
    else
        echo "ℹ️ arduino-cli binary ecosystem is already installed and accessible."
    fi

    # STM32duino Toolchain Core Platform Setup
    BOARD_URL="https://github.com/stm32duino/BoardManagerFiles/raw/main/package_stmicroelectronics_index.json"
    CORE_ID="STMicroelectronics:stm32"

    echo "⚙️ Configuring Arduino CLI Board Manager URL for STM32duino..."
    # Ensure configuration structure is clean and available before tracking URLs
    arduino-cli config init --overwrite || true

    if arduino-cli config dump | grep -q "$BOARD_URL"; then
        echo "ℹ️ STM32duino URL is already present in configuration."
    else
        arduino-cli config add board_manager.additional_urls "$BOARD_URL"
        echo "✅ Successfully added STM32duino URL to configuration."
    fi

    echo "🔄 Updating platform package indexes..."
    arduino-cli core update-index

    echo "📥 Installing $CORE_ID hardware architecture toolchain..."
    arduino-cli core install "$CORE_ID"
    echo "✅ Toolchain setup complete for STM32 hardware abstraction layers."

    # Detect memory requirements specifically before Ollama installation
    echo "Checking memory requirements for Ollama..."
    if [ -n "$TOTAL_MEM" ] && [ "$TOTAL_MEM" -eq "$TOTAL_MEM" ] 2>/dev/null; then
        if [ "$TOTAL_MEM" -lt 4000 ]; then
            echo "Warning: System memory is ${TOTAL_MEM} MB (less than 4GB)."
            echo "Ollama and its LLM models run best with at least 4GB of RAM."
            echo "Proceeding with Ollama installation, but performance may be slow or unstable."
        else
            echo "Memory check passed: ${TOTAL_MEM} MB is sufficient for Ollama."
        fi
    else
        echo "Warning: Could not determine total memory accurately. Proceeding anyway."
    fi

    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sudo sh

    echo "Creating Python Virtual Environment (venv)..."
    # Create venv at the project root directory (one level up from Installer folder)
    python3 -m venv "$(dirname "$0")/../venv"

    # Create a custom temporary directory for pip to avoid running out of space in /tmp (common on SBCs)
    PIP_TMP_DIR="$(dirname "$0")/../pip_tmp"
    mkdir -p "$PIP_TMP_DIR"
    export TMPDIR="$PIP_TMP_DIR"

    echo "Upgrading pip inside virtual environment..."
    "$(dirname "$0")/../venv/bin/pip" install --upgrade pip

    echo "Installing YOLOv8, OpenCV, FastAPI, Uvicorn, and Audio Core (Vosk, PyAudio)..."
    "$(dirname "$0")/../venv/bin/pip" install ultralytics opencv-python fastapi uvicorn vosk pyaudio

    # Clean up custom pip temp dir
    rm -rf "$PIP_TMP_DIR"
    unset TMPDIR

    echo "Creating startup runner script (run.sh)..."
    RUN_SCRIPT_PATH="$(dirname "$0")/../run.sh"
    cat << 'EOF' > "$RUN_SCRIPT_PATH"
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo "Error: Virtual environment (venv) directory not found at ${SCRIPT_DIR}/venv."
    echo "Please run the Installer/install_dependencies.sh script first."
    exit 1
fi
echo "Activating virtual environment and starting middleware..."
source "${SCRIPT_DIR}/venv/bin/activate"
python3 "${SCRIPT_DIR}/middleware/middleware.py"
EOF
    chmod +x "$RUN_SCRIPT_PATH"
    echo "Runner script created at: $RUN_SCRIPT_PATH"

    echo "Installation complete!"
    echo "To run your middleware within the virtual environment, execute:"
    echo "  python3 \"$(dirname "$0")/../middleware/middleware.py\" using the virtual env python:"
    echo "  \"$(dirname "$0")/../venv/bin/python3\" \"$(dirname "$0")/../middleware/middleware.py\""
    echo "Or run the startup script directly:"
    echo "  \"$(dirname "$0")/../run.sh\""
    """
    try:
      with open(installer_script_path, "w") as f:
        f.write(bash_script)
      # Make the script executable
      os.chmod(installer_script_path, 0o755)
      print(f"SBC Dependencies Installer saved to {installer_script_path}")
    except Exception as e:
      print("Error generating SBC Middleware:", e)

    print("\n--- Generating MCU Firmware ---")
    for mcu_dev in store_mcusserial.keys():
      print("----------------------------------------------------------------------------------------------------------------------")
      print("MCU Hardware Node Name:", mcu_dev)

      required_library_queries = build_dynamic_library_queries(
        mcu_dev,
        [
          store_mcusserial.get(mcu_dev, {}),
          store_terminal_connections,
          store_mainbridge_proc,
          store_direct_terminal_proc
        ],
        device_class_map
      )
      lib_check = check_arduino_library_availability(required_library_queries)
      print(f"[ARDUINO LIB CHECK] {mcu_dev} available={lib_check['available']} missing={lib_check['missing']} cli={lib_check['cli_available']}")

      # DEFENSIVE MULTI-THREADED EMBEDDED GENERATION PROMPT
      mcu_prompt = f"Act as an expert Embedded Systems Firmware Engineer specializing in Real-Time Operating Systems. Generate a complete, compileable custom Arduino (.ino) firmware sketch for the '{mcu_dev}' microcontroller.\n"
      mcu_prompt += "DO NOT use the generic Firmata library. Build this custom embedded engine from scratch.\n\n"

      mcu_prompt += f"### Connected I/O Pins & Device Topologies:\n"
      mcu_prompt += f"- Core Connection Data Matrix: {store_mcusserial[mcu_dev]}\n"
      mcu_prompt += f"- Resolved Hardware Graph Mapping (via Internal Bridges): {store_terminal_connections}\n"
      mcu_prompt += f"- Sub-component System Drivers & Multiplexers: {store_mainbridge_proc}\n"
      mcu_prompt += f"- Direct Local Pins Data: {store_direct_terminal_proc}\n\n"

      mcu_prompt += "### Architecture & Thread Safety Guardrails:\n"
      mcu_prompt += "1. Concurrency: Use real-time scheduling via `#include <STM32FreeRTOS.h>` for STM32 targets or `#include <Arduino_FreeRTOS.h>` for AVR targets.\n"
      mcu_prompt += "2. Serial Thread Isolation: Instantiate a `serialMutex` (using FreeRTOS SemaphoreHandle_t). Protect ALL `Serial.print` and `Serial.println` tasks using this handle across every thread to completely stop JSON serialization interleaving or text corruption.\n"
      mcu_prompt += "3. I2C Bus Isolation: If any I2C elements or I2C Multiplexers (like the TCA9548A) are present in the hardware lists, instantiate an `i2cMutex` (SemaphoreHandle_t). Wrap all multi-device I2C transfers securely in this token.\n"
      mcu_prompt += "4. ANTI-DEADLOCK & SETUP RULE: Low-level auxiliary routines (such as selecting a channel on a TCA9548A multiplexer) must NOT attempt to lock a mutex internally if they run inside a parent function that already claims the `i2cMutex`. Furthermore, DO NOT use xSemaphoreTake blocks inside the setup() function, as the FreeRTOS scheduler is not active yet; perform raw I2C setups directly in setup() without seeking tokens.\n"
      mcu_prompt += "5. Task 1 (`taskSerialRead`, Priority 2): Regularly poll incoming bytes via `Serial.available()`. Use an `ArduinoJson` buffer to capture string lines. Accept requests like `{\"cmd\": \"set_servo\", \"pin\": X, \"value\": Y}` or `{\"cmd\": \"set_pwm\", \"pin\": X, \"value\": Y}`. Parse elements defensively, flushing out the hardware ring buffer completely if `deserializeJson()` encounters bad data packets.\n"
      mcu_prompt += "6. Task 2 (`taskSensorRead`, Priority 1): Periodically poll all connected telemetry components. Acquire the `i2cMutex`, switch the multiplexer channel if necessary, read raw sensor data bytes quickly, and release the `i2cMutex`. Next, claim the `serialMutex` and push a single clean JSON string (e.g., `{\"sensor\": \"mpu6050\", \"ax\": X}`) out to the Serial port. Add a short non-blocking `vTaskDelay` (e.g., 20ms) between distinct sensor targets to prevent bus starvation.\n"
      mcu_prompt += "7. Hardware Setup Block: Inside `setup()`, open `Serial` at 115200 baud, initialize all FreeRTOS Mutex structures FIRST. Next, execute `Wire.begin()` and directly run wake-up sequences without locking mutexes (e.g., writing 0 to power register 0x6B on MPU6050). Create and launch your FreeRTOS tasks only as the final step of setup().\n"
      mcu_prompt += "8. Main Execution Loop: The standard `loop()` block must remain empty or call `vTaskDelay(pdMS_TO_TICKS(1000))` so the RTOS scheduler controls all execution.\n"
      mcu_prompt += "9. Actuator Class Rule: If standard native servos are identified in the system profile connections, DO NOT use raw analogWrite() calls. You MUST declare an array of standard `Servo` objects from the `<Servo.h>` library, attach them to their mapped pins during setup(), and invoke `servoObjects[pin].write(value)` inside your command parsing functions to ensure correct 50Hz PWM timing.\n"
      mcu_prompt += "10. I2C Stop Rule: Ensure all raw generic I2C write transactions finalize using `Wire.endTransmission(true)` or `Wire.endTransmission()` to completely release the hardware bus lines.\n\n"
      mcu_prompt += "### Required Firmware Pattern & Syntax (Must Follow):\n"
      mcu_prompt += "- For STM32 targets, include and use this style: `#define USE_PWR_LDO_SUPPLY`, `#include <STM32FreeRTOS.h>`, `#include <Arduino_JSON.h>`, `#include \"ClosedCube_TCA9548A.h\"`, `#include \"I2Cdev.h\"`, `#include \"MPU6050.h\"`, `#include <Adafruit_ICM20X.h>`, `#include <Adafruit_ICM20948.h>`, `#include <Adafruit_Sensor.h>`.\n"
      mcu_prompt += "- Use the same task topology: `taskSerialRead`, `taskActuator`, `taskSensorRead`, `taskTelemetryTx` and use FreeRTOS queues (`commandQueue`, `telemetryQueue`).\n"
      mcu_prompt += "- Serial command parser must expect newline JSON and support: `{\"cmd\":\"set_servo\",\"pin\":<int>,\"value\":<int>}`.\n"
      mcu_prompt += "- Telemetry TX must publish newline JSON using `JSONVar`/`JSON.stringify`, with key `type` (for example `imu`) and IMU fields (`mpu_ax...icm_gz`).\n"
      mcu_prompt += "- Keep mutex wiring semantics: `serialMutex` for serial print/write sections and `i2cMutex` around shared I2C transactions.\n\n"

      if lib_check["cli_available"]:
        mcu_prompt += f"### Arduino CLI Verified Library Availability:\n- Available library queries: {lib_check['available']}\n- Missing library queries: {lib_check['missing']}\n"
        mcu_prompt += "- You MUST prioritize libraries from the available list and avoid inventing non-existent include/library names.\n"
      else:
        mcu_prompt += "### Arduino CLI Verified Library Availability:\n- `arduino-cli` not found on host at generation time; use canonical, widely available Arduino libraries only and avoid non-existent library names.\n"

      try:
        mcu_code_response = ask_llm(email, project_name, mcu_prompt, SERVICE_URLS)
        print(f"MCU Firmware Generated for {mcu_dev}:\n", mcu_code_response)
        mcu_code_extracted = extract_code(mcu_code_response)
        firmware_dir = os.path.join(get_project_dir(email, project_name), "firmware")
        os.makedirs(firmware_dir, exist_ok=True)
        firmware_filename = f"{mcu_dev}_firmware.ino"
        with open(os.path.join(firmware_dir, firmware_filename), "w") as f:
          f.write(mcu_code_extracted)
        print(f"MCU Firmware saved to {firmware_dir}/{firmware_filename}")
      except Exception as e:
        print(f"Error generating MCU Firmware for {mcu_dev}:", e)

    if "Single_Board_computer" in list(req_processor) and "Microcontroller" not in list(req_processor):
      print("Generate the code in the case only detected Single Board Computer in the system")
      sbcname = req_processor["Single_Board_computer"]
      print("Get current SBC name: ", sbcname)
      sbc_node_id_found = None
      for n_id, n_data in reqdat.items():
        if n_data['name'].startswith(sbcname):
          sbc_node_id_found = n_id
          break
      sbc_connected_devices = {}
      if sbc_node_id_found:
        sbc_node_in = reqdat[sbc_node_id_found].get('inputs', {})
        sbc_node_out = reqdat[sbc_node_id_found].get('outputs', {})
        sbc_node_io = {**sbc_node_in, **sbc_node_out}
        for port, port_data in sbc_node_io.items():
          if port_data.get('connections'):
            for conn in port_data['connections']:
              conn_node_id = str(conn['node'])
              if conn_node_id in id_device_mapping[email][project_name]:
                conn_dev_name = id_device_mapping[email][project_name][conn_node_id]
                if port not in sbc_connected_devices:
                  sbc_connected_devices[port] = []
                sbc_connected_devices[port].append(conn_dev_name)
      print(f"SBC {sbcname} (Node {sbc_node_id_found}) connections: {sbc_connected_devices}")

      sbc_prompt = f"Generate Python middleware code for a single board computer ({sbcname}). "
      sbc_prompt += "Since there is no Microcontroller detected in the system, this SBC connects to sensors and actuators directly via its own hardware pins. "
      sbc_prompt += f"The SBC is connected to the following components via specific ports: {sbc_connected_devices}. "
      sbc_prompt += "Ensure the generated Python code is based on the accurate hardware pin mapping of this specific SBC model (e.g., Raspberry Pi 3/4/5, Jetson Nano, Orange Pi, Banana Pi, etc.). "
      sbc_prompt += "Use appropriate Python libraries for direct SBC GPIO control (e.g., RPi.GPIO, Jetson.GPIO, smbus, spidev). "

      if robot_description:
        sbc_prompt += f" The overall robot description is: '{robot_description}'. Please use this description to select the specific computer vision and processing logic. "
        sbc_prompt += " If the robot description requires object detection, you MUST use YOLOv8 with the `ultralytics` library instead of cv2 object detection alone. "

      sbc_prompt += "Create empty functions for reading from and writing to these devices so control logic can be added later. Output ONLY the code, with no explanation."
        
      try:
        middleware_dir = os.path.join(get_project_dir(email, project_name), "middleware")
        os.makedirs(middleware_dir, exist_ok=True)

        sbc_code_extracted = ""
        middleware_error_reason = ""
        for attempt in range(3):
          try:
            sbc_text = ask_llm(email, project_name, sbc_prompt, SERVICE_URLS)
            candidate_code = extract_code(sbc_text)
            if is_valid_python_middleware_code(candidate_code):
              sbc_code_extracted = candidate_code
              break
            middleware_error_reason = "Gemini returned invalid middleware content"
          except Exception as attempt_error:
            middleware_error_reason = str(attempt_error)

          print(f"[WARN] GPIO middleware generation attempt {attempt + 1}/3 failed: {middleware_error_reason}")

        if not sbc_code_extracted:
          print(f"[WARN] Falling back to deterministic local middleware template: {middleware_error_reason}")
          sbc_code_extracted = build_fallback_middleware_code({})

        with open(os.path.join(middleware_dir, "middleware.py"), "w") as f:
          f.write(sbc_code_extracted)
        print(f"SBC Middleware (GPIO Mode) saved to {middleware_dir}/middleware.py")
      except Exception as e:
        print("Error generating SBC Middleware:", e)
                    
    if "Single_Board_computer" not in list(req_processor) and "Microcontroller" in list(req_processor):
        print("Generate the code in the case only detected Microcontroller in the system")
        mcu_name = req_processor["Microcontroller"]
        print("Get current MCU name: ", mcu_name)
        
        # Find MCU node IDs
        mcu_node_ids = []
        for n_id, n_data in reqdat.items():
            if n_data['name'].startswith(mcu_name):
                mcu_node_ids.append(n_id)
                
        for mcu_node_id in mcu_node_ids:
            # Extract connections for this MCU (UART, GPIOs, etc.)
            mcus_nodeconnect_in = reqdat[mcu_node_id].get('inputs', {})
            mcus_nodeconnect_out = reqdat[mcu_node_id].get('outputs', {})
            mcus_nodeconnect = {**mcus_nodeconnect_in, **mcus_nodeconnect_out}
            
            mcu_connected_devices = {}
            for port, port_data in mcus_nodeconnect.items():
                if port_data.get('connections'):
                    for conn in port_data['connections']:
                        conn_node_id = str(conn['node'])
                        if conn_node_id in id_device_mapping[email][project_name]:
                            conn_dev_name = id_device_mapping[email][project_name][conn_node_id]
                            # Store by port so we know if it's UART, GPIO, Camera etc.
                            if port not in mcu_connected_devices:
                                mcu_connected_devices[port] = []
                            mcu_connected_devices[port].append(conn_dev_name)
                            
            print(f"MCU {mcu_name} (Node {mcu_node_id}) connections: {mcu_connected_devices}")

            required_library_queries = build_dynamic_library_queries(
              mcu_name,
              [mcu_connected_devices, robot_description],
              device_class_map
            )
            lib_check = check_arduino_library_availability(required_library_queries)
            print(f"[ARDUINO LIB CHECK] {mcu_name} available={lib_check['available']} missing={lib_check['missing']} cli={lib_check['cli_available']}")
            
            mcu_prompt = f"Generate Arduino (.ino) firmware skeleton for microcontroller {mcu_name}. "
            mcu_prompt += "Since there is no Single Board Computer, this system runs entirely on the MCU. "
            
            if "ESP32" in mcu_name.upper():
                mcu_prompt += "For this ESP32 module, strictly check the memory and flash space limitations of the specific ESP32 version. Ensure the generated code is optimized and fits within the usable memory footprint. "
                mcu_prompt += "If camera modules are connected, use the specific ESP32 Camera API. Do NOT use heavy vision libraries like OpenCV or YOLOv8, as they are not supported on this microcontroller. "
                
            mcu_prompt += f"The MCU is connected to the following components via specific ports (GPIO, UART, etc): {mcu_connected_devices}. "
            mcu_prompt += "Focus on utilizing UART and GPIO for communication rather than generic Serial intended for SBCs. "
            mcu_prompt += "Include necessary Arduino libraries (e.g., Wire.h, HardwareSerial). "
            mcu_prompt += "For STM32 targets, align syntax to this firmware style: USE_PWR_LDO_SUPPLY define, FreeRTOS tasks (`taskSerialRead`, `taskActuator`, `taskSensorRead`, `taskTelemetryTx`), command queue, telemetry queue, `JSONVar` parser/serializer, and servo command schema `{\"cmd\":\"set_servo\",\"pin\":<int>,\"value\":<int>}`. "
            if lib_check["cli_available"]:
              mcu_prompt += f"Arduino CLI check results - available: {lib_check['available']}, missing: {lib_check['missing']}. Use available libraries and avoid non-existent includes. "
            else:
              mcu_prompt += "Arduino CLI is not available at generation time; use only canonical downloadable Arduino libraries and avoid non-existent includes. "
            mcu_prompt += "Leave the setup() and loop() structures, and create empty functions for reading sensors and writing actuator commands. Output ONLY the code, with no explanation."
            
            try:
                mcu_code_response = ask_llm(email, project_name, mcu_prompt, SERVICE_URLS)
                print(f"MCU Firmware Generated for {mcu_name}:\n", mcu_code_response)
                mcu_code_extracted = extract_code(mcu_code_response)
                firmware_dir = os.path.join(get_project_dir(email, project_name), "firmware")
                os.makedirs(firmware_dir, exist_ok=True)
                with open(os.path.join(firmware_dir, f"{mcu_name}_firmware.ino"), "w") as f:
                    f.write(mcu_code_extracted)
                print(f"MCU Firmware saved to {firmware_dir}/{mcu_name}_firmware.ino")
            except Exception as e:
                print(f"Error generating MCU Firmware for {mcu_name}:", e)

    update_status(email, project_name, "complete")
    return {"status": "success", "message": "Code generation complete."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9057)