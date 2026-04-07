import numpy as np

def get_config():
    config = {
        'hostname': '172.16.0.2',
        'username': 'tamarlab',
        'password': 'Panda468#',
        'recording_path': '/mnt/pub/palpation/new_data/panda_4',
        'camera_name' : 'Microsoft',
        'robot_name': 'panda_4',
        'sensor_number': 4,
        'rs_serial' : '218622276852',
        'temp_q': [-0.39453405,  0.21150732,  0.28076104, -2.52075257, -0.12154363,  2.75299339,  0.7356554 ],
        'temp_q_2': [-0.29838777,  0.55722348,  0.23563692, -1.93898518, -0.18877488 , 2.50889932,0.7935656 ],
        'temp_q_3': [-0.01762457 , 0.59781703 , 0.2604072 , -1.87543604 ,-0.2312864 , 2.48085025 ,  1.12010207],
        'sample_positions': [[-0.39453405,  0.21150732,  0.28076104, -2.52075257, -0.12154363,  2.75299339,  0.7356554 ]],
        'lift_pos': [-0.21217873, -0.13201017,  0.25788068, -2.49595508,  0.05939012,  2.41168665,  0.8301551 ],
        'lift_data': [
             {'device_name': '/dev/ttyUSB1', 'id_1': 3, 'id_2': 4,
              'position': [-0.21217873, -0.13201017,  0.25788068, -2.49595508,  0.05939012,  2.41168665,  0.8301551 ],
              'primitive_file': 'pose_34.npy',
              'lift_name': 'Lift34'},
             {'device_name': '/dev/ttyUSB1', 'id_1': 5, 'id_2': 6,
              'position': [-0.22818693,  0.33416555,  0.28397384, -1.85507483, -0.13337539,  2.2363409,    0.92899983],
              'primitive_file': 'pose_56.npy',
              'lift_name': 'Lift56'},
            # {'device_name': '/dev/ttyUSB1', 'id_1': 3, 'id_2': 4,
            #   'position': [-0.22142583, -0.10287286,  0.33933295, -2.49320063,  0.02689746,  2.45281411,  0.85990257],
            #   'primitive_file': 'just_poke_downsampled.npy',
            #   'lift_name': 'i6_tumor'},

              
        ],
        # CAN interface name - set to 'auto' to auto-detect, or specify 'can0' or 'slcan0'
        'can_interface': 'auto',
        'xela_touch_diff': 90
    }
    return config
