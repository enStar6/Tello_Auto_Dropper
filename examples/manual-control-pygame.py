from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import serial
import  torch
from ultralytics import  YOLO



ser = serial.Serial('COM7', 115200, timeout=1)
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
# pygame窗口显示的帧数
# 较低的帧数会导致输入延迟，因为一帧只会处理一次输入信息
FPS = 120

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.

        保持Tello画面显示并用键盘移动它
        按下ESC键退出
        操作说明：
            T：起飞
            L：降落
            方向键：前后左右
            A和D：逆时针与顺时针转向
            W和S：上升与下降

    """

    def __init__(self):
        # Init pygame
        # 初始化pygame
        pygame.init()

        # Creat pygame window
        # 创建pygame窗口
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        # 初始化与Tello交互的Tello对象
        self.tello = Tello()

        # Drone velocities between -100~100
        # 无人机各方向速度在-100~100之间
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 100
        self.S = 60
        self.dropper = "lock"
        self.servo_ord = "a"
        self.send_rc_control = False

        # 视觉相关
        # 检查CUDA设备是否可用，并设置设备
        self.model_path = "ep_train2.pt"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.model = YOLO(self.model_path).to(self.device)

        # create update timer
        # 创建上传定时器
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def run(self):

        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        # 防止视频流已开启。这会在不使用ESC键退出的情况下发生。
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()
        tar_list = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            # battery n. 电池
            cv2.putText(frame, "Battery: ", (5, 720-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            battery = self.tello.get_battery()
            if battery >= 20:
                cv2.putText(frame, str(battery)+"%", (130, 720-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, str(battery)+"%", (130, 720-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # 杆量大小
            cv2.putText(frame, "Velocity: ", (5, 720-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.S == 100:
                cv2.putText(frame, str(self.S), (140, 720-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, str(self.S), (140, 720-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 投掷器状态
            cv2.putText(frame, "dropper: ", (5, 720-75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.dropper == "lock":
                cv2.putText(frame, self.dropper, (140, 720-75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, self.dropper, (140, 720-75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 矩形尺寸
            green_height = 300
            green_width = 400
            red_height = int(green_height * 0.8)
            red_width = int(green_width * 0.8)
            # 大绿色矩形数据
            x1_green = int((960 - green_width) / 2)
            y1_green = int((720 - green_height) / 2)
            x2_green = x1_green + green_width
            y2_green = y1_green + green_height
            cv2.rectangle(frame, (x1_green, y1_green), (x2_green, y2_green), (0, 255, 0), 2)
            # 小红色矩形数据
            x1_red = int((960 - red_width) / 2)
            y1_red = int((720 - red_height) / 2)
            x2_red = x1_red + red_width
            y2_red = y1_red + red_height
            # 添加矩形图像
            cv2.rectangle(frame, (x1_red, y1_red), (x2_red, y2_red), (255, 0, 0), 2)

            # 转换图像方向
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            # 显示图像
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call it always before finishing. To deallocate resources.
        # 通常在结束前调用它以释放资源
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key

        基于键的按下上传各个方向的速度
        参数：
            key：pygame事件循环中的键事件
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = self.S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -self.S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -self.S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = self.S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = self.S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -self.S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -self.S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = self.S
        elif key == pygame.K_RETURN:
            if self.servo_ord == "b":
                self.servo_ord = "a"
                self.dropper = "lock"
                ser.write(self.servo_ord.encode('utf-8'))
                print("lock servo")
        elif key == pygame.K_TAB:
            if self.servo_ord == "a":
                self.servo_ord = "b"
                self.dropper = "free"
                ser.write(self.servo_ord.encode('utf-8'))
                print("attack servo")

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key

        基于键的松开上传各个方向的速度
        参数：
            key：pygame事件循环中的键事件
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False
        elif key == pygame.K_r:  # self.S change
            self.S = 100 if self.S == 60 else 60
        elif key == pygame.K_p:  # fall
            self.tello.emergency()

    def update(self):
        """ Update routine. Send velocities to Tello.

            向Tello发送各方向速度信息
        """
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

    def get_target(self, img):
        tar_list = []
        results = self.model(img)
        xywh = results[0].boxes.xywh.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        for index in range(len(xywh)):
            if cls[index] == 2 and conf[index] > 0.6:
                new_box = np.concatenate([xywh[index], [conf[index]]])  # 将conf添加到xywh数组
                tar_list.append((new_box, cls[index], conf[index]))  # 同时保存类别和置信度
        return tar_list


def main():
    frontend = FrontEnd()

    # run frontend

    frontend.run()


if __name__ == '__main__':
    main()
