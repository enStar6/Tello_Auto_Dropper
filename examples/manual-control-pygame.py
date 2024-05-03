from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import serial
import torch
from ultralytics import YOLO

ser = serial.Serial('COM7', 115200, timeout=1)
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
# pygame窗口显示的帧数
# 较低的帧数会导致输入延迟，因为一帧只会处理一次输入信息
FPS = 120

def nothing(x):
    pass

class PID:
    def __init__(self, kp, ki, kd, maxIntergral, maxOutput, deadZone, errLpfRatio):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.error = 0
        self.last_error = self.error
        self.output = 0
        self.maxIntergral = maxIntergral
        self.maxOutput = maxOutput
        self.deadZone = deadZone
        self.errLpfRatio = errLpfRatio

    def update_err(self, error):
        self.last_error = self.error if self.last_error > self.deadZone else 0
        self.error = error * self.errLpfRatio + self.last_error * (1 - self.errLpfRatio)
        self.integral += self.error
        if self.integral > self.maxIntergral:
            self.integral = self.maxIntergral
        elif self.integral < -self.maxIntergral:
            self.integral = -self.maxIntergral
        derivative = error - self.last_error
        self.output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.output > self.maxOutput:
            self.output = self.maxOutput
        elif self.output < -self.maxOutput:
            self.output = -self.maxOutput
        return self.output
    def update_pid(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

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
        self.model_path = "D:/Robomaster/DJITelloPy-master/bowl.pt"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.model = YOLO(self.model_path).to(self.device)
        self.last_target = []
        self.pid_x = PID(0.2, 0.0, 0.0, 60.0, 100.0, 1.0, 0.5)
        self.pid_y = PID(0.2, 0.0, 0.0, 20.0, 100.0, 1.0, 0.5)
        self.pid_z = PID(0.2, 0.0, 0.0, 60.0, 100.0, 1.0, 0.5)
        self.err_x, self.out_x = 0.0, 0.0
        self.err_y, self.out_y = 0.0, 0.0
        self.err_z, self.out_z = 0.0, 0.0
        self.is_auto = False
        cv2.namedWindow("parameters", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("parameters", 500, 540)
        cv2.createTrackbar("kp_x", "parameters", 20, 200, nothing)
        cv2.createTrackbar("ki_x", "parameters", 0, 200, nothing)
        cv2.createTrackbar("kd_x", "parameters", 0, 200, nothing)
        cv2.createTrackbar("kp_y", "parameters", 20, 200, nothing)
        cv2.createTrackbar("ki_y", "parameters", 0, 200, nothing)
        cv2.createTrackbar("kd_y", "parameters", 0, 200, nothing)
        cv2.createTrackbar("kp_z", "parameters", 20, 200, nothing)
        cv2.createTrackbar("ki_z", "parameters", 0, 200, nothing)
        cv2.createTrackbar("kd_z", "parameters", 0, 200, nothing)

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
            cv2.putText(frame, "Battery: ", (5, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            battery = self.tello.get_battery()
            if battery >= 20:
                cv2.putText(frame, str(battery) + "%", (130, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, str(battery) + "%", (130, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # 杆量大小
            cv2.putText(frame, "Velocity: ", (5, 720 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.S == 100:
                cv2.putText(frame, str(self.S), (140, 720 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, str(self.S), (140, 720 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 投掷器状态
            cv2.putText(frame, "dropper: ", (5, 720 - 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.dropper == "lock":
                cv2.putText(frame, self.dropper, (140, 720 - 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, self.dropper, (140, 720 - 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 自动对位
            cv2.putText(frame, "auto_aim: ", (5, 720 - 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.is_auto:
                cv2.putText(frame, str(self.is_auto), (160, 720 - 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, str(self.is_auto), (160, 720 - 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
            # 添加准心图像
            point_x = int(x1_red + red_width / 2)
            point_y = int(y1_red + red_height / 2)
            cv2.circle(frame, (point_x, point_y), 1, (255, 0, 0), 4)

            # 视觉相关绘制
            tar_list = self.get_target(frame, 0.7)
            if len(tar_list) > 0:
                min_err_dist = 5000 ** 2
                min_box = []
                for box in tar_list:
                    x1 = box[0] - box[2] / 2
                    y1 = box[1] - box[3] / 2
                    x2 = box[0] + box[2] / 2
                    y2 = box[1] + box[3] / 2
                    conf = box[4]
                    err_dist = (box[0] - point_x) ** 2 + (box[1] - point_y) ** 2
                    if err_dist < min_err_dist:
                        min_err_dist = err_dist
                        min_box = box.copy()
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.circle(frame, (int(box[0]), int(box[1])), 1, (255, 0, 0), 4)
                    cv2.putText(frame, "Target" + "   " + str(conf), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0), 2)
                    cv2.line(frame, (int(box[0]), int(box[1])),
                             (int(x1_red + red_width / 2), int(y1_red + red_height / 2)), (0, 255, 255), 3)
                # 更新误差
                self.err_x = min_box[0] - point_x
                self.err_y = point_y - min_box[1]
                self.err_z = 220 - min_box[2]
                # 标出最近的目标
                x1_p = min_box[0] - min_box[2] / 2
                y1_p = min_box[1] - min_box[3] / 2
                x2_p = min_box[0] + min_box[2] / 2
                y2_p = min_box[1] + min_box[3] / 2
                print("tar_width: " + str(min_box[2]))
                print("tar_height: " + str(min_box[3]))
                conf_p = min_box[4]
                cv2.rectangle(frame, (int(x1_p), int(y1_p)), (int(x2_p), int(y2_p)), (255, 0, 255), 2)
                cv2.circle(frame, (int(min_box[0]), int(min_box[1])), 1, (255, 0, 255), 4)
                cv2.putText(frame, "Target" + "   " + str(conf_p), (int(x1_p), int(y1_p)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 255), 2)
                cv2.line(frame, (int(min_box[0]), int(min_box[1])),
                         (int(x1_red + red_width / 2), int(y1_red + red_height / 2)),
                         (0, 255, 255), 3)
            else:
                self.err_x = 0
                self.err_y = 0
                self.err_z = 0

            # 动态调节pid参数
            self.pid_x.update_pid(float(cv2.getTrackbarPos("kp_x", "parameters")) / 100,
                                  float(cv2.getTrackbarPos("ki_x", "parameters")) / 100,
                                  float(cv2.getTrackbarPos("kd_x", "parameters")) / 100)
            self.pid_y.update_pid(float(cv2.getTrackbarPos("kp_y", "parameters")) / 100,
                                  float(cv2.getTrackbarPos("ki_y", "parameters")) / 100,
                                  float(cv2.getTrackbarPos("kd_y", "parameters")) / 100)
            self.pid_z.update_pid(float(cv2.getTrackbarPos("kp_z", "parameters")) / 100,
                                  float(cv2.getTrackbarPos("ki_z", "parameters")) / 100,
                                  float(cv2.getTrackbarPos("kd_z", "parameters")) / 100)
            print(self.pid_x.kp)

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
            self.is_auto = False
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -self.S
            self.is_auto = False
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -self.S
            self.is_auto = False
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = self.S
            self.is_auto = False
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = self.S
            self.is_auto = False
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -self.S
            self.is_auto = False
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -self.S
            self.is_auto = False
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = self.S
            self.is_auto = False
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
        elif key == pygame.K_u:
            self.is_auto = True if not self.is_auto else False

    def update(self):
        """ Update routine. Send velocities to Tello.

            向Tello发送各方向速度信息
        """
        if self.send_rc_control:
            self.out_x = self.pid_x.update_err(self.err_x)
            self.out_y = self.pid_y.update_err(self.err_y)
            self.out_z = self.pid_z.update_err(self.err_z)
            if self.is_auto:
                self.tello.send_rc_control(int(self.out_x), int(self.out_z),
                                           int(self.out_y), self.yaw_velocity)
            else:
                self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                                           self.up_down_velocity, self.yaw_velocity)

    def get_target(self, img, conf_threshold):
        tar_list = []
        # 执行模型得到结果
        results = self.model(img)

        # 提取boxes的xywh坐标，类别，和置信度
        xywh = results[0].boxes.xywh.cpu().numpy()  # 形状为[N, 4]，每行是一个目标的[x_center, y_center, width, height]
        cls = results[0].boxes.cls.cpu().numpy()  # 形状为[N,]
        conf = results[0].boxes.conf.cpu().numpy()  # 形状为[N,]

        # 循环遍历每个检测结果
        for index in range(len(cls)):
            if cls[index] == 0.0 and conf[index] >= conf_threshold:
                # 确保 conf[index] 作为一个数组合并
                conf_array = np.array([conf[index]])  # 转换单个浮点数为数组
                combined = np.concatenate((xywh[index], conf_array))  # 正确使用 concatenate
                tar_list.append(combined)

        return tar_list


def main():
    frontend = FrontEnd()

    # run frontend

    frontend.run()


if __name__ == '__main__':
    main()
