import sys, os, torch, platform
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json, cv2, time
from yolov5 import Ui_MainWindow

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # 自动选择模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./weights')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./weights/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5检测线程
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./weights/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label_1))
        self.det_thread.send_fps.connect(lambda x: self.label.setText(x))
        self.det_thread.send_statistic.connect(self.show_statistic) #将send_statistic信号绑定到

        '''
        绑定按键
        '''
        self.flag = True #设置一个标志位，用于控制检测开始与暂停
        self.PB_1.clicked.connect(self.open_file)
        self.PB_2.clicked.connect(self.open_cam)
        self.PB_3.clicked.connect(self.run_or_continue)
        self.PB_4.clicked.connect(self.stop)
        self.comboBox.currentTextChanged.connect(self.change_model)
        self.doubleSpinBox_1.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.doubleSpinBox_2.valueChanged.connect(lambda x: self.change_val(x, 'ioufSpinBox'))
        self.checkBox.clicked.connect(self.is_save)
        self.load_setting()

    '''
    函数功能：打开文件，第一个按键功能的槽函数
    '''
    def open_file(self):
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)") #创建一个对话框，筛选指定文件夹下的文件
        if name:
            if name.endswith(".jpg"):
                ori_img = cv2.imread(name)
                self.show_image(ori_img,self.label_1)
            elif name.endswith(".png"):
                ori_img = cv2.imread(name)
                self.show_image(ori_img,self.label_1)
            else:
                pass

            self.det_thread.source = name
            self.statistic_msg('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    '''
    函数功能：加载摄像头
    '''
    def open_cam(self):
        self.det_thread.source = '0'
        self.statistic_msg('正在加载摄像头......')

    '''
    运行/暂停按键功能
    '''
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.flag is True: #检查运行/暂停按键是否被选中
            self.flag = False
            # self.saveCheckBox.setEnabled(False) #判断保存结果按钮状态
            self.det_thread.is_continue = True #如果运行/暂停按键标志位为True，开始进行检测
            if not self.det_thread.isRunning(): #如果检测线程没有在运行
                self.det_thread.start() #开启检测线程
                self.statistic_msg('开始检测......')
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('检测中 >> 加载模型为：{}，检测文件为：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source)) #发送模型相关信息
        else:
            self.flag = True
            self.det_thread.is_continue = False
            self.statistic_msg('暂停检测......') #发送暂停信息

    '''
    函数功能：结束检测线程
    '''
    def stop(self):
        self.det_thread.jump_out = True
        self.checkBox.setEnabled(True)
        self.flag = True
        self.statistic_msg('结束检测......')  # 发送结束信息

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.doubleSpinBox_1.setValue(x)
            self.det_thread.conf_thres = x
        elif flag == 'ioufSpinBox':
            self.doubleSpinBox_2.setValue(x)
            self.det_thread.iou_thres = x
        else:
            pass

    '''
    函数功能：保存结果
    '''
    def is_save(self):
        if self.checkBox.isChecked(): #如果保存检测结果的按钮为打开状态
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    '''
    函数功能：加载预设参数
    '''
    def load_setting(self):
        config_file = 'config/setting.json' #配置文件路径
        if not os.path.exists(config_file): #判断配置文件是否存在，如果不存在则加载以下配置文件参数
            iou = 0.25
            conf = 0.45
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2) #将python格式的字典转换为json格式的字符串
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json) #创建并打开配置文件，将新的配置文件写入
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8')) #如果配置文件存在，使用json.load函数打开文件，并将数据解析为python对象
            if len(config) != 3:
                iou = 0.25
                conf = 0.45
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                savecheck = config['savecheck']
        self.doubleSpinBox_1.setValue(conf)
        self.doubleSpinBox_2.setValue(iou)
        self.checkBox.setCheckState(savecheck)
        self.is_save()

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./weights/%s" % self.model_type
        self.statistic_msg('推理模型改变为： %s' % x) #发送信息

    def statistic_msg(self, msg):
        self.result_browser.append(msg) #将接收到的信息内容显示到statistic_label标签上

    '''
    函数功能：展示检测结果
    '''
    def show_statistic(self, statistic_dic):
        try:
            self.result_browser.clear() #清空resultWidget上的内容
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            print(statistic_dic)
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            for i in results:
                result = i
                self.result_browser.append(result)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.doubleSpinBox_1.value()
        config['conf'] = self.doubleSpinBox_2.value()
        # config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.checkBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        sys.exit(0)

    '''
    选择权重
    '''
    def search_pt(self):
        pt_list = os.listdir('./weights') #列出该文件夹下的文件列表
        pt_list = [file for file in pt_list if file.endswith('.pt')] #筛选.pt结尾的文件
        pt_list.sort(key=lambda x: os.path.getsize('./weights/' + x)) #根据.pt文件的大小进行升序排列

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    '''
    函数功能：将图片信息流显示到GUI界面的指定标签上
    '''
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

''''
YOLOv5算法的检测线程
'''
class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False  # jump out of the loop，跳出循环
        self.is_continue = True  # continue/pause
        self.percent_length = 1000  # progress bar
        self.save_fold = './result'

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)

            while True:
                if self.jump_out:
                    self.vid_cap.release()
                    # self.send_percent.emit(0)
                    # self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break
                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                if self.is_continue:
                    path, img, im0s, self.vid_cap, _ = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        fps = int(30 / (time.time() - start_time))
                        self.send_fps.emit('fps：' + str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms,
                                               max_det=max_det)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))

                    im0 = annotator.result()
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if self.save_fold:
                        os.makedirs(self.save_fold, exist_ok=True)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                   time.localtime()) + '.jpg')
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold,
                                                         time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow() #实例化主窗口
    myWin.show() #显示主窗口界面
    sys.exit(app.exec_()) #应用程序的主事件循环，确保应用程序一直运行，直到用户关闭主窗口