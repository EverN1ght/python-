import cv2  
import numpy as np 
from sklearn.datasets import load_digits 
from sklearn.neighbors import KNeighborsClassifier 

# --- 1. 准备和训练KNN模型 ---
# 加载内置的手写数字数据集 (8x8像素的图像)
shuJuJi = load_digits()
# X是图像数据 (每行64个像素值), Y是对应的数字标签 (0-9)
tuPianShuJu = shuJuJi.data
biaoQian = shuJuJi.target

# 初始化KNN分类器，n_neighbors(邻居数量K)是核心参数，这里设为5
knnFenLeiQi = KNeighborsClassifier(n_neighbors=5)
# 使用全部数据训练KNN模型
knnFenLeiQi.fit(tuPianShuJu, biaoQian)
print("KNN模型训练完毕!")

# --- 2. 主识别流程 ---
# 获取用户输入的图片路径
tuPianLuJing = input("请输入图片路径 : ")

if not tuPianLuJing:
    print("未输入路径，程序退出。")
else:
    # 读取用户提供的图片
    yuanShiTuPian = cv2.imread(tuPianLuJing)

    # 创建一个副本用于绘制结果，避免修改原始图片
    jieGuoTuPian = yuanShiTuPian.copy()
    # 将彩色图片转换为灰度图，简化处理
    huiDuTu = cv2.cvtColor(yuanShiTuPian, cv2.COLOR_BGR2GRAY)

    # --- 图像预处理 ---
    # a. 高斯模糊：去除噪声，平滑图像。 (5,5)是核大小, 0是sigmaX
    moHuTu = cv2.GaussianBlur(huiDuTu, (5, 5), 0)

    # b. 自适应阈值化：将灰度图转为二值图（黑白）。
    #    255是最大像素值, ADAPTIVE_THRESH_GAUSSIAN_C 使用高斯邻域计算阈值
    #    THRESH_BINARY_INV 反转阈值，使数字部分变白，背景变黑
    #    blockSize=21, C=7 是关键参数，可能需要根据图片调整
    erZhiTu = cv2.adaptiveThreshold(moHuTu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 7)

    # c. (可选)形态学操作：进一步清理二值图
    #    开运算 (MORPH_OPEN): 去除小的白色噪点
    kaiYunSuanHe = np.ones((2, 2), np.uint8) # 2x2的核
    erZhiTu = cv2.morphologyEx(erZhiTu, cv2.MORPH_OPEN, kaiYunSuanHe, iterations=1)
    #    闭运算 (MORPH_CLOSE): 填充数字内部的小黑洞或连接断裂笔画
    biYunSuanHe = np.ones((3, 3), np.uint8) # 3x3的核
    erZhiTu = cv2.morphologyEx(erZhiTu, cv2.MORPH_CLOSE, biYunSuanHe, iterations=2)

    # --- 轮廓检测和数字分割 ---
    # 查找二值图中的轮廓 (数字的边界)
    # RETR_EXTERNAL: 只检测最外层轮廓
    # CHAIN_APPROX_SIMPLE: 压缩轮廓点，节省存储
    lunKuoLieBiao, _ = cv2.findContours(erZhiTu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"初步找到 {len(lunKuoLieBiao)} 个轮廓。")

    # 如果找到轮廓，按x坐标从左到右排序
    if lunKuoLieBiao:
        lunKuoLieBiao = sorted(lunKuoLieBiao, key=lambda lk: cv2.boundingRect(lk)[0])

    shiBieShuLiang = 0 # 记录成功识别的数字数量

    # 遍历每个找到的轮廓
    for lunKuo in lunKuoLieBiao:
        # 获取包围轮廓的最小矩形 (x, y是左上角坐标, w, h是宽高)
        x, y, w, h = cv2.boundingRect(lunKuo)

        # --- 简单的轮廓过滤 ---
        # 过滤掉太小或形状不合适的轮廓，这些简单阈值需要根据实际图片调整
        if not (w > 8 and h > 15 and cv2.contourArea(lunKuo) > 50):
            continue # 跳过不合格的轮廓

        # --- ROI (Region of Interest, 感兴趣区域) 处理 ---
        # a. 从原始灰度图中提取数字区域 (ROI)
        roiHuiDu = huiDuTu[y:y+h, x:x+w]
        if roiHuiDu.size == 0: continue # 如果ROI为空则跳过

        # b. 颜色反转：KNN模型训练数据是亮数字暗背景，所以这里确保ROI也符合
        roiFanZhuan = cv2.bitwise_not(roiHuiDu)

        # c. 标准化ROI到中间尺寸 (例如20x20)，用于改善缩放效果
        #    目标是让数字居中并保持一定宽高比
        zhongJianBianChang = 20 # 标准化画布的边长
        roiGao, roiKuan = roiFanZhuan.shape
        suoFangYinZi = zhongJianBianChang / max(roiGao, roiKuan)
        xinKuan, xinGao = max(1, int(roiKuan * suoFangYinZi)), max(1, int(roiGao * suoFangYinZi))
        
        roiSuoFangZhongJian = cv2.resize(roiFanZhuan, (xinKuan, xinGao), interpolation=cv2.INTER_AREA)

        # 创建一个黑色的方形画布
        huaBu = np.zeros((zhongJianBianChang, zhongJianBianChang), dtype=np.uint8)
        # 计算偏移量，将缩放后的ROI放到画布中心
        xPianYi = (zhongJianBianChang - xinKuan) // 2
        yPianYi = (zhongJianBianChang - xinGao) // 2
        huaBu[yPianYi:yPianYi + xinGao, xPianYi:xPianYi + xinKuan] = roiSuoFangZhongJian

        # d. 最终缩放到KNN模型输入尺寸 (8x8)
        roiZuiZhongKNN = cv2.resize(huaBu, (8, 8), interpolation=cv2.INTER_AREA)

        # e. 像素值归一化 (0-16范围，匹配load_digits数据) 和展平
        #    先从0-255转到0-1，再乘以16
        chuLiHouKNNShuRu = (roiZuiZhongKNN / 255.0) * 16.0
        #    展平成一维向量 (1行, 64列) 并确保数据类型为float32
        zhanPingKNNShuRu = chuLiHouKNNShuRu.astype(np.float32).flatten().reshape(1, -1)

        # --- 使用KNN进行预测 ---
        yuCeBiaoQian = knnFenLeiQi.predict(zhanPingKNNShuRu)[0]
        # 获取预测概率 (对于KNN是K个邻居中属于该类的比例)
        yuCeGaiLv = knnFenLeiQi.predict_proba(zhanPingKNNShuRu)[0][yuCeBiaoQian]

        shiBieShuLiang += 1
        print(f"  识别到数字: {yuCeBiaoQian}, 概率: {yuCeGaiLv:.2f} @ ({x},{y},{w},{h})")

        # --- 在结果图上绘制方框和预测结果 ---
        cv2.rectangle(jieGuoTuPian, (x, y), (x + w, y + h), (255, 0, 0), 2) # 蓝色方框
        # 确定文本显示位置，避免超出图像边界
        wenBenY = y - 10 if y - 10 > 10 else y + h + 20
        cv2.putText(jieGuoTuPian, f"{yuCeBiaoQian} ({yuCeGaiLv:.2f})", (x, wenBenY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # 蓝色文本

    print(f"\n共识别出 {shiBieShuLiang} 个数字。")

    # --- 显示最终结果 ---
    tuGao, tuKuan = jieGuoTuPian.shape[:2]
    zuiDaXianShiKuan, zuiDaXianShiGao = 1200, 800
    xianShiSuoFang = min(zuiDaXianShiKuan / tuKuan if tuKuan > 0 else 1,
                            zuiDaXianShiGao / tuGao if tuGao > 0 else 1, 1.0)
        
    if xianShiSuoFang < 1.0:
        xianShiTu = cv2.resize(jieGuoTuPian, (int(tuKuan * xianShiSuoFang), int(tuGao * xianShiSuoFang)))
    else:
        xianShiTu = jieGuoTuPian
    
    cv2.imshow("数字识别结果", xianShiTu)
    cv2.waitKey(0)