class Config:

    data_path = './data'  # 数据存放路径
    num_workers = 4  # 多线程加载线程数
    img_size = 64  # 图片尺寸
    batch_size = 32
    max_epoch = 100
    lr = 0.001
    use_gpu = True
    nz = 200  # 噪声维度
    ngf = 64  # 生成器feature map数
    ndf = 64  # 判别器feature map数

    save_path = ''  # 图片保存地址

    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    decay_every = 10  # 每10个epoch保存一次模型

    load_net_bool = 1
    Netg_path = './Module_G/NetG-2030.pth'
    Netd_path = './Module_D/NetD-2030.pth'


opt = Config()
