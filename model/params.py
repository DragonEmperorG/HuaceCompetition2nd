
#网络参数
class Params(object):
    batch_size = 4 #不能改
    map_rows = 61 #tec_map行数
    map_cols = 49 #tec_map列数
    input_time_steps = 24 #输入时间序列长度，36*2小时/24小时=3天
    output_time_steps = 12 #输出时间序列长度，24*2小时/24小时=2天
    conv_nb_filter = 24 #卷积层卷积核数量
    conv_lstm_filters = conv_nb_filter #暂时保持一致
    external_dim = 5 #外源输入维度
    lr = 0.0001
    num_epochs = 20
    
    def __init__(self):
        pass