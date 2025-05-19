import math

from model.mydataclass import TrainingParams # 导入训练参数的数据结构

# 基础调度器类（scheduler 基类）
class scheduler(object):
    def __init__(self, params: TrainingParams, init_beta: int, init_step: int) -> None:
        
        self.beta = init_beta # 当前 beta 值（VAE 中 KL 散度项的权重）
        self.t = init_step  # 当前训练步数（step）
        self.warmup = params.beta_warmup # 热启动阶段，不进行变化
        self.beta_min = params.beta_min  # beta 的最小值
        self.beta_max = params.beta_max # beta 的最大值
        self.beta_anneal_period = params.beta_anneal_period  # 从 min 到 max 的变化周期
    
    def step(self):
        pass

# sigmoid_schedule：使用 sigmoid 曲线平滑增加 beta
class sigmoid_schedule(scheduler):
    def __init__(self, params: TrainingParams, init_beta: int, init_step: int) -> None:
        super(sigmoid_schedule, self).__init__(params, init_beta, init_step)
        self.diff = self.beta_max - self.beta_min  # beta 最大最小值的差值（幅度）
        # anneal_rate 是一个小于 1 的衰减因子，控制增长速度
        self.anneal_rate = math.pow(0.01, 1 / self.beta_anneal_period)
        self.weight = 1  # 初始化时权重为 1（还未变化）
        
    def step(self):
        if self.t < self.warmup:
            self.beta = self.beta_min  # 热启动阶段：保持 beta 为最小值
        else:
        # 计算当前 step 对应的衰减因子（越大越靠近 max）
            self.weight = math.pow(self.anneal_rate, self.t - self.warmup)
            self.beta = self.beta_min + self.diff * (1 - self.weight)
        self.t += 1  # 更新 step
        return self.beta  # 返回当前 beta


# 封装类，用于灵活切换 beta 计划策略（目前仅实现 sigmoid）
class beta_annealing_schedule(object):
    def __init__(self, params: TrainingParams, init_beta: int=0, init_step: int=0) -> None:
        self.schedule = sigmoid_schedule(params, init_beta, init_step)
    
    def step(self):
        return self.schedule.step()  # 调用内部策略的 step
