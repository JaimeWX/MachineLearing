# 导入pgmpy相关模块
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

### step1 构建模型框架，指定DAG中各个节点之间的指向关系
letter_model = BayesianModel([('D', 'G'),
                               ('I', 'G'),
                               ('G', 'L'),
                               ('I', 'S')])
### step2 构建各个节点的条件概率分布，需要指定相关参数和传入概率表
# 考试难度的条件概率分布
difficulty_cpd = TabularCPD(
            variable='D', # 节点名称
            variable_card=2, # 节点取值个数
            values=[[0.6], [0.4]] , # 该节点的概率表
)
# 个人天赋的条件概率分布
intel_cpd = TabularCPD(
            variable='I',
            variable_card=2,
            values=[[0.7], [0.3]]
)
# 学生成绩的条件概率分布
grade_cpd = TabularCPD(
    variable='G', # 节点名称
    variable_card=3, # 节点取值个数
    values=[[0.3, 0.05, 0.9, 0.5], # 该节点的概率表
            [0.4, 0.25, 0.08, 0.3],
            [0.3, 0.7, 0.02, 0.2]],
    evidence=['I', 'D'], # 该节点的依赖节点
    evidence_card=[2, 2] # 依赖节点的取值个数
)

# 推荐信质量的条件概率分布
letter_cpd = TabularCPD(
            variable='L',
            variable_card=2,
            values=[[0.1, 0.4, 0.99],
                    [0.9, 0.6, 0.01]],
            evidence=['G'],
            evidence_card=[3]
)
# SAT考试分数的条件概率分布
sat_cpd = TabularCPD(
            variable='S',
            variable_card=2,
            values=[[0.95, 0.2],
                    [0.05, 0.8]],
            evidence=['I'],
            evidence_card=[2]
)
### step3 将各节点添加到模型中，构建贝叶斯网络
letter_model.add_cpds(
    grade_cpd,
    difficulty_cpd,
    intel_cpd,
    letter_cpd,
    sat_cpd
)
# 导入pgmpy贝叶斯推断模块
from pgmpy.inference import VariableElimination
# 贝叶斯网络推断
letter_infer = VariableElimination(letter_model)
# 天赋较好且考试不难的情况下推断该学生获得推荐信质量的好坏
prob_G = letter_infer.query(
            variables=['G'],
            evidence={'I': 1, 'D': 0})
print(prob_G)
