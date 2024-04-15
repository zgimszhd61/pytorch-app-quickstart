import torch

# 假设模型结构已经定义在RewardModel类中
# 这里需要确保RewardModel类已经被正确导入或定义在你的工作环境中

# 奖励模型的简单定义
class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.encoder = nn.Embedding(10000, 256)  # 假设的文本编码器
        self.fc = nn.Linear(256, 1)  # 线性层，输出奖励分数

    def forward(self, response):
        response_embedding = self.encoder(response).mean(dim=1)
        score = self.fc(response_embedding)
        return score

model = RewardModel()
model = torch.load('reward_model_entity.pth', map_location=torch.device('cpu'))
model.eval()  # 切换到评估模式

# 示例输入
input_sentence = "我讨厌今天的天气。"
# 假设使用相同的简单哈希方法将文本转换为索引
input_indices = [hash(word) % 10000 for word in input_sentence.split()]
print(input_indices)
input_tensor = torch.LongTensor(input_indices).unsqueeze(0)  # 增加一个批次维度

with torch.no_grad():  # 不计算梯度
    predicted_score = model(input_tensor).item()

print(f"Predicted score: {predicted_score}")
