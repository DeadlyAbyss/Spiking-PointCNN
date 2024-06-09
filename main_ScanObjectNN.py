# from torch_geometric.loader import DataLoader
import torch
# from torch import Tensor as T
from torch.utils.data import DataLoader
from Dataloader.ScanObjectNN import ScanObjectNN

# from SpikeModel import Spike_PointCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 指定使用的显卡编号
device = torch.device("cuda:0")  # 使用第一块显卡
# device = torch.device("cpu")
from datetime import datetime
import os
import time
from SpikeModel import *
# from SpikeModel import *
# from Model import *

# 实验结果的保存路径
record_folder_path = "Result/Basic/ScanObjectNN"
# 检查路径是否存在
if not os.path.exists(record_folder_path):
    # 如果路径不存在，则创建路径
    os.makedirs(record_folder_path)
    print(f"路径 '{record_folder_path}' 不存在，已创建成功。")
else:
    print(f"路径 '{record_folder_path}' 已存在。")

# 一些变量
epochs = 1000
# epochs=2
acc_record = []
acc_record.append(-1)
best_epoch = -1
best_OA= -1
best_mACC = -1
wait = 0
patience = 50
mark = 0
total_time = 0

# 训练模型
def train(model,train_loader,optimizer):
    model.train()

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device).squeeze()

        # 生成批次信息，假设所有样本都有相同数量的点
        batch_size, num_points, _ = data.shape
        batch = torch.arange(batch_size).repeat_interleave(num_points).to(device)

        # 数据已经是 [batch_size, num_points, coordinates]，所以直接使用
        pos = data.reshape(-1, 3)  # 重塑为 [batch_size*num_points, coordinates]
        # Assuming your data is already in shape [batch, 3, num_points], no need to permute
        optimizer.zero_grad()
        out = model(pos, batch)
        loss = F.nll_loss(out, label)
        loss.backward()
        optimizer.step()

# 计算over_acc和mACC
def test(model, test_loader):
    model.eval()  # 将模型设置为评估模式
    class_correct = {}
    class_total = {}
    correct = 0

    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device).squeeze()

        # 假设数据的形状为 [batch_size, num_points, coordinates]
        batch_size, num_points, _ = data.shape
        batch = torch.arange(batch_size).repeat_interleave(num_points).to(device)

        # 重塑为 [batch_size*num_points, coordinates]
        pos = data.reshape(-1, 3)
        pred = model(pos, batch).max(1)[1]

        for label, prediction in zip(labels.view(-1), pred):
            if label.item() not in class_correct:
                class_correct[label.item()] = 0
                class_total[label.item()] = 0
            class_correct[label.item()] += (prediction == label).item()
            class_total[label.item()] += 1

        # 统计总体正确预测数量
        correct += pred.eq(labels.view(-1)).sum().item()

    # 计算测试准确率
    over_acc = correct / len(test_loader.dataset)

    # 计算每个类的准确率
    class_accuracies = [class_correct[i] / class_total[i] for i in class_correct if class_total[i] > 0]

    # 计算平均类别准确率（mACC）
    mACC = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0

    return over_acc, mACC

# 记录每次实验的准确率
def record(over_acc,mACC,epoch):
    # 定义文件路径
    file_path = os.path.join(record_folder_path, "record.txt")

    # 获取当前日期和时间
    experiment_date = datetime.now().strftime("%m月%d日,%H:%M:%S")

    # 将数据写入文本文件
    with open(file_path, "a") as f:
        f.write(f"{over_acc:.4f},{mACC:.4f}\t")
        f.write(f"epoch: {epoch}\t")
        f.write(f"Experiment Date: {experiment_date}\n")

# 记录最好的训练结果
def record_bestResult(test_acc,best_epoch,types):
    # 定义文件路径
    file_path = os.path.join(record_folder_path, "bestResult.txt")

    # 获取当前日期和时间
    experiment_date = datetime.now().strftime("%m月%d日,%H:%M:%S")

    # 判断结果的类型，移动光标
    with open(file_path, "r") as f:
        lines = f.readlines()

    if types == "over_acc":
        # 清空第一行到第三行
        lines[:3] = []
        # 将光标移动到第一行开头
        lines.insert(0, "\n")
    elif types == "mACC":
        # 清空第五行以后
        del lines[4:]
        # 将光标移动到第五行开头
        lines.insert(4, "\n")

    # 写入历史最佳准确率的数据
    with open(file_path, "w") as f:
        # 将修改后的内容写回文件
        f.writelines(lines)
        # 写入新内容
        f.write(f"{types}: \t{test_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Experiment Date: {experiment_date}\n")

def save_data_to_txt():
    # 生命使用全局变量
    global epochs, epoch_init, acc_record, best_epoch, best_OA, best_mACC, wait, patience, mark, total_time

    # 定义文件路径
    file_path = os.path.join(record_folder_path, "data.txt")

    # 将要写入的内容准备好
    with open(file_path, 'w') as f:
        f.write(f"epochs = {epochs}\n")
        f.write(f"epoch_init = {epoch_init}\n")
        f.write(f"acc_record = {acc_record}\n")
        f.write(f"best_epoch = {best_epoch}\n")
        f.write(f"best_OA = {best_OA}\n")
        f.write(f"best_mACC = {best_mACC}\n")
        f.write(f"wait = {wait}\n")
        f.write(f"patience = {patience}\n")
        f.write(f"mark = {mark}\n")
        f.write(f"total_time = {total_time}\n")

# 从文件里加载变量
def load_data_from_txt():
    # 定义文件路径
    file_path = os.path.join(record_folder_path, "data.txt")

    # 读取文件内容并解析变量
    with open(file_path, 'r') as f:
        lines = f.readlines()

    variables = {}
    for line in lines:
        key, value = line.strip().split(' = ')
        variables[key] = eval(value)

    return variables

# 加载变量并设置
def load_variables_from_txt():
    global epochs, epoch_init, acc_record, best_epoch, best_OA, best_mACC, wait, patience, mark, total_time
    variables = load_data_from_txt()
    epochs = variables["epochs"]
    epoch_init = variables["epoch_init"]
    acc_record = variables["acc_record"]
    best_epoch = variables["best_epoch"]
    best_OA = variables["best_OA"]
    best_mACC = variables["best_mACC"]
    wait = variables["wait"]
    patience = variables["patience"]
    mark = variables["mark"]
    total_time = variables["total_time"]

# 保留模型，并删除之前的模型,types表示为何种模型
def save_model(model,epoch,types):

    #定义文件名和路径
    file_name = f"{types}_{epoch}.pt"
    file_path = os.path.join(record_folder_path, file_name)

    #先保留当前模型
    torch.save(model.state_dict(), file_path)
    
    #删除上一个模型
    files = os.listdir(record_folder_path)
    for file in files:
        if file.startswith(types) and file.endswith('.pt') and file != file_name:
            file_path = os.path.join(record_folder_path, file)
            # 删除符合条件的文件
            os.remove(file_path)
            break
def load_model():
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(record_folder_path):
        # 检查文件名是否以"latest"开头，并且是以".pt"结尾的文件
        if file_name.startswith("latest") and file_name.endswith(".pt"):
            # 构造完整的文件路径
            file_path = os.path.join(record_folder_path, file_name)
            try:
                # 加载模型
                model = torch.load(file_path)
                print("成功加载模型：", file_path)
                # 找到一个匹配的文件后，结束循环
                return model
            except Exception as e:
                print("加载模型失败：", e)
    else:
        print("未找到以latest开头的.pt文件")

def run_ScanObjectNN():
    # path='/home/tyz/mine/datasets/modelnet/ModelNet40'
    # username = 'taoyingzhi'
    # 生命使用全局变量
    global epochs, epoch_init, acc_record, best_epoch, best_OA, best_mACC, wait, patience, mark, total_time

    username = os.getenv('USER')
    category = {0: 'cnn', 1: 'snn_1', 2: 'snn_2', 3: 'snn_3'}

    # train_dataset, test_dataset = get_dataset(num_points=1024, username=username)
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=2048), num_workers=4,
                            batch_size=32, shuffle=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=2048), num_workers=4,
                            batch_size=32, shuffle=False)
    print("mark# Dataset loaded!!!!!!!")
    # train_loader = DataLoader(train_dataset, 16, shuffle=True)
    # test_loader = DataLoader(test_dataset, 16, shuffle=False)

    # model = PointCNN(train_dataset.num_classes).to(device)
    # model = Spike_PointCNN(train_dataset.num_classes).to(device)
    # from SpikeModel import PointCNN_lif1
    # model = PointCNN_lif1(train_dataset.num_classes).to(device)

    models = []
    # models.append(PointCNN(15).to(device))
    # models.append(PointCNN_lif1(15).to(device))
    # models.append(PointCNN_lif2(15).to(device))
    # models.append(PointCNN_lif3(15).to(device))


    print("mark# Circle loaded!!!!!!!")
    # 加载模型
    model = load_model()
    print("mark# Model loaded!!!!!!!")

    # model = Spike_PointCNN3(train_dataset.num_classes).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    # criterion = torch.nn.KLDivLoss()
    criterion = torch.nn.CrossEntropyLoss()

    print(model)

    # modelSave_path = '/home/{}/savedModels/PointCNN/ScanObjectNN/{}/'.format(username, experiment_category)
    # modelSave_path = '/home/{}/savedModels/SpikePointCNN/Basic/ScanObjectNN/'.format(username)

    # if not os.path.exists(modelSave_path):
    #     os.makedirs(modelSave_path)


    # 加载变量
    load_variables_from_txt()


    for epoch in range(epoch_init, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif (hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available()):
            torch.mps.synchronize()

        t_start = time.perf_counter()

        train(model,train_loader,optimizer)
        over_acc,mACC = test(model,test_loader)
        record(over_acc,mACC)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif (hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available()):
            torch.mps.synchronize()

        t_end = time.perf_counter()
        # if not (epoch % 5):
        #     modelName = 'Epcoh{}.pt'.format(epoch, over_acc)
        #     torch.save(model, modelSave_path + modelName)
        acc_record.append(over_acc)
        # 如果当前over_acc大于最佳准确率
        if over_acc > best_acc:
            best_acc = over_acc
            best_epoch = epoch
            wait = 0
            # 更新记录并保存模型
            record_bestResult(over_acc,epoch,"over_acc")
            save_model(model,epoch,"best_OA")
        
        # 如果当前mACC大于最佳准确率
        if mACC > best_mACC:
            best_mACC = over_acc
            # 更新记录并保存模型
            record_bestResult(mACC,epoch,"mACC")
            save_model(model,epoch,"best_mACC")

        # 每次epoch后输出以下实验信息
        experiment_date = datetime.now().strftime("%m月%d日,%H:%M:%S")

        print(f'Epoch: {epoch:03d}, Test: {over_acc:.4f}, '
            f'Duration: {t_end - t_start:.2f}, '
            f'Best_epoch: {best_epoch}, '
            f'Current_time:{experiment_date}')
        
        # 在每个 epoch 结束后保存模型
        save_model(model,epoch,"latest")
        total_time += (t_end - t_start)

        # 保存所有变量
        save_data_to_txt()


if __name__ == "__main__":
    # 当脚本直接运行时，调用 main 函数
    run_ScanObjectNN()
