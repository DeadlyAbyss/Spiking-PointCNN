from Dataloader.ModelNet40 import get_dataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F

# from SpikeModel import Spike_PointCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 指定使用的显卡编号
device = torch.device("cuda:0")  # 使用第一块显卡
# device = torch.device("cpu")

import os
import time
from SpikeModel import Spike_PointCNN
from datetime import datetime
from SpikeModel import *
# from Model import *
# from modified_model import PointCNN
# from Model import PointCNN


#实验结果的保存路径
record_folder_path = "Result/Basic/ModelNet40"
# 检查路径是否存在
if not os.path.exists(record_folder_path):
    # 如果路径不存在，则创建路径
    os.makedirs(record_folder_path)
    print(f"路径 '{record_folder_path}' 不存在，已创建成功。")
else:
    print(f"路径 '{record_folder_path}' 已存在。")


# 一些变量
epochs = 600
# epochs=2
epoch_init = 0
acc_record = []
acc_record.append(-1)
best_epoch = -1
best_OA = -1
best_mACC = -1
# best_model = model
wait = 0
patience = 50
mark = 0
# last_acc = 0
total_time = 0


# 训练模型
def train(model,train_loader,optimizer):
    model.train()

    for data in train_loader:
        # print(type(data))
        # print(data.shape)
        # exit()
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

# 计算over_acc和mACC
def test(model, test_loader, test_dataset):
    model.eval()

    correct_per_class = [0] * test_dataset.num_classes
    total_per_class = [0] * test_dataset.num_classes
    total_correct = 0
    total_samples = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data.pos, data.batch).max(1)[1]
        correct = pred.eq(data.y).to(torch.float32)

        for i in range(len(data.y)):
            class_label = data.y[i].item()
            correct_per_class[class_label] += correct[i].item()
            total_per_class[class_label] += 1

        total_correct += correct.sum().item()
        total_samples += len(data.y)

    class_acc = [correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0
                 for i in range(len(correct_per_class))]
    mACC = sum(class_acc) / len(class_acc)
    over_acc = total_correct / total_samples

    return over_acc,mACC

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

    # 如果文件不存在，则创建文件
    if not os.path.exists(file_path):
        with open(file_path, 'w'):
            pass  # 不做任何操作，只创建空文件

    # 以读取模式打开文件，并读取所有行（如果需要）
    with open(file_path, 'r') as f:
        lines = f.readlines()  # 读取所有行内容
        
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
                print("成功加载历史模型：", file_path)
                # 加载变量
                load_variables_from_txt()
                # 找到一个匹配的文件后，结束循环
                return model
            except Exception as e:
                print("加载历史模型失败：", e)
    else:
        print("未找到以latest开头的.pt文件,尝试读取原始模型")
        # print("mark# Model loaded!!!!!!!")
        try:
            # 加载模型
            model = Spike_PointCNN(train_dataset.num_classes).to(device)
            print("成功加载原始模型：")
            # 找到一个匹配的文件后，结束循环
            return model
        except Exception as e:
            print("加载原始模型失败：", e)

# 主函数
def run_ModelNet():

    # 生命使用全局变量
    global epochs, epoch_init, acc_record, best_epoch, best_OA, best_mACC, wait, patience, mark, total_time
    # path='/home/tyz/mine/datasets/modelnet/ModelNet40'
    # username = 'taoyingzhi'
    username = os.getenv('USER')
    category = {0: 'cnn', 1: 'snn_1', 2: 'snn_2', 3: 'snn_3'}

    # 声明为全局变量
    global train_dataset, test_dataset

    train_dataset, test_dataset = get_dataset(num_points=2048, username=username)
    print("mark# Dataset getted!!!!!!!")

    train_loader = DataLoader(train_dataset, 32, shuffle=True,num_workers=4)
    test_loader = DataLoader(test_dataset, 32, shuffle=False,num_workers=4)

    print("mark# Dataset loaded!!!!!!!")

    # model = PointCNN(train_dataset.num_classes).to(device)
    # model = Spike_PointCNN(train_dataset.num_classes).to(device)
    # from SpikeModel import PointCNN_lif1
    # model = PointCNN_lif1(train_dataset.num_classes).to(device)
    models = []
    models.append(Spike_PointCNN(train_dataset.num_classes).to(device))
    # models.append(PointCNN(train_dataset.num_classes).to(device))
    # models.append(PointCNN_lif1(train_dataset.num_classes).to(device))
    # models.append(PointCNN_lif2(train_dataset.num_classes).to(device))
    # models.append(PointCNN_lif3(train_dataset.num_classes).to(device))

    # print("mark# Circle loaded!!!!!!!")
    # model = Spike_PointCNN(train_dataset.num_classes).to(device)
    # print("mark# Model loaded!!!!!!!")

    # 加载模型
    model = load_model()

    # model = Spike_PointCNN3(train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    # criterion = torch.nn.KLDivLoss()
    criterion = torch.nn.CrossEntropyLoss()

    print(model)

    #modelSave_path = '/home/{}/savedModels/PointCNN/{}/'.format(username, experiment_category)
    # modelSave_path = '/home/{}/savedModels/SpikePointCNN/Basic/ModelNet40/'.format(username)
    # if not os.path.exists(modelSave_path):
    #     os.makedirs(modelSave_path)


    
    # 从上次的位置开始训练
    for epoch in range(epoch_init, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif (hasattr(torch.backends, 'mps')
            and torch.backends.mps.is_available()):
            torch.mps.synchronize()

        t_start = time.perf_counter()

        train(model,train_loader,optimizer)
        over_acc,mACC = test(model,test_loader,test_dataset)
        record(over_acc,mACC,epoch)

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
        if over_acc > best_OA:
            best_OA = over_acc
            best_epoch = epoch
            wait = 0
            # 更新记录并保存模型
            record_bestResult(over_acc,epoch,"over_acc")
            save_model(model,epoch,"best_OA")
        # 不知道干啥的，不敢动
        elif not mark:
            wait += 1
            if wait >= patience:
                modelName = f'best_Epcoh{epoch:3d}.pt'
                torch.save(model, record_folder_path + modelName)
                mark = 1
                print("Early stopping at best_epoch:", best_epoch)
                print("Early stopping at best_OA:  ", best_OA)

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
        
        # # 训练次数达到最大值
        # if epoch == epochs:
        #     modelName = f'Epcoh{epoch:3d}.pt'
        #     torch.save(model, modelSave_path + modelName)

        # 在每个 epoch 结束后保存模型
        save_model(model,epoch,"latest")
        total_time += (t_end - t_start)

        # 保存所有变量
        save_data_to_txt()

if __name__ == "__main__":
    # 当脚本直接运行时，调用 main 函数
    run_ModelNet()
