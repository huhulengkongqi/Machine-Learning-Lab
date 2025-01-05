import random
import numpy as np
import os  # 确保导入 os 模块
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ---------------------------
# 任务1：BPP的策略学习
# ---------------------------

def generate_bpp_data(num_samples, container_size=(100, 100, 100)):
    """
    生成Bin Packing Problem的合成数据。

    参数：
        num_samples (int): 生成的数据样本数量。
        container_size (tuple): 容器的尺寸，默认为(100, 100, 100)。

    返回:
        List[List[dict]]: 每个样本包含多个物品，每个物品由字典表示。
    """
    data = []
    for _ in range(num_samples):
        items = []
        N = random.randint(10, 50)  # 随机生成物品数量
        for i in range(N):
            # 生成物品尺寸，确保物品尺寸不超过容器尺寸
            size = tuple(random.uniform(1, min(container_size)) for _ in range(3))
            items.append({'id': i, 'size': size})
        data.append(items)
    return data

class BPPDataset(Dataset):
    """
    自定义数据集类，用于加载Bin Packing Problem的数据。
    """
    def __init__(self, data):
        self.data = data  # 数据应为List[List[dict]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        items = self.data[idx]
        sizes = [item['size'] for item in items]
        sizes = torch.tensor(sizes, dtype=torch.float32)

        # 根据体积降序排序，获取排序后的索引
        volumes = [size[0] * size[1] * size[2] for size in sizes]
        sorted_indices = sorted(range(len(volumes)), key=lambda i: volumes[i], reverse=True)

        # 生成配对排名的标签
        pairs = []
        for i in range(len(sorted_indices)):
            for j in range(i+1, len(sorted_indices)):
                pairs.append((sorted_indices[i], sorted_indices[j]))  # (高优先级, 低优先级)

        return sizes, pairs

def collate_fn(batch):
    """
    自定义的collate_fn，用于处理不同长度的样本。

    参数：
        batch (List[Tuple[Tensor, List[Tuple[int, int]]]]): 一个批次的数据。

    返回:
        Tuple[Tensor, Tensor, Tensor]: 尺寸、对索引和标签张量。
    """
    sizes, pairs = zip(*batch)
    # 找到当前批次中最大的物品数量
    max_len = max([s.size(0) for s in sizes])
    # 填充每个样本到max_len
    padded_sizes = []
    all_pairs_a = []
    all_pairs_b = []
    for i, (s, p) in enumerate(zip(sizes, pairs)):
        pad_size = max_len - s.size(0)
        if pad_size > 0:
            pad = torch.zeros((pad_size, 3), dtype=torch.float32)
            s = torch.cat([s, pad], dim=0)
        padded_sizes.append(s)
        for pair in p:
            all_pairs_a.append(pair[0])
            all_pairs_b.append(pair[1])
    # 堆叠成张量
    padded_sizes = torch.stack(padded_sizes, dim=0)  # [batch_size, max_len, 3]
    pair_a = torch.tensor(all_pairs_a, dtype=torch.long)
    pair_b = torch.tensor(all_pairs_b, dtype=torch.long)
    return padded_sizes, pair_a, pair_b

class BPPModel(nn.Module):
    """
    完整的Bin Packing Problem模型，包含编码器和得分预测。
    """
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=1):
        super(BPPModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: [batch_size, max_len, 3]
        batch_size, max_len, _ = x.size()
        x = x.view(-1, x.size(-1))  # [batch_size * max_len, 3]
        scores = self.encoder(x)    # [batch_size * max_len, 1]
        scores = scores.view(batch_size, max_len)  # [batch_size, max_len]
        return scores  # [batch_size, max_len]

def train_model(model, dataloader, epochs=50, lr=0.001):
    """
    训练模型。

    参数：
        model (nn.Module): 需要训练的模型。
        dataloader (DataLoader): 训练数据加载器。
        epochs (int): 训练轮数。
        lr (float): 学习率。

    返回:
        nn.Module: 训练后的模型。
    """
    criterion = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sizes, pair_a, pair_b in dataloader:
            optimizer.zero_grad()
            # 确保batch_size=1
            if dataloader.batch_size > 1:
                raise NotImplementedError("当前训练循环仅支持 batch_size=1。")
            scores = model(sizes)  # [batch_size, max_len]
            scores = scores[0]  # [max_len]
            scores_a = scores[pair_a]  # [num_pairs]
            scores_b = scores[pair_b]  # [num_pairs]
            target = torch.ones_like(scores_a)  # score_a should be greater than score_b
            loss = criterion(scores_a, scores_b, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model

def evaluate_model(model, dataloader):
    """
    评估模型的准确率。

    参数：
        model (nn.Module): 需要评估的模型。
        dataloader (DataLoader): 测试数据加载器。

    返回:
        float: 模型的准确率。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sizes, pair_a, pair_b in dataloader:
            if dataloader.batch_size > 1:
                raise NotImplementedError("当前评估循环仅支持 batch_size=1。")
            scores = model(sizes)  # [batch_size, max_len]
            scores = scores[0]      # [max_len]
            scores_a = scores[pair_a]
            scores_b = scores[pair_b]
            correct += (scores_a > scores_b).sum().item()
            total += pair_a.size(0)
    accuracy = correct / total if total > 0 else 0
    print(f"模型准确率: {accuracy * 100:.2f}%")
    return accuracy

# ---------------------------
# 任务2：神经引导的搜索算法
# ---------------------------

def neural_guided_beam_search(model, items, beam_width=3):
    """
    使用神经网络引导的Beam Search算法选择打包物品。

    参数：
        model (nn.Module): 训练好的策略网络。
        items (List[dict]): 物品列表，每个物品包含'sku_code'和'size'。
        beam_width (int): Beam宽度。

    返回:
        List[dict]: 选择的物品列表。
    """
    model.eval()
    scores = []
    with torch.no_grad():
        # 准备输入
        sizes = torch.tensor([item['size'] for item in items], dtype=torch.float32).unsqueeze(0)  # [1, num_items, 3]
        item_scores = model(sizes).squeeze(0)  # [num_items]
        scores = item_scores.numpy()
    # 按得分排序
    sorted_indices = np.argsort(-scores)  # 从高到低
    sorted_items = [items[i] for i in sorted_indices]
    # 基于体积和最大维度进一步排序
    sorted_items = sort_items_by_volume_and_dimension(sorted_items)
    return sorted_items[:beam_width]  # 返回beam_width个物品

def sort_items_by_volume_and_dimension(items):
    """
    根据物品的体积和最大单维度进行排序。

    参数：
        items (List[dict]): 物品列表，每个物品包含'size'。

    返回:
        List[dict]: 排序后的物品列表。
    """
    def sort_key(item):
        volume = item['size'][0] * item['size'][1] * item['size'][2]
        max_dimension = max(item['size'])
        return (-volume, -max_dimension)
    
    return sorted(items, key=sort_key)

# ---------------------------
# 任务3：电子商务打包问题
# ---------------------------

CONTAINER_SIZES = [
    (35, 23, 13),
    (37, 26, 13),
    (38, 26, 13),
    (40, 28, 16),
    (42, 30, 18),
    (42, 30, 40),
    (52, 40, 17),
    (54, 45, 36)
]

def pack_ecommerce_order(model, order_items, beam_width=3):
    """
    为单个订单打包物品，使用多个容器。

    参数：
        model (nn.Module): 训练好的策略网络。
        order_items (List[dict]): 订单中的物品，每个物品包含'sta_code', 'sku_code', '长(CM)', '宽(CM)', '高(CM)', 'qty'。
        beam_width (int): Beam宽度。

    返回:
        Tuple[List[dict], float]: 容器列表和打包效率比率。
    """
    # 展开物品，根据数量
    items = []
    for item in order_items:
        for _ in range(int(item['qty'])):
            items.append({
                'sku_code': item['sku_code'],
                'size': (item['长(CM)'], item['宽(CM)'], item['高(CM)'])
            })
    # 使用神经引导的Beam Search选择物品的优先级
    sorted_items = neural_guided_beam_search(model, items, beam_width=beam_width)

    # 自定义多容器装箱算法
    containers = []

    for item in sorted_items:
        placed = False
        for container in containers:
            position, rotation = place_item_in_container(container, item)
            if position is not None:
                container['items'].append({
                    'sku_code': item['sku_code'],
                    'size': item['size'],
                    'position': position,
                    'rotation': rotation
                })
                placed = True
                break
        if not placed:
            # 尝试开启一个新的容器
            for size in CONTAINER_SIZES:
                container = {
                    'container_size': size,
                    'items': []
                }
                position, rotation = place_item_in_container(container, item)
                if position is not None:
                    container['items'].append({
                        'sku_code': item['sku_code'],
                        'size': item['size'],
                        'position': position,
                        'rotation': rotation
                    })
                    containers.append(container)
                    placed = True
                    break
            if not placed:
                print(f"无法放置物品: {item['sku_code']} 尺寸: {item['size']}")

    # 计算装箱效率
    total_packed_volume = sum([item['size'][0] * item['size'][1] * item['size'][2] for container in containers for item in container['items']])
    total_container_volume = sum([c['container_size'][0] * c['container_size'][1] * c['container_size'][2] for c in containers])
    ratio = total_packed_volume / total_container_volume if total_container_volume > 0 else 0

    return containers, ratio

def place_item_in_container(container, item, step=0.1):
    """
    尝试将物品放置在容器中，优先从左下角开始。

    参数：
        container (dict): 容器信息，包括尺寸和已放置的物品。
        item (dict): 物品信息，包括SKU和尺寸。
        step (float): 步长，用于位置遍历。

    返回:
        Tuple[Tuple[float, float, float], bool]: 物品的位置和是否旋转。
    """
    container_size = container['container_size']
    placed_items = container['items']
    # 尝试所有旋转方式（6种）
    rotations = [
        (item['size'][0], item['size'][1], item['size'][2]),
        (item['size'][0], item['size'][2], item['size'][1]),
        (item['size'][1], item['size'][0], item['size'][2]),
        (item['size'][1], item['size'][2], item['size'][0]),
        (item['size'][2], item['size'][0], item['size'][1]),
        (item['size'][2], item['size'][1], item['size'][0]),
    ]
    # 按照体积从大到小的顺序尝试旋转方式
    rotations = sorted(rotations, key=lambda x: x[0]*x[1]*x[2], reverse=True)
    for rotated_size in rotations:
        if all(rotated_size[i] <= container_size[i] for i in range(3)):
            # 尝试从左下角开始放置
            for x in np.arange(0, container_size[0] - rotated_size[0] + step, step):
                for y in np.arange(0, container_size[1] - rotated_size[1] + step, step):
                    # 找到当前位置下的最低z坐标
                    z = find_lowest_z(x, y, rotated_size, placed_items)
                    if z is not None and z + rotated_size[2] <= container_size[2]:
                        if not check_overlap(x, y, z, rotated_size, placed_items):
                            return (round(x, 2), round(y, 2), round(z, 2)), rotated_size != item['size']
    return None, False

def find_lowest_z(x, y, size, placed_items):
    """
    在给定的x和y坐标下，找到物品放置的最低z坐标。

    参数：
        x (float): x坐标。
        y (float): y坐标。
        size (tuple): 物品的尺寸。
        placed_items (List[dict]): 已放置的物品列表。

    返回:
        float: 物品放置的z坐标。
    """
    max_z = 0
    for item in placed_items:
        ix, iy, iz = item['position']
        iw, ih, id_ = item['size']
        # 检查物品是否在x和y的覆盖范围内
        if (x < ix + iw and x + size[0] > ix and
            y < iy + ih and y + size[1] > iy):
            max_z = max(max_z, iz + id_)
    return max_z

def check_overlap(x, y, z, size, placed_items):
    """
    检查新放置的物品是否与已放置的物品重叠。

    参数：
        x, y, z (float): 新物品的位置。
        size (tuple): 新物品的尺寸。
        placed_items (List[dict]): 已放置的物品列表。

    返回:
        bool: 是否重叠。
    """
    for item in placed_items:
        ix, iy, iz = item['position']
        iw, ih, id_ = item['size']
        if (x < ix + iw and x + size[0] > ix and
            y < iy + ih and y + size[1] > iy and
            z < iz + id_ and z + size[2] > iz):
            return True
    return False

def read_task3_csv(file_path):
    """
    读取task3.csv文件，返回一个订单字典。

    参数：
        file_path (str): CSV文件路径。

    返回:
        Dict[str, List[dict]]: 订单字典，键为订单号，值为物品列表。
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    orders = {}
    for _, row in df.iterrows():
        sta_code = row['sta_code']
        if sta_code not in orders:
            orders[sta_code] = []
        orders[sta_code].append({
            'sta_code': row['sta_code'],
            'sku_code': row['sku_code'],
            '长(CM)': float(row['长(CM)']),
            '宽(CM)': float(row['宽(CM)']),
            '高(CM)': float(row['高(CM)']),
            'qty': int(row['qty'])
        })
    return orders

def process_task3(model, csv_file_path):
    """
    处理任务3：电子商务打包问题。

    参数：
        model (nn.Module): 训练好的策略网络。
        csv_file_path (str): task3.csv文件路径。
    """
    orders = read_task3_csv(csv_file_path)
    all_ratios = []
    for sta_code, items in orders.items():
        containers, ratio = pack_ecommerce_order(model, items, beam_width=3)
        all_ratios.append(ratio)
        print(f"\n订单号: {sta_code}")
        print(f"打包效率: {ratio * 100:.2f}%")
        for idx, container in enumerate(containers):
            print(f"\n容器 {idx+1}:")
            print(f"尺寸: {container['container_size']}")
            for item_idx, item in enumerate(container['items']):
                print(f"  物品 {item_idx+1}: SKU {item['sku_code']}, 尺寸 {item['size']}, 位置 {item['position']}, 旋转: {item['rotation']}")
    # 计算总体打包效率
    overall_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 0
    print(f"\n总体打包效率: {overall_ratio * 100:.2f}%")

# ---------------------------
# 主函数
# ---------------------------

def main():
    # 生成合成数据
    print("生成训练数据...")
    train_data = generate_bpp_data(1000)
    test_data = generate_bpp_data(200)

    # 创建数据集和数据加载器
    train_dataset = BPPDataset(train_data)
    test_dataset = BPPDataset(test_data)
    # 设置batch_size=1以简化配对索引处理
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    input_dim = 3
    hidden_dim = 128
    output_dim = 1
    model = BPPModel(input_dim, hidden_dim, output_dim)

    # 训练模型
    print("训练模型...")
    trained_model = train_model(model, train_loader, epochs=50, lr=0.001)

    # 评估模型
    print("评估模型...")
    evaluate_model(trained_model, test_loader)

    # 处理任务3
    print("处理任务3：电子商务打包问题...")
    csv_file_path = 'task3.csv'  # 请确保task3.csv文件在当前目录下
    if os.path.exists(csv_file_path):
        process_task3(trained_model, csv_file_path)
    else:
        print(f"文件 {csv_file_path} 未找到，请确保文件存在于当前目录。")

if __name__ == "__main__":
    main()
