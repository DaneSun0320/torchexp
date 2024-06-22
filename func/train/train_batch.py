def train_batch(data, criterion, net, device):
    img, label = data
    # 将数据放入device
    img = img.to(device)
    label = label.to(device)
    # 模型推理
    output = net(img)
    # 计算损失
    loss = criterion(output, label)
    # 返回损失
    return loss