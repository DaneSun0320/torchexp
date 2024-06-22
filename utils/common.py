def snake_to_pascal(name):
    """
    将蛇形命名法的字符串转换为大驼峰命名法。

    参数：
    name (str): 蛇形命名法的字符串

    返回：
    str: 大驼峰命名法的字符串
    """
    if '_' not in name:
        return name
    # 使用 split('_') 将字符串分割成单词列表
    words = name.split('_')
    # 将每个单词的首字母大写，然后合并
    pascal_case = ''.join(word.capitalize() for word in words)
    return pascal_case
