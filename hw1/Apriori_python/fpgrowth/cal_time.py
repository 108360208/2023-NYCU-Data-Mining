def cal_speed_up(data):
    for key, config_data in data.items():  # 使用.items()来迭代字典的键值对
        for minsup, (step2, step3) in config_data.items():
            speed_up = (step2 - step3) / step2 * 100
            print(f"Configuration for dataset {key} Minsup : {minsup}")
            print(f"The performance (on efficiency) improvements is {speed_up}%")

# data = {"A": {
#     "0.2": [112.80, 0.69],
#     "0.5": [4.77, 0.19],
#     "1.0": [1.78, 0.13]
# }}
# data = {"B": {
#     "0.15": [5476.077, 56.791],
#     "0.2": [2856.707, 37.043],
#     "0.5": [926.0249, 21.616]
# }}
data = {"C": {
    "1.0": [4432.618, 601.729],
    "2.0": [1563.794, 277.869],
    "3.0": [572.754, 92.941]
}}
cal_speed_up(data)

# data = {"A":{
#     "0.2":[112.80, ],
#     "0.5":[4.77,],
#     "1.0":[1.78,]
# }}