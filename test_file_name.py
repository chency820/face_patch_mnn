
data_root_path = ''
path = "S129/002/S129_002_00000011.png"
base_path = path[:-12]
# if neutral expression included:
# path_neu = data_root_path + base_path + '00000001.png'
path_num_part = path[-12:-4]
print(len(str(int(path_num_part))))
path_last3 = data_root_path + base_path + '0' * (8 - len(str(int(path_num_part) - 2))) + str(int(path_num_part) - 2) + '.png'
path_last2 = data_root_path + base_path + '0' * (8 - len(str(int(path_num_part) - 1))) + str(int(path_num_part) - 1) + '.png'
path_last1 = data_root_path + path
print(path_last3, path_last2, path_last1)