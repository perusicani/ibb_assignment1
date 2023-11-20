from utils import read_from_txt, save_to_txt

test = 3.456
save_to_txt('code/test', [test])
test2 = read_from_txt('code/test')
print(test2[0])
test3 = float(test2[0])
print(type(test3))