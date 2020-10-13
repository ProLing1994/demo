import os 

if __name__ == "__main__":
  input_path = "/home/huanyuan/code/demo/common/common/example/csrc/demo.cpp"
  code_path = "/home/huanyuan/code/demo/common/common/lib/Release/demo"
  output_path = "/home/huanyuan/code/demo/common/common/lib/Release/problem_06/"
  
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  os.system("cp {} {}".format(input_path, output_path))

  input_loop = 20 
  for idx in range(input_loop):
    input_str = input("请输入_{}：".format(idx))
    
    commond = ["echo", "{}".format(input_str.strip()), ">", "{}".format(os.path.join(output_path, str(idx + 1) + '.in'))]
    os.system(" ".join(commond))
    print(commond)
    
    input_list = input_str.strip().split(" ") 
    commond = ["{}".format(code_path)]
    commond.extend(input_list)
    os.system(" ".join(commond))
    commond += [">", "{}".format(os.path.join(output_path, str(idx + 1) + '.out'))]
    os.system(" ".join(commond))
    print(commond)
    