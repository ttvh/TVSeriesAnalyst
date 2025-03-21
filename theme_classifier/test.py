import os
import sys
import pathlib 

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))

print(folder_path) 
print(sys.path)  # In ra danh sách đường dẫn tìm kiếm module