class FindPath:
    """
    用于寻找一定深度下的文件路径
    由于pp分析的结果导致多级目录出现，后续处理的文件位置较深和分散
    存在多级目录，因此需要使用递归的方式
    
    """
    def __init__(self):
        pass

    def get_floder_tree(self, base_dir,sub_dir, folder_tree):
        """用于获取base_dir的目录结构，
        输入base_dir，获取下面的各级子目录内容并输出
        用户输入target_folder
        展示目录树的结构，
        然后递归获取target_folder下的所有文件路径"""
        return folder_tree
    

    def get_target_folder(self, base_dir, target_folder):

        base_dir = get_folder_tree(base_dir, target_folder)
        return target_folder

    def find_json_path(self, base_dir, target_folder, target_file):
        """
        获取指定的json文件路径
        """
        pass

    def find_txt_path(self, base_dir, target_folder, target_file):
        """
        
        """
        pass

    def find_sub_path(self, base_dir, target_folder, target_file):
    
        pass

    def find_img_path(self, base_dir, target_folder, target_file):
        pass

    def get_overall_path(self, base_dir, target_folder, target_file):
        pass

    def get_part_path(self, base_dir, target_folder, target_file):
        pass