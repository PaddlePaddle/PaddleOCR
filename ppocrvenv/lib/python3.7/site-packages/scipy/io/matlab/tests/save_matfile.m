function save_matfile(test_name, v)
% saves variable passed in m with filename from prefix
  
global FILEPREFIX FILESUFFIX
eval([test_name ' = v;']);
save([FILEPREFIX test_name FILESUFFIX], test_name)