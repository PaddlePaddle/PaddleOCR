% Generates mat files for loadmat unit tests
% Uses save_matfile.m function
% This is the version for matlab 4

% work out matlab version and file suffix for test files
global FILEPREFIX FILESUFFIX
sepchar = '/';
if strcmp(computer, 'PCWIN'), sepchar = '\'; end
FILEPREFIX = [pwd sepchar 'data' sepchar];
mlv = version;
FILESUFFIX = ['_' mlv '_' computer '.mat'];

% basic double array
theta = 0:pi/4:2*pi;
save_matfile('testdouble', theta);

% string
save_matfile('teststring', '"Do nine men interpret?" "Nine men," I nod.')

% complex
save_matfile('testcomplex', cos(theta) + 1j*sin(theta));

% asymmetric array to check indexing
a = zeros(3, 5);
a(:,1) = [1:3]';
a(1,:) = 1:5;

% 2D matrix
save_matfile('testmatrix', a);

% minus number - tests signed int 
save_matfile('testminus', -1);

% single character
save_matfile('testonechar', 'r');

% string array
save_matfile('teststringarray', ['one  '; 'two  '; 'three']);

% sparse array
save_matfile('testsparse', sparse(a));

% sparse complex array
b = sparse(a);
b(1,1) = b(1,1) + j;
save_matfile('testsparsecomplex', b);

% Two variables in same file
save([FILEPREFIX 'testmulti' FILESUFFIX], 'a', 'theta')

