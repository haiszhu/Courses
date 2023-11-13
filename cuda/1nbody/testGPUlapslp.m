%
%

if 0
files = {'mexGPUlapslp'};
cmd1 = ['/usr/local/cuda-12.3/bin/nvcc -c --compiler-options=-D_GNU_SOURCE,-DMATLAB_MEX_FILE' ...
        ' -I"/usr/local/cuda-12.3/include"' ...
        ' -I"/usr/local/MATLAB/R2022a/extern/include"' ...
        ' -I"/usr/local/MATLAB/R2022a/simulink/include"' ...
        ' -I"/usr/local/MATLAB/R2022a/toolbox/parallel/gpu/extern/include/"' ...
        ' -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70  -gencode arch=compute_80,code=sm_80' ...
        ' -std=c++11 --compiler-options=-ansi,-fexceptions,-fPIC,-fno-omit-frame-pointer,-lcufft,-lcufftw,-lcublas,-pthread -O0 -DNDEBUG' ...
        ' ' files{1} '.cu -o ' files{1} '.o'];

cmd2 = ['/usr/bin/g++ -pthread -Wl,--no-undefined -Wl,--no-as-needed -shared -O' ...
        ' -Wl,--version-script,"/usr/local/MATLAB/R2022a/extern/lib/glnxa64/mexFunction.map"' ...
        ' ' strcat(files{1},'.o') ' -ldl' ...
        ' /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcusparse.so' ... % -lcusparse
        ' /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublas_static.a' ... % -lcublas_static
        ' /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcusparse_static.a' ... % -lcusparse_static
        ' /usr/local/cuda-12.3/targets/x86_64-linux/lib/libculibos.a' ... % -lculibos'
        ' -L/usr/local/cuda-12.3/lib64 -Wl,-rpath-link,/usr/local/MATLAB/R2022a/bin/glnxa64' ...
        ' -L"/usr/local/MATLAB/R2022a/bin/glnxa64" -lmx -lmex -lmat -lm -lstdc++ -lmwgpu -lcufft -lcufftw -lcublas -lcublasLt' ...
        ' /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudart.so' ... % /usr/local/MATLAB/R2016a/bin/glnxa64/libcudart.so.7.5
        ' -o ' files{1} '.mexa64'];

system(cmd1);
system(cmd2);
end

%
N = 20000;
M = 1000;
srcx = rand(1,N); srcy = rand(1,N); srcz = rand(1,N);
targx = rand(1,M); targy = rand(1,M); targz = rand(1,M);
x = rand(1,N);

%
tic, 
y = mexGPUlapslp(srcx,srcy,srcz,targx,targy,targz,x); 
% A = reshape(A(:),N,M)';
toc

%
tic, 
A2 = 1./sqrt((srcx - targx(:)).^2+(srcy - targy(:)).^2+(srcz - targz(:)).^2); 
y2 = A2*x(:);
toc

%
diff = abs(y-y2); max(diff)
% diffA = abs(A-A2); max(diffA(:))

keyboard