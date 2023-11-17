% setenv("MW_NVCC_PATH","/usr/local/cuda-12.3/bin")
% mexcuda('-v', 'mexGPUlapslppot.cu','NVCCFLAGS=-gencode=arch=compute_80,code=sm_80','CFLAGS="\$CFLAGS -DMATLAB_DEFAULT_RELEASE=R2021b"');
%

if 0
  % below load testGPUlapslppot.cu data
  % !nvcc testGPUlapslppot.cu
  % !./a.out 
  %
  
  % load from ./a.out
  mu = readmatrix('mu.txt');
  src = readmatrix('src.txt');
  targ = readmatrix('targ.txt');
  pot = readmatrix('pot.txt');
  N = size(src,2);
  M = size(targ,2);
  
  % compute 
  pot2 = zeros(M,1);
  for j=1:N
    pot2 = pot2 + mu(j)./sqrt((src(1,j) - targ(1,:)').^2+(src(2,j) - targ(2,:)').^2+(src(3,j) - targ(3,:)').^2); 
  end

end

if 0
files = {'mexGPUlapslppot'};
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

pmmops = 9;
sqrtops = 1;

%============= source ~= target =============
%
N = 1e+05;
M = 1e+05;
src = rand(3,N);
targ = rand(3,M); targy = rand(1,M); targz = rand(1,M);
x = rand(1,N);
%
[pot,curuntime] = mexGPUlapslppot(src,targ,x); 
% A = reshape(A(:),N,M)';
disp([' ========== check src ~= targ ==========  ']);
disp([' cuda kernel run time: ',num2str(curuntime),' milliseconds']); % about 140 seconds for 1e+06 to 1e+06 pts
disp([' total plus minus multiplication: ', num2str(pmmops*N*M,'%.3e')]);
disp([' total rsqrt: ', num2str(sqrtops*N*M,'%.3e')]);
disp([' performance (total/run time): ', num2str((pmmops+sqrtops)*N*M/(curuntime/1e+03),'%.3e'), ' FLOPS' ])

%
tic, 
pot2 = zeros(M,1);
for j=1:N
  pot2 = pot2 + x(j)./sqrt((src(1,j) - targ(1,:)').^2+(src(2,j) - targ(2,:)').^2+(src(3,j) - targ(3,:)').^2); 
end
cpuruntime = toc;
disp([' cpu kernel run time: ',num2str(cpuruntime*1e+03),' milliseconds']); % 
disp([' speedup: ',num2str(floor(cpuruntime*1e+03/curuntime)),' times']); % 

%
diff = abs(pot-pot2)/max(abs(pot)); 
disp([' max rel diff between cpu and gpu: ', num2str(max(diff)), ' ']); 
disp([' ']);

% diffA = abs(A-A2); max(diffA(:))

%============= source = target =============
%
N = 1e+04;
src = rand(3,N); 
targ = src;
x = rand(1,N);

%
[pot,curuntime] = mexGPUlapslppot(src,targ,x); 
% A = reshape(A(:),N,M)';
disp([' ========== check src = targ ==========  ']);
disp([' cuda kernel run time: ',num2str(curuntime),' milliseconds']); % about 140 seconds for 1e+06 to 1e+06 pts

tic, 
A = 1./sqrt((src(1,:) - targ(1,:)').^2+(src(2,:) - targ(2,:)').^2+(src(3,:) - targ(3,:)').^2); 
A(diagind(A)) = 0;
pot2 = A*x(:);
cpuruntime = toc;
disp([' cpu kernel run time: ',num2str(cpuruntime*1e+03),' milliseconds']); % 
disp([' speedup: ',num2str(floor(cpuruntime*1e+03/curuntime)),' times']); % 

%
diff = abs(pot-pot2)/max(abs(pot)); 
disp([' max rel diff between cpu and gpu: ', num2str(max(diff)), ' ']); 
disp([' ']);

keyboard

function i = diagind(A)
% function i = diagind(A)
%
% return diagonal indices of a square matrix, useful for changing a diagonal
% in O(N) effort, rather than O(N^2) if add a matrix to A using matlab diag()
%
% barnett 2/6/08

N = size(A,1);
if size(A,2)~=N
  disp('input must be square!');
end
i = sub2ind(size(A), 1:N, 1:N);
end