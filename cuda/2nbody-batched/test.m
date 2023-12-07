%
%
clear all

load('U.txt','U');
load('Xt.txt','Xt');
load('trg_cnt.txt','trg_cnt');
load('trg_dsp.txt','trg_dsp');
load('F.txt','F');
load('Xs.txt','Xs');
load('src_cnt.txt','src_cnt');
load('src_dsp.txt','src_dsp');
load('trg_src_lst.txt','trg_src_lst');

% load('trg_cnt_cumsum.txt');
% load('trg_box_lst.txt');
% load('trg_src_lstmat.txt');

% Xt vector containing the target particle locations in array-of-struct (AoS) order {x1,y1,z1, ...,xn,yn,zn}.
Xt = reshape(Xt,3,[]);

% trg_cnt vector containing the number of target particles in each leaf-node of the tree.
trg_cnt;

% trg_dsp vector containing the offset of target particles in each leaf-node of the tree.
trg_dsp;

% F vector containing the densities at each source particle.
F;

% Xs vector containing the source particle locations in array-of-struct (AoS) order.
Xs = reshape(Xs,3,[]);

% src_cnt vector containing the number of source particles in each leaf-node of the tree.
src_cnt;

% src_dsp vector containing the offset of source particles in each leaf-node of the tree.
src_dsp;

% trg_src_lst list of pairs of target and source leaf-node indices in the interaction list. Assume sorted by targets.
trg_src_lst;

figure(1),clf,
% plot3(Xt(1,:),Xt(2,:),Xt(3,:),'.'); axis equal, hold on
% plot3(Xt(1,1:100),Xt(2,1:100),Xt(3,1:100),'o');
% plot3(Xt(1,end/2+1:end/2+100),Xt(2,end/2+1:end/2+100),Xt(3,end/2+1:end/2+100),'o');
% plot3(Xt(1,end-100:end),Xt(2,end-100:end),Xt(3,end-100:end),'o');
for k=1:numel(trg_cnt)-1
  if trg_cnt(k)
    tmpidx = trg_dsp(k)+1:trg_dsp(k+1);
    tmpXt = Xt(:,tmpidx);
    plot3(tmpXt(1,:),tmpXt(2,:),tmpXt(3,:),'.'), hold on
%     pause(0.1);
  end
end
axis equal

DIM = 3;
assert(numel(U) == (trg_dsp(end)-trg_cnt(end)))
U2 = zeros(size(U));
U3 = zeros(size(U));
tic
for kk=1:size(trg_src_lst,1) % loop over the interaction list
  trg_src = trg_src_lst(kk,:); % interaction list
  trg_node = trg_src(1); % index of target node
  src_node = trg_src(2); % index of source node
  trg_offset = trg_dsp(trg_node+1); % offset for target particles
  src_offset = src_dsp(src_node+1); % offset for source particles
  Nt = trg_cnt(trg_node+1); % number of target particles
  Ns = src_cnt(src_node+1); % number of source particles
  
  for t = 1:Nt
    for s = 1:Ns
      r2 = 0;
      for k = 1:DIM
        dx = Xt(k,trg_offset+t) - Xs(k,src_offset+s);
        r2 = r2 + dx*dx;
      end
      if r2>0
        U2(trg_offset+t) = U2(trg_offset+t) + F(src_offset+s) / sqrt(r2);
      end
    end
    U3(trg_offset+t) = U3(trg_offset+t) + 1;
  end
  % what pair of target & source
  Xtkk = Xt(:,trg_offset+(1:Nt));
  Xskk = Xs(:,src_offset+(1:Ns));
%   figure(),plot3(Xtkk(1,:),Xtkk(2,:),Xtkk(3,:),'.'); axis equal, hold on
%   plot3(Xskk(1,:),Xskk(2,:),Xskk(3,:),'o')
end
toc

U4 = zeros(size(U));

%%% compute two additional info to allow looping over target op
% Xt vector containing the target particle locations in array-of-struct (AoS) order {x1,y1,z1, ...,xn,yn,zn}.
% Xt = reshape(Xt,3,[]);trg_cnt_cumsum = cumsum(trg_cnt);
% map trg to its box
trg_cnt_cumsum = cumsum(trg_cnt);
ibox = 1; % start from 1st box
trg_box_lst = zeros(size(Xt,2),1);
for t = 1:size(Xt,2)
  % which box does this target belong to?
  while t > trg_cnt_cumsum(ibox)
    ibox = ibox + 1;
  end
  trg_box_lst(t) = ibox-1;
  % when I do it in parallel, how to map target index to box index
  % does not seem to be trivial ... coord -> box index, isn't this tree construction ?
end
% then map trg box to its neighbor boxes
trg_src_lstmat = -ones(trg_box_lst(end),2*DIM^3); % largest num of neighbor boxes?
row_idx0 = trg_box_lst(1); % probably just 1st and last entry
col_idx = 0;
for k = 1:size(trg_src_lst,1)
  row_idx = trg_src_lst(k,1);
  if row_idx == row_idx0 % current row
    col_idx = col_idx + 1; 
    trg_src_lstmat(row_idx,col_idx) = trg_src_lst(k,2); % basically copy src box index to this interaction matrix list
  else % should go to next row
    row_idx0 = row_idx;
    col_idx = 1; % reset column index for next row copy
    trg_src_lstmat(row_idx,col_idx) = trg_src_lst(k,2);
  end
end

tic
for t = 1:size(Xt,2)
  trg_node = trg_box_lst(t);
  for kk=1:2*DIM^3
    src_node = trg_src_lstmat(trg_node,kk);
    if src_node>0
      src_offset = src_dsp(src_node+1); % offset for source particles
      Ns = src_cnt(src_node+1); % number of source particles
      for s = 1:Ns
        r2 = 0;
        for k = 1:DIM
          dx = Xt(k,t) - Xs(k,src_offset+s);
          r2 = r2 + dx*dx;
        end
        if r2>0
          U4(t) = U4(t) + F(src_offset+s) / sqrt(r2);
        end
      end
    end
  end
end
toc


keyboard
