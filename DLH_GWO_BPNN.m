%DLH-GWO-BPNN核心代码
clear
clc
tic
global SamIn SamOut HiddenUnitNum InDim OutDim TrainSamNum 
%% 导入训练数据
data = xlsread('D:/项目/DLH-GWO-BPNN/数据/方法2/80_90.csv');
[data_m,data_n] = size(data);%获取数据维度

P = 70;  %百分之P的数据用于训练
Ind = floor(P * data_m / 100);

train_data = data(1:Ind,1:end-1)'; 
train_result = data(1:Ind,end)';
n = length(unique(train_result));
train_result = full(ind2vec(train_result,n));

test_data = data(Ind+1:end,1:end-1)';% 利用训练好的网络进行预测
test_result = data(Ind+1:end,end)';
test_result = full(ind2vec(test_result,n));

%% 初始化参数
[InDim,TrainSamNum] = size(train_data);% 学习样本数量
[OutDim,TrainSamNum] = size(train_result);
HiddenUnitNum = 10;                     % 隐含层神经元个数


[SamIn,PS_i] = mapminmax(train_data,0,1);    % 归一化
[SamOut,PS_o] = mapminmax(train_result,0,1);

W1 = HiddenUnitNum*InDim;      % 初始化输入层与隐含层之间的权值
B1 = HiddenUnitNum;          % 初始化输入层与隐含层之间的阈值
W2 = OutDim*HiddenUnitNum;     % 初始化输出层与隐含层之间的权值
B2 = OutDim;                % 初始化输出层与隐含层之间的阈值

L = W1+B1+W2+B2;        %粒子维度
%%优化参数的设定
dim=L; % 优化的参数 number of your variables
for j=1:L
lb(1,j)=-3.55; % 参数取值下界3.55   5.55
ub(1,j)=3.55;
end% 参数取值上界

%%GWO算法初始化
SearchAgents_no=30; % 狼群数量，Number of search agents   50
Max_iteration=2000; % 最大迭代次数，Maximum numbef of iterations   500

lu = [lb .* ones(1, dim); ub .* ones(1, dim)];

% initialize alpha, beta, and delta_pos
Alpha_score=inf; % 初始化Alpha狼的目标函数值，change this to -inf for maximization problems
Alpha_pos=zeros(1,dim); % 初始化Alpha狼的位置
Beta_pos=zeros(1,dim); % 初始化Beta狼的位置
Beta_score=inf; % 初始化Beta狼的目标函数值，change this to -inf for maximization problems

Delta_pos=zeros(1,dim); % 初始化Delta狼的位置
Delta_score=inf; % 初始化Delta狼的目标函数值，change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);
Positions = boundConstraint (Positions, Positions, lu);

% 计算每头狼的适应度
for i=1:size(Positions,1)
    Fit(i) = f(Positions(i,:));
end

% Personal best fitness and position obtained by each wolf   
% 每只狼获得的个人最佳适应度和位置
pBestScore = Fit;
pBest = Positions;
neighbor = zeros(SearchAgents_no,SearchAgents_no);
Convergence_curve=zeros(1,Max_iteration);

l=0; % Loop counter循环计数器

    % Main loop主循环
    while l<Max_iteration  % 对迭代次数循环
        
        for i=1:size(Positions,1)  % 遍历每个狼
            
            % 计算适应度函数值
            fitness=Fit(i);
            
            % Update Alpha, Beta, and Delta
            if fitness<Alpha_score % 如果目标函数值小于Alpha狼的目标函数值
                Alpha_score=fitness; % 则将Alpha狼的目标函数值更新为最优目标函数值，Update alpha
                Alpha_pos=Positions(i,:); % 同时将Alpha狼的位置更新为最优位置
            end
            
            if fitness>Alpha_score && fitness<Beta_score % 如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
                Beta_score=fitness; % 则将Beta狼的目标函数值更新为最优目标函数值，Update beta
                Beta_pos=Positions(i,:); % 同时更新Beta狼的位置
            end
            
            if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score  % 如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
                Delta_score=fitness; % 则将Delta狼的目标函数值更新为最优目标函数值，Update delta
                Delta_pos=Positions(i,:); % 同时更新Delta狼的位置
            end
        end
        
        a=2-l*((2)/Max_iteration); % 对每一次迭代，计算相应的a值，a decreases linearly fron 2 to 0
        
        
       
        
        % Update the Position of search agents including omegas
        for i=1:size(Positions,1) % 遍历每个狼
            for j=1:size(Positions,2) % 遍历每个维度
                
                % 包围猎物，位置更新
                
                r1=rand(); % r1 is a random number in [0,1]
                r2=rand(); % r2 is a random number in [0,1]
                
                A1=2*a*r1-a; % 计算系数A，Equation (3.3)
                C1=2*r2; % 计算系数C，Equation (3.4)
                
                % Alpha狼位置更新
                D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
                X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                
                r1=rand();
                r2=rand();
                
                A2=2*a*r1-a; % 计算系数A，Equation (3.3)
                C2=2*r2; % 计算系数C，Equation (3.4)
                
                % Beta狼位置更新
                D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
                X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2
                
                r1=rand();
                r2=rand();
                
                A3=2*a*r1-a; % 计算系数A，Equation (3.3)
                C3=2*r2; % 计算系数C，Equation (3.4)
                
                % Delta狼位置更新
                D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
                X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3
                
                % 位置更新
                Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
                X_GWO(i,j)=(X1+X2+X3)/3;
                
            end
            X_GWO(i,:) = boundConstraint(X_GWO(i,:), Positions(i,:), lu);
            Fit_GWO(i) = f(X_GWO(i,:));
        end
        % Calculate the candiadate position Xi-DLH
        radius = pdist2(Positions, X_GWO, 'euclidean');    %计算X中任意一个行向量与Y中任意一个行向量的距离,默认采用欧氏距离公式 ，函数的返回值为向量D，D是具有一行，(m*(m-1)/2)列的行向量。    % Equation (10)
        dist_Position = squareform(pdist(Positions));  
        % 把一行（或一列）数据整理成方阵
        r1 = randperm(SearchAgents_no,SearchAgents_no);   %前SearchAgents_no个数中选SearchAgents_no个
        
        for t=1:SearchAgents_no
            neighbor(t,:) = (dist_Position(t,:)<=radius(t,t));
            [~,Idx] = find(neighbor(t,:)==1);                   % Equation (11)
            random_Idx_neighbor = randi(size(Idx,2),1,dim);
            
            for d=1:dim
                X_DLH(t,d) = Positions(t,d) + rand .*(Positions(Idx(random_Idx_neighbor(d)),d)...
                    - Positions(r1(t),d));                      % Equation (12)
            end
            X_DLH(t,:) = boundConstraint(X_DLH(t,:), Positions(t,:), lu);
            Fit_DLH(t) = f(X_DLH(t,:));
        end
        
        % Selection
        tmp = Fit_GWO < Fit_DLH;                                % Equation (13)
        tmp_rep = repmat(tmp',1,dim);
        
        tmpFit = tmp .* Fit_GWO + (1-tmp) .* Fit_DLH;
        tmpPositions = tmp_rep .* X_GWO + (1-tmp_rep) .* X_DLH;
        
        % Updating
        tmp = pBestScore <= tmpFit;                             % Equation (13)
        tmp_rep = repmat(tmp',1,dim);
        
        pBestScore = tmp .* pBestScore + (1-tmp) .* tmpFit;
        pBest = tmp_rep .* pBest + (1-tmp_rep) .* tmpPositions;
        
        Fit = pBestScore;
        Positions = pBest;
        
        
        l=l+1;
        neighbor = zeros(SearchAgents_no,SearchAgents_no);
        Convergence_curve(l)=Alpha_score;
        
    end


x=Alpha_pos;
%%
% x = gb;
W1 = x(1:HiddenUnitNum*InDim);
L1 = length(W1);
W1 = reshape(W1,[HiddenUnitNum, InDim]);
B1 = x(L1+1:L1+HiddenUnitNum)';
L2 = L1 + length(B1);
W2 = x(L2+1:L2+OutDim*HiddenUnitNum);
L3 = L2 + length(W2);
W2 = reshape(W2,[OutDim, HiddenUnitNum]);
B2 = x(L3+1:L3+OutDim)';
HiddenOut = tansig(W1 * SamIn + repmat(B1, 1, TrainSamNum));   % 隐含层网络输出tansig
NetworkOut = W2 * HiddenOut + repmat(B2, 1, TrainSamNum);      % 输出层网络输出
temp1 = softmax(NetworkOut);
Error1 = SamOut - temp1;       % 实际输出与网络输出之差
Forcast_data = mapminmax('reverse',temp1,PS_o);
[val1, index1] = max(Forcast_data,[],1);
[val11, index11] = max(train_result,[],1);
p1 = 0;
for i=1:size(train_result,2)
    if index1(i) == index11(i)
        p1 = p1 + 1;
    end
end
predict1 = p1 / size(train_result,2)

[OutDim,ForcastSamNum] = size(test_result);
SamIn_test= mapminmax('apply',test_data,PS_i); % 原始样本对（输入和输出）初始化
HiddenOut_test = tansig(W1 * SamIn_test + repmat(B1, 1, ForcastSamNum));  % 隐含层输出预测结果
NetworkOut_test = W2 * HiddenOut_test + repmat(B2, 1, ForcastSamNum);          % 输出层输出预测结果
temp2 = softmax(NetworkOut_test);
Forcast_data_test = mapminmax('reverse',temp2,PS_o);

[val2, index2] = max(Forcast_data_test,[],1);
[val22, index22] = max(test_result,[],1);
p2 = 0;
for i=1:size(test_result,2)
    if index2(i) == index22(i)
        p2 = p2 + 1;
    end
end
predict2 = p2 / size(test_result,2)
toc

%% 绘制结果
figure

plot(Convergence_curve,'r')
xlabel('迭代次数')
ylabel('适应度')
title('收敛曲线')


