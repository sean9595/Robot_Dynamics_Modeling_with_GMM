%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc; clear; 
addpath('m_fcts/'); 
%%
% 데이터 로드
simout=sim('TDoFs_mass_CL.slx');
trq_in=simout.ScopeData2{1}.Values.Data;
angle_out=simout.ScopeData{1}.Values.Data;
vel_out=simout.ScopeData{2}.Values.Data;
acc_out=simout.ScopeData{3}.Values.Data;
%%
angle_out_J01 = angle_out(1000:10001,1);
angle_out_J02 = angle_out(1000:10001,2);
acc_out_J01 = acc_out(1000:10001,1);
acc_out_J02 = acc_out(1000:10001,2);
trq_in_J01 = trq_in(1000:10001,1);
trq_in_J02 = trq_in(1000:10001,2);
%%
% n x m 행렬, n은 데이터의 차원, m은 데이터의 길이
Data_temp = [angle_out_J01';angle_out_J02';acc_out_J01';acc_out_J02';trq_in_J01';trq_in_J02']; 
Data = gpuArray(Data_temp);
%%
% load('D:\data_archive\Ref_Data\Jan_Peters\sarcos_inv.mat')
% Joint_Positions = sarcos_inv(1:15001,1:7);
% Joint_Velocities = sarcos_inv(1:15001,8:14);
% Joint_Accelerations = sarcos_inv(1:15001,15:21);
% Joint_Torques = sarcos_inv(1:15001,22:28);
% %%
% % n x m 행렬, n은 데이터의 차원, m은 데이터의 길이
% Data_temp = [Joint_Positions';Joint_Accelerations';Joint_Torques']; 
% Data = gpuArray(Data_temp);
%%
% d_new = fillmissing(d,'next');
% 
% % Data = [angle_out';acc_out';trq_in'];    
% Data = [angle_out';d_new]; 
% Data = [angle_out';d];
size(Data);

% 데이터의 길이 (데이터 포인트의 개수)
nbData = size(Data,2);
% nbData_noisy = size(Data,2);
% 데이터의 차원
model.nbVar = size(Data,1); %Number of variables [x1,x2]
% model_noisy.nbVar = size(Data_noisy,1) %Number of variables [x1,x2]
%%
% figure(4)
hold on
% plot3(Data(1,:),Data(3,:),Data(5,:),'color',"#727272");  
% plot3(Data(2,:),Data(4,:),Data(6,:),'k');
plot3(Data(1,:),Data(3,:),Data(5,:));  
plot3(Data(2,:),Data(4,:),Data(6,:));  
xlabel('Joint position (rad)', 'FontName','times new roman', 'FontSize',25);
ylabel('Joint acceleration (rad/s^2)', 'FontName','times new roman', 'FontSize',25);
zlabel('Torque (Nm)','FontName','times new roman', 'FontSize',25); 
legend('Joint 1 state map','Joint 2 state map','FontName','times new roman', 'FontSize',15)
grid on
hold off
%% 1-2. GMM setup
% 1) Gaussian distribution 개수 결정
% 2) GMM parameter initialization
%

model.nbStates = 50;
% model_noisy.nbStates = 15;

% % Prior(i): i번째 State (Gaussian distribution) 별 weight.
%     % K개의 State가 있다면 Prior는 1 x K 행렬
%     % 합이 1이 되도록 초기화: Obvious한 조건
%     % 이 예제에서는 동일한 값(0.5, 0.5)을 가지도록 지정
% model.Priors = ones(1,model.nbStates)/model.nbStates;
% 
% % Mu(:,i): i번째 State (Gaussian distribution) 별 Mu 값.
%     % N차원 데이터가 있다면 Mu는 N x K 행렬
%     % 이 예제에서는 임의의 값 지정
%     % Mu [angle_mean; velocity_mean; acc_mean; torque_mean]
% model.Mu = [-1.5 1.5 -.5  .0 .0; -1e-5 1e-5 -1  1 .0];
% 
% % Sigma(:,:,i): i번째 State (Gaussian distribution) 별 Sigma 값.
%     % N차원 데이터가 있다면 Sigma는 N x N x K 행렬
%     % 이 예제에서는 임의의 값 지정
% model.Sigma = repmat(diag([1, 1]),[1 1 model.nbStates]);
model = init_GMM_kmeans(Data, model);
model_KNN = init_GMM_kmeans(Data, model);
% model_noisy = init_GMM_kmeans(Data_noisy, model_noisy);
%% 1-3. GMM fitting with EM method
% 1-3-1. EM parameter designation
% 최소 iteration 수
nbMinSteps = 10; %50->10

% 최대 iteration 수.
% Converge하지 않아도 해당 iteration에 도달하면 중지
nbMaxSteps = 1000;

% Converge 판단 기준값
% iteration 후 log-likelihood 변화량이 해당 값보다 작으면 중지
maxDiffLL = 1E-8;

% Numerical error를 방지하기 위해 estimated covariance 행렬의
% 대각성분에 더해주는 작은 양의 스칼라 값
% Optional 한 값으로, 0으로 두어도 됨
diagRegularizationFactor = 1E-4; %?
%%
% 1-3-2. EM iteration:Expection step, Maximization step
for nbIter=1:nbMaxSteps
    % E step
    L = zeros(model.nbStates,size(Data,2));
    for i=1:model.nbStates
        %GMM-Basics 문서에 Gaussian mixture를 정의하는 함수
        %mvnpdf는 MultiVariate Normal Probability Density Function (MATL
        % AB 자체함수)
    	L(i,:) = model.Priors(i) * (mvnpdf(Data', model.Mu(:,i)', model.Sigma(:,:,i)'))';
    end
    GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
    GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
    
    % M step
    for i=1:model.nbStates
	    %Update Priors
	    model.Priors(i) = sum(GAMMA(i,:)) / nbData;
	    
	    %Update Mu
	    model.Mu(:,i) = Data * GAMMA2(i,:)';
	    
	    %Update Sigma
	    DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
	    model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(size(Data,1)) * diagRegularizationFactor;
    end

    % log-likelihood 계산
	LL(nbIter) = sum(log(sum(L,1))) / nbData;
    
	% Iteration 중단 여부 판단
    if nbIter>nbMinSteps
	    if LL(nbIter)-LL(nbIter-1)<maxDiffLL || nbIter==nbMaxSteps-1
		    disp(['EM converged after ' num2str(nbIter) ' iterations.']);
		    break;
	    end
    end
    nbIter
end
%%
model_cpu = gather(model);
%%
figure(5)
hold on
plot3(Data(1,:),Data(3,:),Data(5,:),'color',"#727272");  
plotGMM3D(model.Mu([1,3,5],:), model.Sigma([1,3,5],[1,3,5],:), [.8 0 0],.5); 
view(26,30)
grid on
hold off
%%
figure(4)
hold on
plot3(Data(2,:),Data(4,:),Data(6,:),'color',"#727272");  
plotGMM3D(model.Mu([2,4,6],:), model.Sigma([2,4,6],[2,4,6],:), [.8 0 0],.5); 
view(26,30)
grid on
hold off
%%
figure(2)
hold on
plot3(Data(1,:),Data(3,:),Data(5,:),'color',"#727272");  
plotGMM3D(model_KNN.Mu([1,3,5],:), model_KNN.Sigma([1,3,5],[1,3,5],:), [.8 0 0],.5); 
view(26,30)
grid on
hold off
%%
figure(3)
hold on
plot3(Data(2,:),Data(4,:),Data(6,:),'color',"#727272");  
plotGMM3D(model_KNN.Mu([2,4,6],:), model_KNN.Sigma([2,4,6],[2,4,6],:), [.8 0 0],.5); 
view(26,30)
grid on
hold off
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMR Process
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
t = 0:0.001:5;
t_con = 0:0.001:10;
y = trapmf(t,[1 2 3 4]);
y_add = trapmf(t, [0 2 4 5]);
y_com = cat(2, y, y_add(1:5000));
trap_traj = cat(2, t_con', y_com');

%%
simout02=sim('TDoFs_mass_CL_val.slx');
trq_in_val=simout02.ScopeData2{1}.Values.Data;
angle_out_val=simout02.ScopeData{1}.Values.Data;
vel_out_val=simout02.ScopeData{2}.Values.Data;
acc_out_val=simout02.ScopeData{3}.Values.Data;
%%
trq_in_val=simout.ScopeData2{1}.Values.Data;
angle_out_val=simout.ScopeData{1}.Values.Data;
vel_out_val=simout.ScopeData{2}.Values.Data;
acc_out_val=simout.ScopeData{3}.Values.Data;
%%
angle_out_J01_val = angle_out_val(1000:10001,1);
angle_out_J02_val = angle_out_val(1000:10001,2);
acc_out_J01_val = acc_out_val(1000:10001,1);
acc_out_J02_val = acc_out_val(1000:10001,2);
trq_in_J01_val = trq_in_val(1000:10001,1);
trq_in_J02_val = trq_in_val(1000:10001,2);
%%
% % 1번 파트(GMM)에서 사용했던 Data와 계산된 Model 사용
% % load("gmm_examples_data.mat")
% sampling_time = 0.001;
% f_cut = 50;
% % DataIn: 독립변수, DataOut: GMR을 사용해 추정된 종속변수 값
% % 이 예제에서는 데이터의 첫번째 행에 대응하는 값을 DataIn으로 지정
% DataIn_ang = -3:0.01:3;
% L = length(DataIn_ang);
% DataIn_vel = (zeros(L,1))';
% DataIn_acc = (zeros(L,1))';
% for i = 1:L-1
%     DataIn_vel(i+1) = -(DataIn_vel(i)*(pi*sampling_time*f_cut - 1) - 2*pi*f_cut*DataIn_ang(i+1) + 2*pi*f_cut*DataIn_ang(i))/(pi*sampling_time*f_cut + 1);
%     DataIn_acc(i+1) = -(DataIn_acc(i)*(pi*sampling_time*f_cut - 1) - 2*pi*f_cut*DataIn_vel(i+1) + 2*pi*f_cut*DataIn_vel(i))/(pi*sampling_time*f_cut + 1);
% end
DataIn_temp = cat(1, angle_out_J01_val', angle_out_J02_val', acc_out_J01_val', acc_out_J02_val');
DataIn = gpuArray(DataIn_temp);
% DataIn = cat(1, angle_out_fin', acc_out_fin');
%%
% load('D:\data_archive\Ref_Data\Jan_Peters\sarcos_inv_test.mat')
% %%
% Joint_Positions_test = sarcos_inv_test(:,1:7);
% Joint_Velocities_test = sarcos_inv_test(:,8:14);
% Joint_Accelerations_test = sarcos_inv_test(:,15:21);
% Joint_Torques_test = sarcos_inv_test(:,22:28);
% %%
% % n x m 행렬, n은 데이터의 차원, m은 데이터의 길이
% Data_temp = [Joint_Positions_test';Joint_Accelerations_test']; 
% DataIn = gpuArray(Data_temp);
%%
% Data 길이
nbData = size(DataIn,2);

% DataOut의 차원. 즉, 추정해야 할 변수의 개수
nbVarOut = 2;

% DataIn과 DataOut이 Data 행렬의 어떤 변수(몇 번째 행)에 해당하는지
% DataIn, DataOut이 각각 N, M개의 변수일때,
% in, out은 1 x N, 1 x M 배열
% in = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
% out = [15,16,17,18,19,20,21];
in = [1,2,3,4];
out = [5,6];

% 대각성분에 더해주는 작은 양의 스칼라 값
% Optional 한 값으로, 0으로 두어도 됨
model.params_diagRegFact = 1E-8;
%%
% 각 변수들 0으로 초기화
MuTmp = zeros(nbVarOut, model.nbStates);
expData = zeros(nbVarOut, nbData);
expSigma = zeros(nbVarOut, nbVarOut, nbData);

for t=1:nbData
    
	%Compute activation weight
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) .* (mvnpdf(DataIn(:,t)', model.Mu(in,i)', model.Sigma(in,in,i)'))';
    end
	H(:,t) = H(:,t) ./ sum(H(:,t)+realmin);

    %Compute conditional means
	for i=1:model.nbStates
		MuTmp(:,i) = model.Mu(out,i) + model.Sigma(out,in,i) / model.Sigma(in,in,i) * (DataIn(:,t)-model.Mu(in,i));
		expData(:,t) = expData(:,t) + H(i,t) .* MuTmp(:,i);
    end
    
    %Compute conditional covariances
	for i=1:model.nbStates
		SigmaTmp = model.Sigma(out,out,i) - model.Sigma(out,in,i) / model.Sigma(in,in,i) * model.Sigma(in,out,i);
		expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) .* (SigmaTmp + MuTmp(:,i) * MuTmp(:,i)');
	end
	expSigma(:,:,t) = expSigma(:,:,t) - expData(:,t) * expData(:,t)' + eye(nbVarOut) * model.params_diagRegFact; 
    t
end
%%
figure(6)
hold on;
% plot3(DataIn(1,:),DataIn(3,:),expData(1,:),'.','markersize',3,'color',[1.0 .0 .0]);
plot3(Data(1,:),Data(3,:),Data(5,:));
plot3(DataIn(1,:),DataIn(3,:),trq_in_J01_val,'.','markersize',3,'color',[0.0 1.0 .0]);
grid on
%%
figure(5)
hold on;
plot3(DataIn(2,:),DataIn(4,:),expData(2,:),'.','markersize',3,'color',[1.0 .0 .0]);
plot3(DataIn(2,:),DataIn(4,:),trq_in_J02_val,'.','markersize',3,'color',[0.0 1.0 .0]);
grid on
%%
figure(12)
hold on
plot(t,expData(2,:)','r', LineWidth=2);
plot(t,trq_in_J02_val', 'b', LineWidth=2);
title('Trapezoidal (Joint 2)', 'FontName','times new roman', 'FontSize',15)
xlabel('Time (sec)', 'FontName','times new roman', 'FontSize',15);
ylabel('Torque (Nm)','FontName','times new roman', 'FontSize',15); 
legend('\tau_{2,true}','\tau_{2,pred}','FontName','times new roman', 'FontSize',10)
xlim([1,10])
grid on
hold off
%%
t = 0.999:0.001:10;
figure(10)
hold on
plot(t,expData(1,:)','r',LineWidth=2);
plot(t,trq_in_J01_val', 'b', LineWidth=2);
title('Trapezoidal (Joint 1)', 'FontName','times new roman', 'FontSize',15)
xlabel('Time (sec)', 'FontName','times new roman', 'FontSize',15);
ylabel('Torque (Nm)','FontName','times new roman', 'FontSize',15); 
legend('\tau_{1,true}','\tau_{1,pred}','FontName','times new roman', 'FontSize',10)
xlim([1,10])
grid on
hold off
%%
figure(12)
hold on
plot(t,expData(2,:)','r', LineWidth=2);
plot(t,trq_in_J02_val', 'b', LineWidth=2);
title('Trapezoidal (Joint 2)', 'FontName','times new roman', 'FontSize',15)
xlabel('Time (sec)', 'FontName','times new roman', 'FontSize',15);
ylabel('Torque (Nm)','FontName','times new roman', 'FontSize',15); 
legend('\tau_{2,true}','\tau_{2,pred}','FontName','times new roman', 'FontSize',10)
xlim([1,10])
grid on
hold off
%%
t_tra = 0:0.001:10;
figure(17)
hold on
plot(t_tra,angle_out_val);
title('Multiple sinusoidal', 'FontName','times new roman', 'FontSize',15)
xlabel('Time (sec)', 'FontName','times new roman', 'FontSize',15);
ylabel('Joint position (rad)','FontName','times new roman', 'FontSize',15); 
legend('Joint 1','Joint 2','FontName','times new roman', 'FontSize',10)
% xlim([1,10])
grid on
hold off
%%
RMSE_J01 = sqrt(mean((trq_in_J01_val' - expData(1,:)).^2));
RMSE_J02 = sqrt(mean((trq_in_J02_val' - expData(2,:)).^2));
%%
MaxE_J01 = max(abs(trq_in_J01_val' - expData(1,:)));
MaxE_J02 = max(abs(trq_in_J02_val' - expData(2,:)));
%%
DataIn_trap = DataIn;
ExpData_trap = expData;
trq_in_J01_val_trap = trq_in_J01_val;
trq_in_J02_val_trap = trq_in_J02_val;
%%
DataIn_step = DataIn;
ExpData_step = expData;
trq_in_J01_val_step = trq_in_J01_val;
trq_in_J02_val_step = trq_in_J02_val;