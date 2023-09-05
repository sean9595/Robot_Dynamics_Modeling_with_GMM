%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc; clear; 
addpath('C:\Users\User\OneDrive - 대구경북과학기술원\바탕 화면\KTH_workspace\SIMULATION\Dynamics_Modeling\GMM\m_fcts\'); 
%%
% 데이터 로드
% run('C:\Users\User\OneDrive - 대구경북과학기술원\바탕 화면\KTH_workspace\SIMULATION\Trajectory_optimization\Trajectory_Optimization_for_GMM_Dynamics_Modeling\Traj_opt_for_GMM_find_initvalue_Reduced_Param_TDoF.m');
load('D:\data_archive\GMM_Dynamics_w_TrajOpt\230904\M_11.mat','model_cpu','theta_opt','output_opt')
model_M_11 = model_cpu;
theta_opt_M_11 = theta_opt;
output_opt_M_11 = output_opt;
load('D:\data_archive\GMM_Dynamics_w_TrajOpt\230904\M_12.mat','model_cpu','theta_opt','output_opt')
model_M_12 = model_cpu;
theta_opt_M_12 = theta_opt;
output_opt_M_12 = output_opt;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMR Process
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
simout02=sim('C:\Users\User\OneDrive - 대구경북과학기술원\바탕 화면\KTH_workspace\SIMULATION\Dynamics_Modeling\GMM\TDoFs_mass_CL_val.slx');
trq_in_val=simout02.ScopeData2{1}.Values.Data;
angle_out_val=simout02.ScopeData{1}.Values.Data;
vel_out_val=simout02.ScopeData{2}.Values.Data;
acc_out_val=simout02.ScopeData{3}.Values.Data;
%% angle_out_J02';acc_out_J01';acc_out_J02';trq_in_J01'
angle_out_J02_val = angle_out_val(500:10001,2);
acc_out_J01_val = acc_out_val(500:10001,1);
acc_out_J02_val = acc_out_val(500:10001,2);
trq_in_J01_val = trq_in_val(500:10001,1);
trq_in_J02_val = trq_in_val(500:10001,2);
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
% DataIn_temp = cat(1, angle_out_J01_val', acc_out_J01_val');
DataIn_temp = cat(1, angle_out_J02_val');
DataIn = gpuArray(DataIn_temp);
% DataIn = cat(1, angle_out_fin', acc_out_fin');
%%
% Data 길이
nbData = size(DataIn,2);

% DataOut의 차원. 즉, 추정해야 할 변수의 개수
nbVarOut = 1;

% DataIn과 DataOut이 Data 행렬의 어떤 변수(몇 번째 행)에 해당하는지
% DataIn, DataOut이 각각 N, M개의 변수일때,
% in, out은 1 x N, 1 x M 배열
% in = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
% out = [15,16,17,18,19,20,21];
in = [1];
out = [2];

% 대각성분에 더해주는 작은 양의 스칼라 값
% Optional 한 값으로, 0으로 두어도 됨
model_M_11.params_diagRegFact = 1E-8;
model_M_12.params_diagRegFact = 1E-8;
%%
% 각 변수들 0으로 초기화
model = model_M_11;
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
M_11_pred = expData;
%%
% 각 변수들 0으로 초기화
model = model_M_12;
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
M_12_pred = expData;
%%
Torq_1_pred = zeros(length(M_12_pred),1);
Torq_2_pred = zeros(length(M_12_pred),1);
for i=1:length(M_12_pred)
    Torq_1_pred(i) = M_11_pred(i)*acc_out_J01_val(i)+M_12_pred(i)*acc_out_J02_val(i);
    Torq_2_pred(i) = M_12_pred(i)*acc_out_J01_val(i)+1.00563*acc_out_J02_val(i);
end
%%
% figure(10)
subplot(2,1,1)
hold on
plot(trq_in_J01_val, 'b');
plot(Torq_1_pred,'r');
grid on
hold off
title('Joint 1 Torque')
xlabel('Time [sec]')
ylabel('\tau_1 [Nm]')

subplot(2,1,2)
hold on
plot(trq_in_J02_val, 'b');
plot(Torq_2_pred,'r');
grid on
hold off
title('Joint 2 Torque')
xlabel('Time [sec]')
ylabel('\tau_2 [Nm]')
%%
err_torq_1 = Torq_1_pred - trq_in_J01_val;
err_torq_2 = Torq_2_pred - trq_in_J02_val;
%%
figure(2)
subplot(2,1,1)
plot(err_torq_1, 'b');
grid on
title('Joint 1 Torque error')
xlabel('Time [sec]')
ylabel('\tau_1 error [Nm]')

subplot(2,1,2)
plot(err_torq_2, 'b');
grid on
title('Joint 2 Torque error')
xlabel('Time [sec]')
ylabel('\tau_2 error [Nm]')