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
load('D:\data_archive\Trajectory_opt_data\GMM_test\230821_5EA_Cost01_m2_p2_0_err0_05.mat','q_opt')
q_opt_temp = zeros(length(q_opt)/10,1);
for i = 1:length(q_opt)/10
    q_opt_temp(i) = q_opt(10*i);
end
%%
figure(1)
plot(q_opt);
grid on
figure(2)
plot(q_opt_temp);
grid on
%%
q_opt_fin = cat(1,q_opt_temp,q_opt_temp);
for i = 1:8
    q_opt_fin = cat(1,q_opt_fin,q_opt_temp);
end
%%
figure(3)
plot(q_opt_fin);
grid on
%%
t_con = 0:0.001:9.999;
q_opt_tw = cat(2, t_con', q_opt_fin);
%%
% 데이터 로드
simout=sim('TDoFs_mass_CL_Trajopt.slx');
trq_in=simout.ScopeData2{1}.Values.Data;
angle_out=simout.ScopeData{1}.Values.Data;
vel_out=simout.ScopeData{2}.Values.Data;
acc_out=simout.ScopeData{3}.Values.Data;
%%
angle_out_J01 = angle_out(1:10001,1);
angle_out_J02 = angle_out(1:10001,2);
acc_out_J01 = acc_out(1:10001,1);
acc_out_J02 = acc_out(1:10001,2);
trq_in_J01 = trq_in(1:10001,1);
trq_in_J02 = trq_in(1:10001,2);
%%
% n x m 행렬, n은 데이터의 차원, m은 데이터의 길이
Data_temp_01 = [angle_out_J02';acc_out_J01';acc_out_J02';trq_in_J01']; 
Data_01 = gpuArray(Data_temp_01);
%%
% d_new = fillmissing(d,'next');
% 
% % Data = [angle_out';acc_out';trq_in'];    
% Data = [angle_out';d_new]; 
% Data = [angle_out';d];
size(Data_01);

% 데이터의 길이 (데이터 포인트의 개수)
nbData = size(Data_01,2);
% nbData_noisy = size(Data,2);
% 데이터의 차원
model.nbVar = size(Data_01,1); %Number of variables [x1,x2]
% model_noisy.nbVar = size(Data_noisy,1) %Number of variables [x1,x2]
%% 1-2. GMM setup
% 1) Gaussian distribution 개수 결정
% 2) GMM parameter initialization
%

model.nbStates = 3*2;
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

model = init_GMM_kmeans(Data_01, model);
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
Data = Data_01;
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
[idx,Center] = kmeans(q_opt,4);
%%
figure(2)
hold on
plot3(Data(1,:),Data(3,:),Data(4,:),'color',"#727272");  
plotGMM3D(model.Mu([1,3,4],:), model.Sigma([1,3,4],[1,3,4],:), [.8 0 0],.5); 
view(26,30)
grid on
hold off
%%
figure(3)
hold on
plot(theta,output,'b');  
plotGMM(model.Mu, model.Sigma, [.8 0 0],.5); 
grid on
hold off
%%
figure(4)
hold on
plot(theta_opt,output_opt,'b');  
plotGMM(model.Mu, model.Sigma, [.8 0 0],.5); 
grid on
hold off
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
% DataIn_temp = cat(1, angle_out_J02',acc_out_J01',acc_out_J02');
% DataIn_temp = cat(1, theta_opt');
DataIn_temp = cat(1, angle_out_J02_val',acc_out_J01_val',acc_out_J02_val');
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
in = [1,2,3];
out = [4];

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
figure(4)
hold on;
plot3(DataIn(1,:),DataIn(3,:),expData,'.','markersize',3,'color',[1.0 .0 .0]);
plot3(DataIn(1,:),DataIn(3,:),trq_in_J01_val,'.','markersize',3,'color',[0.0 1.0 .0]);
grid on
%%
figure(10)
hold on
plot(trq_in_J01_val, 'b');
plot(expData(1,:),'r');
grid on
hold off
%%
figure(9)
hold on
plot(DataIn(1,:));
plot(DataIn(2,:));
hold off
grid on
figure(10)
hold on
plot(DataIn(8,:));
plot(DataIn(9,:));
hold off
grid on
%%
figure(8)
hold on
plot(Joint_Torques_test(:,1));
plot(expData(1,:));
hold off
grid on
%%
err_torq = Joint_Torques_test(:,1)' - expData(1,:);

figure(6)
plot(err_torq,'b');
grid on
%%
err_torq01 = trq_in_J01_val' - expData;
figure(6)
plot(err_torq01,'b');
grid on
%%
err_torq02 = trq_in_J02_val' - expData(2,:);

figure(7)
plot(err_torq02,'b');
grid on
%% FFT
fft_torque = fft(Joint_Torques_test(:,1));
L = 4449;
Fs = 100;
P2 = abs(fft_torque/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) 
xlabel("f (Hz)")
ylabel("|P1(f)|")
grid on