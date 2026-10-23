%% =========================================================================
% Deep Dynamics Baseline - 단일 파일 검증 및 시각화 스크립트 (Stateful GRU)
% 특징: 7-Input, 17-Output (dr = 4차원), 25스텝 Recursive 파라미터 고정 평가
% =========================================================================
clear vars;      % 기존 변수 삭제
clear functions; % 함수 캐시 삭제
clear all; close all;

% --- [가시성 및 폰트 설정] ---
font_all = 18; font_title = 24; font_rmse = 16; font_legend = 18;
lw_real = 2.5; lw_pinn = 2.5; lw_multi = 2.5;

%% 1. 모델 가중치 및 테스트 데이터 로드
model_path = "best_weights_baseline_carsim.mat"; 
model_data = load(model_path);
data_path = "C:\workspace\carsim_processed_data_test_25hz\v40_new_nonzero1p65right.csv"; 

data = readmatrix(data_path, 'NumHeaderLines', 1);

%% 2. 차량 파라미터 및 데이터 매핑
params.dt = 0.04; % 25Hz 주기에 맞춤 (0.04s)
params.m = 1270;
params.lf = 1.015; params.lr = 1.895;
params.seq_len = 4; 

% 네트워크 출력 Boundary 파라미터 (총 17차원: tf(6), tr(6), dr(4), Iz(1))
params.t_min = model_data.t_min; params.t_max = model_data.t_max;
params.dr_min = model_data.dr_min; params.dr_max = model_data.dr_max;
params.Iz_min = model_data.Iz_min; params.Iz_max = model_data.Iz_max;

% 정규화 데이터
mean_vec = reshape(model_data.in_mean, 1, []);
std_vec  = reshape(model_data.in_std, 1, []);

% 데이터 파싱
time      = data(:, 1);             
pos_gt    = data(:, 2:4);      % X, Y, psi
states_gt = data(:, 5:7);      % vx, vy, r
ctrls     = data(:, 8:10);     % T, Brake, delta
accels_gt = data(:, 11:12);    % ax, ay

% 7차원 입력 피처 구성 (Baseline 방식)
T_raw = ctrls(:, 1);
d_raw = ctrls(:, 3);
dT = [0; diff(T_raw)];
d_d = [0; diff(d_raw)];
X_inputs = [states_gt, T_raw, d_raw, dT, d_d];

%% 3. 시뮬레이션 수행
num_samples = size(data, 1) - params.seq_len - 2; 
hidden = 64;

preds_pinn_s = zeros(num_samples, 3); 
preds_pinn_p = zeros(num_samples, 3); 
preds_pinn_a = zeros(num_samples, 2);

% --- 파라미터 저장을 위한 배열 초기화 ---
preds_pinn_tf = zeros(num_samples, 6);
preds_pinn_tr = zeros(num_samples, 6);
preds_pinn_dr = zeros(num_samples, 4); % Baseline: dr = 4차원
preds_pinn_Iz = zeros(num_samples, 1); 

gt_s_compare = zeros(num_samples, 3); 
gt_p_compare = zeros(num_samples, 3); 
gt_a_compare = zeros(num_samples, 2); 

horizon = 25; % 1초 예측
multi_count = 0; multi_idx_list = [];

% ADE / FDE 계산을 위한 임시 변수
total_ade_err = 0;
total_fde_err = 0;
total_25_sequences = 0;

fprintf('연산 중... \n');
for i = 1:num_samples
    idx = i + params.seq_len - 1; % 현재 시점 t
    
    curr_t = time(idx);
    curr_s = states_gt(idx, :); 
    curr_p = pos_gt(idx, :); 
    curr_c = ctrls(idx, :);
    next_c = ctrls(idx+1, :); 
    
    % 입력 시퀀스 및 파라미터 추론
    seq_input = X_inputs(i:idx, :); 
    input_norm = (seq_input - mean_vec) ./ (std_vec + 1e-6);
    [tf, tr, dr, Iz_pred, ~] = get_pinn_params_stateful(input_norm, model_data, params, hidden, []);
    
    % 추정된 파라미터 저장
    preds_pinn_tf(i, :) = tf;
    preds_pinn_tr(i, :) = tr;
    preds_pinn_dr(i, :) = dr;
    preds_pinn_Iz(i)   = Iz_pred;
    
    % 1-Step 물리 스텝 실행 (Position은 2-step 시점까지 예측)
    [p_s_t1, p_p_t1, p_a_t] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz_pred, params);
    [~, p_p_t2, ~] = physics_step(p_s_t1, p_p_t1, next_c, tf, tr, dr, Iz_pred, params);
    
    preds_pinn_s(i, :) = p_s_t1; 
    preds_pinn_p(i, :) = p_p_t2; 
    preds_pinn_a(i, :) = p_a_t;
    
    % GT 저장
    gt_s_compare(i, :) = states_gt(idx+1, :);
    gt_p_compare(i, :) = pos_gt(idx+2, :);
    gt_a_compare(i, :) = accels_gt(idx, :);
    
    % --- 2. 25-Step Recursive 예측 (파라미터 고정, 1초) ---
    if mod(i, 25) == 0 && curr_t >= 1.0 && curr_t <= time(end-100)
        multi_count = multi_count + 1;
        multi_idx_list(multi_count) = idx;
        
        ps_tmp = curr_s; 
        pp_tmp = curr_p; 
        
        p_m_s = zeros(horizon, 3); 
        p_m_p = zeros(horizon, 3); 
        p_m_a = zeros(horizon, 2);
        local_dist_err = zeros(horizon, 1);
        seq_completed = true;
        
        for h_step = 1:horizon
            f_idx = idx + h_step - 1; 
            if f_idx+1 > size(ctrls, 1)
                seq_completed = false;
                break; 
            end
            f_ctrl = ctrls(f_idx, :);
            
            [ps_tmp, pp_tmp, pa_tmp] = physics_step(ps_tmp, pp_tmp, f_ctrl, tf, tr, dr, Iz_pred, params);
            
            p_m_s(h_step,:) = ps_tmp;
            p_m_p(h_step,:) = pp_tmp; 
            p_m_a(h_step,:) = pa_tmp;
            
            % 글로벌 좌표 (X, Y) 기준 유클리드 거리 오차 계산
            gt_xy = pos_gt(f_idx+1, 1:2);
            pred_xy = pp_tmp(1:2);
            local_dist_err(h_step) = sqrt(sum((pred_xy - gt_xy).^2));
        end
        
        multi_pinn(multi_count).s = p_m_s; 
        multi_pinn(multi_count).p = p_m_p; 
        multi_pinn(multi_count).a = p_m_a;
        multi_pinn(multi_count).t = curr_t + (1:horizon)*params.dt;
        
        if seq_completed
            total_ade_err = total_ade_err + sum(local_dist_err);
            total_fde_err = total_fde_err + local_dist_err(end);
            total_25_sequences = total_25_sequences + 1;
        end
    end
end
t_eval = time(params.seq_len : params.seq_len + num_samples - 1);

%% 4. 성능 지표 계산 (RMSE, ADE, FDE)
rmse_1_s_p = sqrt(mean((preds_pinn_s - gt_s_compare).^2));
rmse_1_p_p = sqrt(mean((preds_pinn_p - gt_p_compare).^2));
rmse_1_a_p = sqrt(mean((preds_pinn_a - gt_a_compare).^2));

sq_err_25_s_p = zeros(1,3); 
sq_err_25_p_p = zeros(1,3); 
sq_err_25_a_p = zeros(1,2); 
total_25_pts = 0;

for st = 1:multi_count
    g_idx = multi_idx_list(st) + (1:horizon);
    if g_idx(end) > size(data,1), break; end
    
    sq_err_25_s_p = sq_err_25_s_p + sum((multi_pinn(st).s - states_gt(g_idx,:)).^2);
    sq_err_25_p_p = sq_err_25_p_p + sum((multi_pinn(st).p - pos_gt(g_idx,:)).^2);
    sq_err_25_a_p = sq_err_25_a_p + sum((multi_pinn(st).a - accels_gt(g_idx,:)).^2);
    
    total_25_pts = total_25_pts + horizon;
end
rmse_25_s_p = sqrt(sq_err_25_s_p / total_25_pts); 
rmse_25_p_p = sqrt(sq_err_25_p_p / total_25_pts); 
rmse_25_a_p = sqrt(sq_err_25_a_p / total_25_pts); 

avg_ade = total_ade_err / (total_25_sequences * horizon);
avg_fde = total_fde_err / total_25_sequences;

%% 5. 플롯 생성
st_names={'v_x [m/s]','v_y [m/s]','r [rad/s]'}; 
pos_names={'X [m]','Y [m]','\psi [rad]'}; 
acc_names={'a_x [m/s^2]','a_y [m/s^2]'};

% --- Fig 1: State 1-step ---
figure('Color','w','Name','Fig 1: State 1-step','Position',[50,50,1100,850]);
tl1 = tiledlayout(3,1,'TileSpacing','tight'); 
for k=1:3
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(t_eval, gt_s_compare(:,k),'k--','LineWidth',lw_real,'DisplayName','Real (GT)');
    plot(t_eval, preds_pinn_s(:,k),'r','LineWidth',lw_pinn,'DisplayName','Baseline');
    subtitle(sprintf('RMSE - Baseline: %.4f', rmse_1_s_p(k)), 'FontSize', font_rmse); 
    ylabel(st_names{k}); 
    if k==1
        legend('show','Orientation','horizontal','Location','northoutside','FontSize',font_legend); 
    end
end
title(tl1,'Figure 1. State (Velocity) 1-Step Prediction','FontSize',font_title);

% --- Fig 3: Position 1-step ---
figure('Color','w','Name','Fig 3: Position 1-step','Position',[150,50,1100,850]);
tl3 = tiledlayout(3,1,'TileSpacing','tight');
for k=1:3
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(t_eval, gt_p_compare(:,k),'k--','LineWidth',lw_real,'DisplayName','Real (GT)');
    plot(t_eval, preds_pinn_p(:,k),'r','LineWidth',lw_pinn,'DisplayName','Baseline');
    subtitle(sprintf('RMSE - Baseline: %.4f', rmse_1_p_p(k)), 'FontSize', font_rmse);
    ylabel(pos_names{k}); 
    if k==1
        legend('show','Orientation','horizontal','Location','northoutside','FontSize',font_legend); 
    end
end
title(tl3,'Figure 3. Position 1-Step Prediction','FontSize',font_title);

% --- Fig 5: Accel 1-step ---
figure('Color','w','Name','Fig 5: Accel 1-step','Position',[250,50,1100,850]);
tl5 = tiledlayout(2,1,'TileSpacing','tight');
for k=1:2
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(t_eval, gt_a_compare(:,k),'k--','LineWidth',lw_real,'DisplayName','Real (GT)');
    plot(t_eval, preds_pinn_a(:,k),'r','LineWidth',lw_pinn,'DisplayName','Baseline');
    subtitle(sprintf('RMSE - Baseline: %.4f', rmse_1_a_p(k)), 'FontSize', font_rmse); 
    ylabel(acc_names{k}); 
    if k==1
        legend('show','Orientation','horizontal','Location','northoutside','FontSize',font_legend); 
    end
end
title(tl5,'Figure 5. Acceleration 1-Step Prediction','FontSize',font_title);

% --- Fig 2: State 25-step ---
figure('Color','w','Name','Fig 2: State 25-step','Position',[100,50,1100,850]);
tl2 = tiledlayout(3,1,'TileSpacing','tight');
for k=1:3
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(time, states_gt(:,k),'k--','LineWidth',lw_real,'DisplayName','Real (GT)');
    for st=1:multi_count
        h1 = plot(multi_pinn(st).t, multi_pinn(st).s(:,k),'r','LineWidth',lw_multi); 
        if st==1
            set(h1,'DisplayName','Baseline'); 
        else
            set(h1,'HandleVisibility','off'); 
        end
    end
    subtitle(sprintf('Multi-step RMSE - Baseline: %.4f', rmse_25_s_p(k)), 'FontSize', font_rmse);
    ylabel(st_names{k}); 
    if k==1
        legend('show','Orientation','horizontal','Location','northoutside','FontSize',font_legend); 
    end
end
title(tl2,'Figure 2. State (Velocity) 25-Step Recursive Prediction','FontSize',font_title);

% --- Fig 4: Position 25-step ---
figure('Color','w','Name','Fig 4: Position 25-step','Position',[200,50,1100,850]);
tl4 = tiledlayout(3,1,'TileSpacing','tight');
for k=1:3
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(time, pos_gt(:,k),'k--','LineWidth',lw_real,'DisplayName','Real (GT)');
    for st=1:multi_count
        h1 = plot(multi_pinn(st).t, multi_pinn(st).p(:,k),'r','LineWidth',lw_multi); 
        if st==1
            set(h1,'DisplayName','Baseline'); 
        else
            set(h1,'HandleVisibility','off'); 
        end
    end
    subtitle(sprintf('Multi-step RMSE - Baseline: %.4f', rmse_25_p_p(k)), 'FontSize', font_rmse);
    ylabel(pos_names{k}); 
    if k==1
        legend('show','Orientation','horizontal','Location','northoutside','FontSize',font_legend); 
    end
end
title(tl4,'Figure 4. Position 25-Step Recursive Prediction','FontSize',font_title);

% --- Fig 6: Accel 25-step ---
figure('Color','w','Name','Fig 6: Accel 25-step','Position',[300,50,1100,850]);
tl6 = tiledlayout(2,1,'TileSpacing','tight');
for k=1:2
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(time, accels_gt(:,k),'k--','LineWidth',lw_real,'DisplayName','Real (GT)');
    for st=1:multi_count
        h1 = plot(multi_pinn(st).t, multi_pinn(st).a(:,k),'r','LineWidth',lw_multi); 
        if st==1
            set(h1,'DisplayName','Baseline'); 
        else
            set(h1,'HandleVisibility','off'); 
        end
    end
    subtitle(sprintf('Multi-step RMSE - Baseline: %.4f', rmse_25_a_p(k)), 'FontSize', font_rmse);
    ylabel(acc_names{k}); 
    if k==1
        legend('show','Orientation','horizontal','Location','northoutside','FontSize',font_legend); 
    end
end
title(tl6,'Figure 6. Acceleration 25-Step Recursive Prediction','FontSize',font_title);

% --- 파라미터(tf, tr, dr, Iz) 플롯 ---
% 1. Iz Parameter
figure('Color','w','Name','Variable Iz Prediction','Position',[400,50,1100,300]);
plot(t_eval, preds_pinn_Iz, 'b', 'LineWidth', 2); grid on;
title('Predicted Variable Iz over Time', 'FontSize', font_title);
ylabel('Iz [kg\cdotm^2]', 'FontSize', font_all); xlabel('Time [s]', 'FontSize', font_all);
set(gca, 'FontSize', font_all, 'FontWeight', 'bold', 'LineWidth', 1.2);

% 2. Front Tire Parameters (tf)
figure('Color','w','Name','Fig 7: Baseline Estimated tf Parameters','Position',[450,50,1100,850]);
tl_tf = tiledlayout(3,2,'TileSpacing','tight');
for k = 1:6
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(t_eval, preds_pinn_tf(:,k), 'b', 'LineWidth', 2);
    ylabel(sprintf('tf(%d)', k)); xlabel('Time [s]');
end
title(tl_tf, 'Figure 7. Estimated Front Tire Parameters (tf)', 'FontSize', font_title);

% 3. Rear Tire Parameters (tr)
figure('Color','w','Name','Fig 8: Baseline Estimated tr Parameters','Position',[500,50,1100,850]);
tl_tr = tiledlayout(3,2,'TileSpacing','tight');
for k = 1:6
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(t_eval, preds_pinn_tr(:,k), 'b', 'LineWidth', 2);
    ylabel(sprintf('tr(%d)', k)); xlabel('Time [s]');
end
title(tl_tr, 'Figure 8. Estimated Rear Tire Parameters (tr)', 'FontSize', font_title);

% 4. Drivetrain Parameters (dr) -> 💡 4개 파라미터이므로 2x2 레이아웃 적용
figure('Color','w','Name','Fig 9: Baseline Estimated dr Parameters','Position',[550,50,1100,850]);
tl_dr = tiledlayout(2,2,'TileSpacing','tight');
for k = 1:4
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(t_eval, preds_pinn_dr(:,k), 'b', 'LineWidth', 2);
    ylabel(sprintf('dr(%d)', k)); xlabel('Time [s]');
end
title(tl_dr, 'Figure 9. Estimated Drivetrain Parameters (dr)', 'FontSize', font_title);

%% 6. 전체 성능 결과 출력 
fprintf('\n==========================================================\n');
fprintf('         Deep Dynamics Baseline Prediction Performance    \n');
fprintf('==========================================================\n');
fprintf(' [1-Step Prediction RMSE]\n');
fprintf('  - States (vx, vy, r)  : [%.6f, %.6f, %.6f]\n', rmse_1_s_p(1), rmse_1_s_p(2), rmse_1_s_p(3));
fprintf('  - Position (X, Y, psi): [%.6f, %.6f, %.6f]\n', rmse_1_p_p(1), rmse_1_p_p(2), rmse_1_p_p(3));
fprintf('  - Accel (ax, ay)      : [%.6f, %.6f]\n', rmse_1_a_p(1), rmse_1_a_p(2));
fprintf('----------------------------------------------------------\n');
fprintf(' [25-Step Recursive Average Performance]\n');
fprintf('  - States (vx, vy, r)  : [%.6f, %.6f, %.6f]\n', rmse_25_s_p(1), rmse_25_s_p(2), rmse_25_s_p(3));
fprintf('  - Position (X, Y, psi): [%.6f, %.6f, %.6f]\n', rmse_25_p_p(1), rmse_25_p_p(2), rmse_25_p_p(3));
fprintf('  - Accel (ax, ay)      : [%.6f, %.6f]\n', rmse_25_a_p(1), rmse_25_a_p(2));
fprintf('  - Trajectory Error    : ADE = %.6f m, FDE = %.6f m\n', avg_ade, avg_fde);
fprintf('==========================================================\n');

%% --- [함수부] ---
function [tf, tr, dr, Iz_pred, h_next] = get_pinn_params_stateful(input_data, md, p, hidden_dim, h_prev)
    persistent W_ih W_hh b_ih b_hh W_mlp b_mlp W_res b_res W_out b_out
    if isempty(W_ih)
        W_ih = md.gru_weight_ih_l0; W_hh = md.gru_weight_hh_l0;
        b_ih = md.gru_bias_ih_l0';  b_hh = md.gru_bias_hh_l0';
        W_mlp = md.mlp_backbone_0_weight; b_mlp = md.mlp_backbone_0_bias';
        W_res = md.mlp_backbone_3_block_0_weight; b_res = md.mlp_backbone_3_block_0_bias';
        W_out = md.output_layer_weight; b_out = md.output_layer_bias';
    end
    h = zeros(hidden_dim, 1);
    idx1 = hidden_dim; idx2 = 2 * hidden_dim; idx3 = 3 * hidden_dim;
    for t = 1:size(input_data, 1)
        x_t = input_data(t, :)'; gi = W_ih * x_t + b_ih; gh = W_hh * h + b_hh;
        r_gate = 1 ./ (1 + exp(-(gi(1:idx1) + gh(1:idx1))));
        z_gate = 1 ./ (1 + exp(-(gi(idx1+1:idx2) + gh(idx1+1:idx2))));
        n_gate = tanh(gi(idx2+1:idx3) + r_gate .* gh(idx2+1:idx3));
        h = (1 - z_gate) .* n_gate + z_gate .* h;
    end
    h_next = h;
    x = tanh(W_mlp * h + b_mlp); x = tanh(W_res * x + b_res);
    out_sig = 1 ./ (1 + exp(-(W_out * x + b_out)));
    tf = p.t_min + (p.t_max - p.t_min) .* out_sig(1:6)';
    tr = p.t_min + (p.t_max - p.t_min) .* out_sig(7:12)';
    dr = p.dr_min + (p.dr_max - p.dr_min) .* out_sig(13:16)'; % dr = 4차원 파싱
    Iz_pred = p.Iz_min + (p.Iz_max - p.Iz_min) * out_sig(17);  % Iz = 17번째 인덱스
end

function [next_s, next_p, acc_body] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz, p)
    vx = curr_s(1); vy = curr_s(2); r = curr_s(3); X = curr_p(1); Y = curr_p(2); psi = curr_p(3); 
    T = curr_c(1); delta = curr_c(3);
    Frx = (dr(1) - dr(2)*vx)*T - dr(3) - dr(4)*(vx^2); % Baseline의 4파라미터 구동계 모델 수식
    af = delta - atan2(r*p.lf + vy, vx + 1e-6) + tf(5); 
    ar = atan2(r*p.lr - vy, vx + 1e-6) + tr(5);
    Ffy = tf(6) + tf(3)*sin(tf(2)*atan(tf(1)*af - tf(4)*(tf(1)*af - atan(tf(1)*af))));
    Fry = tr(6) + tr(3)*sin(tr(2)*atan(tr(1)*ar - tr(4)*(tr(1)*ar - atan(tr(1)*ar))));
    vx_dot = (- Ffy*sin(delta) + Frx)/p.m + vy*r; 
    vy_dot = (  Ffy*cos(delta) + Fry)/p.m - vx*r;
    r_dot = (Ffy*cos(delta)*p.lf - Fry*p.lr) / Iz;
    next_s = [vx + vx_dot*p.dt, vy + vy_dot*p.dt, r + r_dot*p.dt];
    next_p = [X + (vx*cos(psi) - vy*sin(psi))*p.dt, Y + (vx*sin(psi) + vy*cos(psi))*p.dt, psi + r*p.dt];
    acc_body = [vx_dot - vy*r, vy_dot + vx*r];
end