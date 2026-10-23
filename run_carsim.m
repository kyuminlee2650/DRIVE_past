%% =========================================================================
% Deep Dynamics Model - 단일 CSV 평가 및 시각화 (튜닝 파라미터 적용)
% =========================================================================
clear vars;      
clear functions; 
clear all; close all;

% --- [가시성 및 폰트 설정] ---
font_all = 18; font_title = 24; font_rmse = 16; font_legend = 18;
lw_real = 2.5; lw_pinn = 2.5; lw_multi = 2.5;

%% 0. ⚙️ [하이퍼파라미터 튜닝 선언부]
tuning.seq_len        = 16;     % 시퀀스 길이
tuning.hidden_dim     = 256;    % 은닉층(Hidden) 차원
tuning.num_gru_layers = 1;      % GRU 레이어 개수
tuning.num_fc_layers  = 4;      % MLP 백본 내 ResidualBlock(FC) 반복 추가 횟수
tuning.act_type       = 3;      % 활성화 함수 (1: 'tanh', 2: 'gelu', 3: 'relu')

%% 1. 모델 가중치 및 테스트 데이터 로드
rehash;
model_data = load("C:\workspace\tuning_trial_19_best_model_carsim.mat");
data_path = "C:\workspace\carsim_processed_data_test_25hz\v40_new_nonzero1p65right.csv"; 

data = readmatrix(data_path, 'NumHeaderLines', 1);

%% 2. 차량 파라미터 및 데이터 매핑
params.dt = 0.04; 
params.m = 1270;
params.lf = 1.015; 
params.lr = 1.895;
params.seq_len = tuning.seq_len; % 튜닝 파라미터 연동

params.t_min = model_data.t_min;
params.t_max = model_data.t_max;
params.dr_min = model_data.dr_min;
params.dr_max = model_data.dr_max;
params.Iz_min = model_data.Iz_min;
params.Iz_max = model_data.Iz_max;

time = data(:, 1);             
pos_gt = data(:, 2:4);         
states_gt = data(:, 5:7);      
ctrls = data(:, 8:10);         
accels_gt = data(:, 11:12);    

%% 3. 시뮬레이션 수행
num_samples = size(data, 1) - params.seq_len - 2; 

preds_pinn_s = zeros(num_samples, 3); 
preds_pinn_p = zeros(num_samples, 3); 
preds_pinn_a = zeros(num_samples, 2);

preds_pinn_tf = zeros(num_samples, 6);
preds_pinn_tr = zeros(num_samples, 6);
preds_pinn_dr = zeros(num_samples, 5);
preds_pinn_Iz = zeros(num_samples, 1); 

gt_s_compare = zeros(num_samples, 3); 
gt_p_compare = zeros(num_samples, 3); 
gt_a_compare = zeros(num_samples, 2); 

horizon = 25; 
multi_count = 0; multi_idx_list = [];

fprintf('연산 중... \n');
for i = 1:num_samples
    idx = i + params.seq_len - 1; 
    
    curr_t = time(idx);
    curr_s = states_gt(idx, :); 
    curr_p = pos_gt(idx, :); 
    curr_c = ctrls(idx, :);
    next_c = ctrls(idx+1, :); 
    
    seq_pos = pos_gt(i:idx, :);
    seq_states = states_gt(i:idx, :);
    seq_ctrls = ctrls(i:idx, :);
    
    mid_idx = floor(params.seq_len / 2) + 1;
    ref_pos = seq_pos(mid_idx, :);
    
    loc_pos_seq = global_to_local_matlab(seq_pos, ref_pos);
    in_unnorm = [loc_pos_seq, seq_states, seq_ctrls]; 
    input_seq = (in_unnorm - model_data.in_mean) ./ (model_data.in_std + 1e-6);
    
    % --- 1. 1-Step 예측 ---
    [tf, tr, dr, Iz_pred, ~] = get_pinn_params_stateful(input_seq, model_data, params, tuning, []);
    
    preds_pinn_tf(i, :) = tf;
    preds_pinn_tr(i, :) = tr;
    preds_pinn_dr(i, :) = dr;
    preds_pinn_Iz(i)   = Iz_pred;
    
    [p_s_t1, p_p_t1, p_a_t] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz_pred, params);
    [~, p_p_t2, ~] = physics_step(p_s_t1, p_p_t1, next_c, tf, tr, dr, Iz_pred, params);
    
    preds_pinn_s(i, :) = p_s_t1; 
    preds_pinn_p(i, :) = p_p_t2; 
    preds_pinn_a(i, :) = p_a_t;
    
    gt_s_compare(i, :) = states_gt(idx+1, :);
    gt_p_compare(i, :) = pos_gt(idx+2, :);
    gt_a_compare(i, :) = accels_gt(idx, :);
    
    % --- 2. 25-Step Recursive 예측 ---
    if mod(i, 25) == 0 && curr_t >= 1.0 && curr_t <= time(end-100)
        multi_count = multi_count + 1;
        multi_idx_list(multi_count) = idx;
        
        ps_tmp = curr_s; 
        pp_tmp = curr_p; 
        
        p_m_s = zeros(horizon, 3); 
        p_m_p = zeros(horizon, 3); 
        p_m_a = zeros(horizon, 2);
        
        for h_step = 1:horizon
            f_idx = idx + h_step - 1; 
            if f_idx+1 > size(ctrls, 1), break; end
            f_ctrl = ctrls(f_idx, :);
            
            [ps_tmp, pp_tmp, pa_tmp] = physics_step(ps_tmp, pp_tmp, f_ctrl, tf, tr, dr, Iz_pred, params);
            
            p_m_s(h_step,:) = ps_tmp;
            p_m_p(h_step,:) = pp_tmp; 
            p_m_a(h_step,:) = pa_tmp;
        end
        
        multi_pinn(multi_count).s = p_m_s; 
        multi_pinn(multi_count).p = p_m_p; 
        multi_pinn(multi_count).a = p_m_a;
        multi_pinn(multi_count).t = curr_t + (1:horizon)*params.dt;
    end
end
t_eval = time(params.seq_len : params.seq_len + num_samples - 1);
fprintf('연산 완료!\n');

%% 4. RMSE 계산
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
    plot(t_eval, preds_pinn_s(:,k),'r','LineWidth',lw_pinn,'DisplayName','PINN');
    subtitle(sprintf('RMSE - PINN: %.4f', rmse_1_s_p(k)), 'FontSize', font_rmse); 
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
    plot(t_eval, preds_pinn_p(:,k),'r','LineWidth',lw_pinn,'DisplayName','PINN');
    subtitle(sprintf('RMSE - PINN: %.4f', rmse_1_p_p(k)), 'FontSize', font_rmse);
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
    plot(t_eval, preds_pinn_a(:,k),'r','LineWidth',lw_pinn,'DisplayName','PINN');
    subtitle(sprintf('RMSE - PINN: %.4f', rmse_1_a_p(k)), 'FontSize', font_rmse); 
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
            set(h1,'DisplayName','PINN'); 
        else
            set(h1,'HandleVisibility','off'); 
        end
    end
    subtitle(sprintf('Multi-step RMSE - PINN: %.4f', rmse_25_s_p(k)), 'FontSize', font_rmse);
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
            set(h1,'DisplayName','PINN'); 
        else
            set(h1,'HandleVisibility','off'); 
        end
    end
    subtitle(sprintf('Multi-step RMSE - PINN: %.4f', rmse_25_p_p(k)), 'FontSize', font_rmse);
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
            set(h1,'DisplayName','PINN'); 
        else
            set(h1,'HandleVisibility','off'); 
        end
    end
    subtitle(sprintf('Multi-step RMSE - PINN: %.4f', rmse_25_a_p(k)), 'FontSize', font_rmse);
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
figure('Color','w','Name','Fig 7: PINN Estimated tf Parameters','Position',[450,50,1100,850]);
tl_tf = tiledlayout(3,2,'TileSpacing','tight');
for k = 1:6
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(t_eval, preds_pinn_tf(:,k), 'b', 'LineWidth', 2);
    ylabel(sprintf('tf(%d)', k)); xlabel('Time [s]');
end
title(tl_tf, 'Figure 7. Estimated Front Tire Parameters (tf)', 'FontSize', font_title);

% 3. Rear Tire Parameters (tr)
figure('Color','w','Name','Fig 8: PINN Estimated tr Parameters','Position',[500,50,1100,850]);
tl_tr = tiledlayout(3,2,'TileSpacing','tight');
for k = 1:6
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(t_eval, preds_pinn_tr(:,k), 'b', 'LineWidth', 2);
    ylabel(sprintf('tr(%d)', k)); xlabel('Time [s]');
end
title(tl_tr, 'Figure 8. Estimated Rear Tire Parameters (tr)', 'FontSize', font_title);

% 4. Drivetrain Parameters (dr)
figure('Color','w','Name','Fig 9: PINN Estimated dr Parameters','Position',[550,50,1100,850]);
tl_dr = tiledlayout(3,2,'TileSpacing','tight');
for k = 1:5
    ax = nexttile; hold on; grid on; set(ax,'FontSize',font_all,'FontWeight','bold','LineWidth',1.2);
    plot(t_eval, preds_pinn_dr(:,k), 'b', 'LineWidth', 2);
    ylabel(sprintf('dr(%d)', k)); xlabel('Time [s]');
end
title(tl_dr, 'Figure 9. Estimated Drivetrain Parameters (dr)', 'FontSize', font_title);

%% 6. 전체 RMSE 결과 출력 
fprintf('\n==========================================================\n');
fprintf('       Deep Dynamics (PINN) Prediction Performance        \n');
fprintf('==========================================================\n');
fprintf(' [1-Step Prediction RMSE]\n');
fprintf('  - States (vx, vy, r)  : [%.6f, %.6f, %.6f]\n', rmse_1_s_p(1), rmse_1_s_p(2), rmse_1_s_p(3));
fprintf('  - Position (X, Y, psi): [%.6f, %.6f, %.6f]\n', rmse_1_p_p(1), rmse_1_p_p(2), rmse_1_p_p(3));
fprintf('  - Accel (ax, ay)      : [%.6f, %.6f]\n', rmse_1_a_p(1), rmse_1_a_p(2));
fprintf('----------------------------------------------------------\n');
fprintf(' [25-Step Recursive RMSE]\n');
fprintf('  - States (vx, vy, r)  : [%.6f, %.6f, %.6f]\n', rmse_25_s_p(1), rmse_25_s_p(2), rmse_25_s_p(3));
fprintf('  - Position (X, Y, psi): [%.6f, %.6f, %.6f]\n', rmse_25_p_p(1), rmse_25_p_p(2), rmse_25_p_p(3));
fprintf('  - Accel (ax, ay)      : [%.6f, %.6f]\n', rmse_25_a_p(1), rmse_25_a_p(2));
fprintf('==========================================================\n');

%% --- [함수부] ---

function loc_pos = global_to_local_matlab(pos_seq, ref_pos)
    dx = pos_seq(:, 1) - ref_pos(1); 
    dy = pos_seq(:, 2) - ref_pos(2); 
    dpsi = pos_seq(:, 3) - ref_pos(3);
    
    cos_p = cos(ref_pos(3)); sin_p = sin(ref_pos(3));
    loc_x = dx .* cos_p + dy .* sin_p; 
    loc_y = -dx .* sin_p + dy .* cos_p;
    
    loc_pos = [loc_x, loc_y, dpsi];
end

function [tf, tr, dr, Iz_pred, h_next] = get_pinn_params_stateful(input_data, md, p, tuning, h_prev)
    % 튜닝 변수 추출
    hidden_dim = tuning.hidden_dim;
    gru_layers = tuning.num_gru_layers;
    fc_layers  = tuning.num_fc_layers;
    act_type   = tuning.act_type;
    
    persistent W_ih b_ih W_hh b_hh W_mlp b_mlp W_out b_out Is_Initialized
    
    if isempty(Is_Initialized)
        % --- GRU 다중 레이어 가중치 로드 ---
        W_ih = cell(gru_layers, 1); b_ih = cell(gru_layers, 1);
        W_hh = cell(gru_layers, 1); b_hh = cell(gru_layers, 1);
        
        for l = 1:gru_layers
            suffix = sprintf('_l%d', l-1);
            W_ih{l} = md.(['gru_weight_ih', suffix]);
            W_hh{l} = md.(['gru_weight_hh', suffix]);
            b_ih{l} = md.(['gru_bias_ih', suffix])(:); 
            b_hh{l} = md.(['gru_bias_hh', suffix])(:);
        end
        
        % --- 가변 MLP Backbone 가중치 로드 ---
        W_mlp = cell(1 + fc_layers, 1);
        b_mlp = cell(1 + fc_layers, 1);
        
        W_mlp{1} = md.mlp_backbone_0_weight;
        b_mlp{1} = md.mlp_backbone_0_bias(:);
        
        for f = 1:fc_layers
            block_idx = 3*f; 
            W_mlp{1+f} = md.(sprintf('mlp_backbone_%d_weight', block_idx));
            b_mlp{1+f} = md.(sprintf('mlp_backbone_%d_bias', block_idx))(:);
        end
        
        W_out = md.output_layer_weight;
        b_out = md.output_layer_bias(:);
        
        Is_Initialized = true;
    end
    
    if isempty(h_prev)
        h = zeros(hidden_dim, gru_layers);
    else
        h = h_prev;
    end
    
    % --- GRU Forward Pass ---
    idx1 = hidden_dim; idx2 = 2 * hidden_dim; idx3 = 3 * hidden_dim;
    seq_len = size(input_data, 1);
    
    for t = 1:seq_len
        x_t = input_data(t, :)'; 
        for l = 1:gru_layers
            gi = W_ih{l} * x_t + b_ih{l}; 
            gh = W_hh{l} * h(:, l) + b_hh{l};
            
            r_gate = 1 ./ (1 + exp(-(gi(1:idx1) + gh(1:idx1))));
            z_gate = 1 ./ (1 + exp(-(gi(idx1+1:idx2) + gh(idx1+1:idx2))));
            n_gate = tanh(gi(idx2+1:idx3) + r_gate .* gh(idx2+1:idx3));
            
            h(:, l) = (1 - z_gate) .* n_gate + z_gate .* h(:, l);
            x_t = h(:, l); 
        end
    end
    h_next = h;
    
    % --- 가변 MLP Backbone Forward Pass ---
    x = h(:, end); 
    for i = 1:numel(W_mlp)
        x = W_mlp{i} * x + b_mlp{i};
        x = custom_activate(x, act_type);
    end
    
    % --- Output Layer & 18차원 파라미터 분할 추출 ---
    out_sig = 1 ./ (1 + exp(-(W_out * x + b_out)));
    
    tf = p.t_min + (p.t_max - p.t_min) .* out_sig(1:6)';
    tr = p.t_min + (p.t_max - p.t_min) .* out_sig(7:12)';
    dr = p.dr_min + (p.dr_max - p.dr_min) .* out_sig(13:17)'; 
    Iz_pred = p.Iz_min + (p.Iz_max - p.Iz_min) * out_sig(18); 
end

function out = custom_activate(in, act_type)
    if act_type == 1
        out = tanh(in);
    elseif act_type == 2
        out = 0.5 .* in .* (1 + tanh(sqrt(2/pi) .* (in + 0.044715 .* in.^3)));
    elseif act_type == 3
        out = max(0, in);
    else
        out = tanh(in);
    end
end

function [next_s, next_p, acc_body] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz, p)
    vx = curr_s(1); vy = curr_s(2); r = curr_s(3); 
    X = curr_p(1); Y = curr_p(2); psi = curr_p(3); 
    
    T = curr_c(1); brake = curr_c(2); delta = curr_c(3);
    
    Frx = (dr(1) - dr(2)*vx)*T - dr(5) * brake - dr(3) - dr(4)*vx^2;
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